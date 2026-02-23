import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from RAG_utils.rag_pipeline import build_and_store, count_records, query_similar
from llm_utils.gemini_utils import generate_answer_gemini_llm
from config import (
    MATHS_ASSETS_DIR,
    MATHS_SOURCE_MAIN_FILE,
    RAG_CHUNK_MAX_CHARS,
    RAG_CHUNK_OVERLAP,
    RAG_EMBEDDING_DIM,
    RAG_TABLE_NAME,
    RAG_TOP_K_DEFAULT,
)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:  # pragma: no cover
    letter = None
    getSampleStyleSheet = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    ListFlowable = None
    ListItem = None

router = APIRouter()


class MCQ(BaseModel):
    question: str = ""
    options: List[str] = Field(default_factory=list)
    answer: str = ""
    explanation: str = ""


class LessonPlanRequest(BaseModel):
    topic: str = Field(..., min_length=2, description="Main topic title")
    description: str = Field(default="", description="Detailed topic description")
    objectives: List[str] = Field(default_factory=list)
    duration_minutes: int = Field(default=40, ge=20, le=180)

    subject: str = "CBC Mathematics"
    grade: str = "Grade 10"
    top_k: int = Field(default=RAG_TOP_K_DEFAULT, ge=1, le=20)
    exercises_auto: bool = True
    manual_exercises: List[MCQ] = Field(default_factory=list)


class LessonPlanResponse(BaseModel):
    plan: Dict[str, Any]
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    raw_text: Optional[str] = None


class LessonPlanPDFRequest(BaseModel):
    plan: Dict[str, Any]


class IndexTextbookRequest(BaseModel):
    force_reindex: bool = False
    max_chars: int = Field(default=RAG_CHUNK_MAX_CHARS, ge=500, le=6000)
    overlap: int = Field(default=RAG_CHUNK_OVERLAP, ge=0, le=800)
    main_file: Optional[str] = None
    assets_dir: Optional[str] = None


class IndexTextbookResponse(BaseModel):
    indexed: bool
    record_count: int
    table_name: str
    force_reindex: bool = False
    message: str = ""


class LessonPlannerStatusResponse(BaseModel):
    indexed: bool
    record_count: int
    table_name: str


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_option_text(value: Any) -> str:
    option = str(value or "").strip()
    option = re.sub(r"^\s*[-*]\s*", "", option)
    # Remove repeated labels like "A. A. ..." or "Option B: ..."
    label_pattern = re.compile(r"^\s*(?:Option\s*)?(?:[A-Da-d]|[1-4])[\)\.\:\-]\s*", re.I)
    while True:
        updated = label_pattern.sub("", option, count=1)
        if updated == option:
            break
        option = updated.strip()
    return option.strip()


def _normalize_answer_letter(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return "A"

    m = re.search(r"\b([ABCD])\b", raw)
    if m:
        return m.group(1)

    m = re.search(r"\b([1-4])\b", raw)
    if m:
        return chr(64 + int(m.group(1)))

    if raw[0] in {"A", "B", "C", "D"}:
        return raw[0]
    return "A"


def _normalize_manual_exercises(exercises: List[MCQ]) -> List[Dict[str, Any]]:
    normalized = []
    for mcq in exercises:
        options = [_normalize_option_text(opt) for opt in mcq.options][:4]
        if len(options) < 4:
            options.extend([""] * (4 - len(options)))
        normalized.append(
            {
                "question": mcq.question.strip(),
                "options": options,
                "answer": _normalize_answer_letter(mcq.answer),
                "explanation": mcq.explanation.strip(),
            }
        )
    return normalized


def _normalize_presentation_text(text: Any) -> str:
    raw = str(text or "").replace("\r\n", "\n").strip()
    if not raw:
        return ""

    raw = raw.replace("**", "").replace("__", "")
    raw = re.sub(r"^\s*#{1,6}\s*", "", raw, flags=re.M)
    raw = re.sub(r"^\s*[-*]\s+", "", raw, flags=re.M)
    raw = re.sub(r"^\s*(Teacher|Tutor|Instructor)\s*:\s*", "", raw, flags=re.M)

    phrase_replacements = [
        (r"\b[Bb]egin by asking (?:learners|students|the class|class) to\b", ""),
        (r"\b[Ss]tart by asking (?:learners|students|the class|class) to\b", ""),
        (r"\b[Aa]sk (?:learners|students|the class|class) to\b", ""),
        (r"\b[Tt]ell (?:learners|students|the class|class) to\b", ""),
        (r"\b[Ee]ncourage (?:learners|students|the class|class) to\b", ""),
        (r"\b[Tt]eacher note\s*:\s*", ""),
        (r"\b[Pp]resenter note\s*:\s*", ""),
    ]
    for pattern, replacement in phrase_replacements:
        raw = re.sub(pattern, replacement, raw)

    raw = re.sub(r"\b[Ll]earners\b", "us", raw)
    raw = re.sub(r"\b[Ss]tudents\b", "us", raw)
    raw = re.sub(r"\b[Cc]lass\b", "us", raw)
    raw = re.sub(r"\b[Tt]heir\b", "our", raw)
    raw = re.sub(r"\b[Tt]hem\b", "us", raw)

    raw = re.sub(r"\s*(Step\s*\d+\s*:?)", r"\n\1", raw, flags=re.I)
    raw = re.sub(
        r"\s*(Concept:|Example:|Try:|Practice:|Check:|Formula:|Why it works:|Think:)",
        r"\n\1",
        raw,
        flags=re.I,
    )
    raw = re.sub(r"\s*(\d+\.)\s+", r"\n\1 ", raw)

    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        line = re.sub(r"^to\s+", "", line, flags=re.I)
        if line:
            lines.append(line)

    text_out = "\n".join(lines)
    text_out = re.sub(r"[ \t]{2,}", " ", text_out)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out)
    return text_out.strip()


def _build_prompt(req: LessonPlanRequest, sources: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for i, row in enumerate(sources, start=1):
        meta = row.get("metadata") or {}
        title = " > ".join(
            [p for p in [meta.get("chapter"), meta.get("section"), meta.get("subsection")] if p]
        )
        context_blocks.append(f"[{i}] {title}\n{row.get('content', '')}")

    context_text = "\n\n".join(context_blocks)
    objectives_text = "\n".join(f"- {item}" for item in req.objectives) if req.objectives else "-"
    manual_exercises = _normalize_manual_exercises(req.manual_exercises)

    return (
        "You are an expert Grade 10 mathematics teacher creating slides-ready instructional content.\n"
        "Write content that can be projected and explained directly to students in class.\n"
        "Use textbook context as the base source of truth.\n"
        "If context is limited, stay practical and avoid made-up textbook facts.\n\n"
        "Return ONLY valid JSON using this exact schema:\n"
        "{"
        '"title": string, '
        '"subject": string, '
        '"grade": string, '
        '"topic": string, '
        '"duration_minutes": integer, '
        '"overview": string, '
        '"objectives": [string], '
        '"lesson_flow": [{"title": string, "time_minutes": integer, "content": string}], '
        '"mcq_exercises": [{"question": string, "options": [string,string,string,string], "answer": "A|B|C|D", "explanation": string}], '
        '"resources": [string], '
        '"presentation_notes": string'
        "}\n\n"
        "Constraints:\n"
        f"- duration_minutes must be exactly {req.duration_minutes}\n"
        "- sum of lesson_flow.time_minutes must equal duration_minutes\n"
        "- keep language simple and engaging for Grade 10\n"
        "- use direct presentation language with we/us/our framing\n"
        "- do NOT write teacher-instruction phrases like 'Begin by asking learners to ...'\n"
        "- do NOT use dialogue format or speaker tags\n"
        "- do NOT use markdown symbols like **, ###, or bullet markdown in content\n"
        "- do NOT use words like teacher/tutor/instructor in the output text\n"
        "- for each lesson_flow.content, keep related sentences in one line and add line breaks only when context changes (Step, Concept, Example, Try, Practice, Check, Formula, Why it works, Think)\n"
        "- include formula lines where needed using plain text (example: Formula: a^2 + b^2 = c^2)\n"
        "- MCQ options must always be exactly 4 items\n"
        "- if exercises_auto is false, use manual_exercises exactly as given\n\n"
        f"Subject: {req.subject}\n"
        f"Grade: {req.grade}\n"
        f"Topic: {req.topic}\n"
        f"Detailed Topic Description: {req.description}\n"
        f"Objectives:\n{objectives_text}\n\n"
        f"exercises_auto: {str(req.exercises_auto).lower()}\n"
        f"manual_exercises: {json.dumps(manual_exercises, ensure_ascii=True)}\n\n"
        f"Textbook context:\n{context_text}\n\n"
        "JSON:"
    )


def _normalize_plan(req: LessonPlanRequest, plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(plan or {})
    plan.pop("image_ids", None)
    plan.pop("textbook_images", None)

    plan.setdefault("title", f"Lesson Plan: {req.topic}")
    plan.setdefault("subject", req.subject)
    plan.setdefault("grade", req.grade)
    plan.setdefault("topic", req.topic)
    plan.setdefault("duration_minutes", req.duration_minutes)
    plan.setdefault("overview", "")
    plan.setdefault("objectives", req.objectives)
    plan.setdefault("lesson_flow", [])
    plan.setdefault("mcq_exercises", [])
    plan.setdefault("resources", [])
    plan.setdefault("presentation_notes", "")

    plan["overview"] = _normalize_presentation_text(plan.get("overview", ""))
    legacy_notes = plan.pop("notes_for_teacher", "")
    plan["presentation_notes"] = _normalize_presentation_text(
        plan.get("presentation_notes", "") or legacy_notes
    )

    if not req.exercises_auto and req.manual_exercises:
        plan["mcq_exercises"] = _normalize_manual_exercises(req.manual_exercises)

    clean_flow = []
    for step in plan.get("lesson_flow", []) or []:
        if not isinstance(step, dict):
            continue

        clean_flow.append(
            {
                "title": str(step.get("title", "")).strip(),
                "time_minutes": _to_int(step.get("time_minutes", 0), default=0),
                "content": _normalize_presentation_text(step.get("content", "")),
            }
        )
    plan["lesson_flow"] = clean_flow

    clean_mcq = []
    for mcq in plan.get("mcq_exercises", []) or []:
        if not isinstance(mcq, dict):
            continue
        options = list(mcq.get("options", []))[:4]
        if len(options) < 4:
            options.extend([""] * (4 - len(options)))
        clean_mcq.append(
            {
                "question": str(mcq.get("question", "")).strip(),
                "options": [_normalize_option_text(o) for o in options],
                "answer": _normalize_answer_letter(mcq.get("answer", "A")),
                "explanation": _normalize_presentation_text(mcq.get("explanation", "")),
            }
        )
    plan["mcq_exercises"] = clean_mcq

    clean_resources = []
    for resource in plan.get("resources", []) or []:
        if isinstance(resource, dict):
            title = str(resource.get("title", "") or resource.get("name", "")).strip()
            detail = str(resource.get("description", "") or resource.get("detail", "")).strip()
            merged = " - ".join([item for item in [title, detail] if item])
            if merged:
                clean_resources.append(merged)
        else:
            text = str(resource).strip()
            if text:
                clean_resources.append(text)

    plan["resources"] = clean_resources
    plan["objectives"] = [str(o).strip() for o in (plan.get("objectives", []) or []) if str(o).strip()]
    return plan


def _resolve_paths(main_file: Optional[str], assets_dir: Optional[str]) -> tuple[Path, Path]:
    resolved_main_file = Path(main_file).expanduser().resolve() if main_file else Path(MATHS_SOURCE_MAIN_FILE)
    resolved_assets_dir = Path(assets_dir).expanduser().resolve() if assets_dir else Path(MATHS_ASSETS_DIR)
    if not resolved_main_file.exists():
        raise HTTPException(status_code=400, detail=f"PreTeXt main file not found: {resolved_main_file}")
    if not resolved_assets_dir.exists():
        raise HTTPException(status_code=400, detail=f"Assets directory not found: {resolved_assets_dir}")
    return resolved_main_file, resolved_assets_dir


@router.get("/lesson_planner/status", response_model=LessonPlannerStatusResponse)
async def lesson_planner_status():
    count = count_records(table_name=RAG_TABLE_NAME)
    return LessonPlannerStatusResponse(indexed=count > 0, record_count=count, table_name=RAG_TABLE_NAME)


@router.post("/lesson_planner/index_textbook", response_model=IndexTextbookResponse)
async def index_textbook(req: IndexTextbookRequest):
    existing_count = count_records(table_name=RAG_TABLE_NAME)
    if existing_count > 0 and not req.force_reindex:
        return IndexTextbookResponse(
            indexed=True,
            record_count=existing_count,
            table_name=RAG_TABLE_NAME,
            force_reindex=False,
            message="RAG index already exists. Use force_reindex=true to rebuild.",
        )

    main_file, assets_dir = _resolve_paths(req.main_file, req.assets_dir)
    try:
        records = build_and_store(
            main_file=main_file,
            assets_dir=assets_dir,
            table_name=RAG_TABLE_NAME,
            embedding_dim=RAG_EMBEDDING_DIM,
            max_chars=req.max_chars,
            overlap=req.overlap,
            force_reindex=req.force_reindex,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index textbook: {exc}")

    return IndexTextbookResponse(
        indexed=True,
        record_count=len(records),
        table_name=RAG_TABLE_NAME,
        force_reindex=req.force_reindex,
        message="Textbook indexed successfully.",
    )


@router.post("/lesson_planner/generate", response_model=LessonPlanResponse)
async def generate_lesson_plan(req: LessonPlanRequest):
    if count_records(table_name=RAG_TABLE_NAME) == 0:
        raise HTTPException(
            status_code=400,
            detail="RAG index is empty. Run /api/lesson_planner/index_textbook first.",
        )

    try:
        query = " ".join([req.subject, req.grade, req.topic, req.description] + req.objectives).strip()
        sources = query_similar(query, top_k=req.top_k, table_name=RAG_TABLE_NAME, embedding_dim=RAG_EMBEDDING_DIM)
        prompt = _build_prompt(req, sources)
        raw = generate_answer_gemini_llm(prompt)
        parsed = _extract_json(raw)
        plan = _normalize_plan(req, parsed or {})
        return LessonPlanResponse(plan=plan, sources=sources, raw_text=None if parsed else raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plan: {exc}")


def _ensure_reportlab():
    if SimpleDocTemplate is None:
        raise HTTPException(status_code=500, detail="reportlab is not installed")


def _pdf_text(text: Any) -> str:
    return str(text or "").replace("\n", "<br/>")


def _build_pdf(plan: Dict[str, Any]) -> bytes:
    _ensure_reportlab()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48)

    styles = getSampleStyleSheet()
    heading = styles["Heading2"]
    body = styles["BodyText"]
    title_style = styles["Title"]

    story = [Paragraph(_pdf_text(plan.get("title", "Lesson Plan")), title_style), Spacer(1, 12)]

    meta_lines = [
        f"Subject: {plan.get('subject', '')}",
        f"Grade: {plan.get('grade', '')}",
        f"Topic: {plan.get('topic', '')}",
    ]
    for line in meta_lines:
        story.append(Paragraph(_pdf_text(line), body))
    story.append(Spacer(1, 12))

    if plan.get("overview"):
        story.extend([Paragraph("Overview", heading), Paragraph(_pdf_text(plan.get("overview", "")), body), Spacer(1, 12)])

    objectives = plan.get("objectives", []) or []
    if objectives:
        story.append(Paragraph("Objectives", heading))
        story.append(ListFlowable([ListItem(Paragraph(_pdf_text(obj), body)) for obj in objectives], bulletType="bullet"))
        story.append(Spacer(1, 12))

    lesson_flow = plan.get("lesson_flow", []) or []
    if lesson_flow:
        story.append(Paragraph("Lesson Flow", heading))
        for step in lesson_flow:
            title = step.get("title", "")
            content = step.get("content", "")
            story.append(Paragraph(_pdf_text(title), styles["Heading3"]))
            if content:
                story.append(Paragraph(_pdf_text(content), body))
            story.append(Spacer(1, 8))
        story.append(Spacer(1, 12))

    mcq = plan.get("mcq_exercises", []) or []
    if mcq:
        story.append(Paragraph("MCQ Exercises", heading))
        for idx, q in enumerate(mcq, start=1):
            story.append(Paragraph(_pdf_text(f"{idx}. {q.get('question', '')}"), body))
            options = q.get("options", []) or []
            if options:
                story.append(
                    ListFlowable(
                        [
                            ListItem(Paragraph(_pdf_text(f"{chr(65 + i)}. {_normalize_option_text(opt)}"), body))
                            for i, opt in enumerate(options)
                        ],
                        bulletType="bullet",
                    )
                )
            story.append(Spacer(1, 8))
        story.append(Spacer(1, 12))

    resources = plan.get("resources", []) or []
    if resources:
        story.append(Paragraph("Resources", heading))
        story.append(ListFlowable([ListItem(Paragraph(_pdf_text(r), body)) for r in resources], bulletType="bullet"))
        story.append(Spacer(1, 12))

    notes = plan.get("presentation_notes", "") or plan.get("notes_for_teacher", "")
    if notes:
        story.append(Paragraph("Quick Recap", heading))
        story.append(Paragraph(_pdf_text(notes), body))
        story.append(Spacer(1, 12))

    if mcq:
        story.append(Paragraph("Answer Key", heading))
        for idx, q in enumerate(mcq, start=1):
            answer = _normalize_answer_letter(q.get("answer", "A"))
            story.append(Paragraph(_pdf_text(f"{idx}. {answer}"), body))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


@router.post("/lesson_planner/pdf")
async def lesson_plan_pdf(req: LessonPlanPDFRequest):
    try:
        pdf = _build_pdf(req.plan)
        headers = {"Content-Disposition": "attachment; filename=lesson_plan.pdf"}
        return StreamingResponse(BytesIO(pdf), media_type="application/pdf", headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
