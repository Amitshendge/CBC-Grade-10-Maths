"""
RAG utilities for the CBC Grade 10 Mathematics PreTeXt source.

This module:
- Parses PreTeXt XML with xi:include resolution.
- Extracts structured blocks (paragraphs, activities, examples, exercises, notes, etc.).
- Builds chunked records with hierarchy metadata.
- Embeds content with Gemini embeddings and stores in Postgres (pgvector).
- Retrieves similar chunks for a query and optionally calls Gemini to answer.

Expected repo layout (relative to this file):
- Maths_Class_Data/source/main.ptx
- Maths_Class_Data/assets/
- code_base/backend_api/databases_framework/postgre_util.py
- code_base/backend_api/llm_utils/gemini_utils.py
- code_base/backend_api/cloud_utils/aws_utils.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# Local utilities
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from databases_framework.postgre_util import get_connection, execute_query  # noqa: E402
from cloud_utils.aws_utils import upload_file  # noqa: E402
from llm_utils.gemini_utils import embed_texts_gemini, generate_answer_gemini_llm  # noqa: E402
from config import (  # noqa: E402
    AWS_S3_BUCKET_NAME,
    GEMINI_EMBED_MODEL,
    MATHS_ASSETS_DIR,
    MATHS_CLASS_DATA_DIR,
    MATHS_SOURCE_MAIN_FILE,
    RAG_EMBEDDING_DIM,
    RAG_TABLE_NAME,
)

DEFAULT_BOOK_DIR = MATHS_CLASS_DATA_DIR
DEFAULT_SOURCE_DIR = DEFAULT_BOOK_DIR / "source"
DEFAULT_ASSETS_DIR = MATHS_ASSETS_DIR
DEFAULT_MAIN_FILE = MATHS_SOURCE_MAIN_FILE
DEFAULT_TABLE = RAG_TABLE_NAME
DEFAULT_EMBED_MODEL = GEMINI_EMBED_MODEL
DEFAULT_EMBED_DIM = RAG_EMBEDDING_DIM

XI_NS = "http://www.w3.org/2001/XInclude"
XML_NS = "http://www.w3.org/XML/1998/namespace"

DIVISION_TAGS = {
    "frontmatter",
    "chapter",
    "section",
    "subsection",
    "subsubsection",
    "backmatter",
}

BLOCK_TAGS = {
    "p",
    "activity",
    "example",
    "exercise",
    "exercises",
    "note",
    "insight",
    "definition",
    "theorem",
    "proof",
    "remark",
    "observation",
    "lemma",
    "corollary",
    "fact",
    "task",
    "project",
    "investigation",
    "exploration",
    "warning",
    "tabular",
    "table",
    "figure",
    "sidebyside",
}

INTRO_CONTAINER_TAGS = {"introduction", "conclusion", "summary"}
SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(value: str) -> str:
    if not SQL_IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return value


# -------------------------
# XML Parsing Helpers
# -------------------------

def _local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _get_xml_id(elem: ET.Element) -> Optional[str]:
    return elem.get(f"{{{XML_NS}}}id") or elem.get("xml:id") or elem.get("id")


def _get_first_child(elem: ET.Element, tag_name: str) -> Optional[ET.Element]:
    for child in elem:
        if _local_name(child.tag) == tag_name:
            return child
    return None


def _get_title(elem: ET.Element) -> Optional[str]:
    title_elem = _get_first_child(elem, "title")
    if title_elem is None:
        return None
    text, _ = _collect_text(title_elem, skip_tags=None)
    return text or None


def _parse_xml(path: Path) -> ET.Element:
    tree = ET.parse(path)
    root = tree.getroot()
    root.set("_source_path", str(path))
    return root


def _resolve_includes(elem: ET.Element, base_dir: Path) -> None:
    for child in list(elem):
        if child.tag == f"{{{XI_NS}}}include":
            href = child.get("href")
            if not href:
                elem.remove(child)
                continue
            include_path = (base_dir / href).resolve()
            included_root = _parse_xml(include_path)
            _resolve_includes(included_root, include_path.parent)
            idx = list(elem).index(child)
            elem.remove(child)
            elem.insert(idx, included_root)
        else:
            _resolve_includes(child, base_dir)


def load_pretext_book(main_file: Path = DEFAULT_MAIN_FILE) -> ET.Element:
    root = _parse_xml(main_file)
    _resolve_includes(root, main_file.parent)
    return root


# -------------------------
# Content Extraction
# -------------------------

def _collect_text(
    elem: ET.Element,
    skip_tags: Optional[set] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    if skip_tags is None:
        skip_tags = set()

    parts: List[str] = []
    math_items: List[Dict[str, str]] = []

    def walk(node: ET.Element) -> None:
        tag = _local_name(node.tag)
        if tag in skip_tags:
            return

        if tag in {"m", "md"}:
            latex = "".join(node.itertext()).strip()
            if latex:
                math_items.append(
                    {"type": "inline" if tag == "m" else "display", "latex": latex}
                )
                if tag == "m":
                    parts.append(f"${latex}$")
                else:
                    parts.append(f"$$ {latex} $$")
            return

        if tag == "latex-image":
            return

        if tag == "xref":
            ref = node.get("ref")
            if ref:
                parts.append(f"[xref:{ref}]")
            return

        if node.text:
            parts.append(node.text)

        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(elem)
    text = " ".join("".join(parts).split())
    return text, math_items


def _extract_stack_refs(elem: ET.Element) -> List[Dict[str, str]]:
    items = []
    for node in elem.iter():
        if _local_name(node.tag) == "stack":
            label = node.get("label")
            source = node.get("source")
            items.append({"label": label or "", "source": source or ""})
    return items


def _extract_images(elem: ET.Element, assets_dir: Optional[Path]) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    figure_caption_map: Dict[int, str] = {}

    for figure in elem.iter():
        if _local_name(figure.tag) != "figure":
            continue
        caption_elem = _get_first_child(figure, "caption")
        caption = None
        if caption_elem is not None:
            caption, _ = _collect_text(caption_elem, skip_tags=None)
        for img in figure.iter():
            if _local_name(img.tag) == "image":
                figure_caption_map[id(img)] = caption or ""

    for img in elem.iter():
        if _local_name(img.tag) != "image":
            continue
        source = img.get("source")
        shortdesc_elem = _get_first_child(img, "shortdescription")
        shortdesc = None
        if shortdesc_elem is not None:
            shortdesc, _ = _collect_text(shortdesc_elem, skip_tags=None)
        latex_image_elem = _get_first_child(img, "latex-image")
        latex_image = None
        if latex_image_elem is not None:
            latex_image = "".join(latex_image_elem.itertext()).strip() or None

        resolved_path = None
        exists = None
        if assets_dir is not None and source:
            resolved_path = str(assets_dir / source)
            exists = (assets_dir / source).exists()

        images.append(
            {
                "source": source or "",
                "caption": figure_caption_map.get(id(img), "") or "",
                "shortdescription": shortdesc or "",
                "latex_image": latex_image or "",
                "resolved_path": resolved_path or "",
                "exists": bool(exists) if exists is not None else False,
            }
        )

    return images


def _build_embedding_text(text: str, hierarchy: Dict[str, str], block_type: str) -> str:
    parts = []
    for key in ("book", "chapter", "section", "subsection", "subsubsection"):
        if hierarchy.get(key):
            parts.append(hierarchy[key])
    header = " > ".join(parts)
    if header:
        return f"{header}\n[{block_type}] {text}"
    return f"[{block_type}] {text}"


def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append(" ".join(current))
        if overlap > 0:
            overlap_words: List[str] = []
            total = 0
            for w in reversed(current):
                total += len(w) + 1
                overlap_words.append(w)
                if total >= overlap:
                    break
            current = list(reversed(overlap_words))
            current_len = sum(len(w) + 1 for w in current)
        else:
            current = []
            current_len = 0

    for word in words:
        add_len = len(word) + (1 if current else 0)
        if current_len + add_len > max_chars:
            flush()
        current.append(word)
        current_len += add_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _make_record_id(*parts: str) -> str:
    raw = "|".join(p or "" for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_text_fallback(text: str, block_type: str, stack_refs: List[Dict[str, str]], images: List[Dict[str, Any]]) -> str:
    if text:
        return text
    if stack_refs:
        labels = ", ".join([s.get("label", "") for s in stack_refs if s.get("label")])
        return f"STACK exercise: {labels}" if labels else "STACK exercise"
    if images:
        return f"Image reference ({block_type})"
    return ""


def extract_records(
    main_file: Path = DEFAULT_MAIN_FILE,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    max_chars: int = 2000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    root = load_pretext_book(main_file)

    # Find the book element
    book_elem = None
    if _local_name(root.tag) == "pretext":
        for child in root:
            if _local_name(child.tag) == "book":
                book_elem = child
                break
    if book_elem is None and _local_name(root.tag) == "book":
        book_elem = root

    if book_elem is None:
        raise ValueError("Could not locate <book> in PreTeXt source")

    book_title = _get_title(book_elem) or ""

    records: List[Dict[str, Any]] = []
    division_block_counts: Dict[Tuple[str, str], int] = {}

    def _next_block_index(source_path: str, division_id: str) -> int:
        key = (source_path, division_id)
        division_block_counts[key] = division_block_counts.get(key, 0) + 1
        return division_block_counts[key]

    def walk_division(elem: ET.Element, hierarchy: Dict[str, str], source_path: str) -> None:
        tag = _local_name(elem.tag)
        title = _get_title(elem) or ""
        elem_id = _get_xml_id(elem) or ""

        next_hierarchy = dict(hierarchy)
        if tag == "book":
            next_hierarchy["book"] = title or book_title
        elif tag in DIVISION_TAGS:
            if tag == "chapter":
                next_hierarchy["chapter"] = title
                next_hierarchy["section"] = ""
                next_hierarchy["subsection"] = ""
                next_hierarchy["subsubsection"] = ""
            elif tag == "section":
                next_hierarchy["section"] = title
                next_hierarchy["subsection"] = ""
                next_hierarchy["subsubsection"] = ""
            elif tag == "subsection":
                next_hierarchy["subsection"] = title
                next_hierarchy["subsubsection"] = ""
            elif tag == "subsubsection":
                next_hierarchy["subsubsection"] = title

        elem_source_path = elem.get("_source_path") or source_path

        # Extract blocks from this division
        for child in list(elem):
            child_tag = _local_name(child.tag)
            if child_tag == "title":
                continue
            if child_tag in DIVISION_TAGS:
                continue

            if child_tag in INTRO_CONTAINER_TAGS:
                # extract blocks inside introduction/conclusion
                for sub in list(child):
                    sub_tag = _local_name(sub.tag)
                    if sub_tag in DIVISION_TAGS or sub_tag == "title":
                        continue
                    block_seq = _next_block_index(elem_source_path, elem_id or tag)
                    record = _build_block_record(
                        sub,
                        next_hierarchy,
                        elem_id,
                        elem_source_path,
                        block_seq,
                        assets_dir,
                        max_chars,
                        overlap,
                    )
                    records.extend(record)
                continue

            block_seq = _next_block_index(elem_source_path, elem_id or tag)
            record = _build_block_record(
                child,
                next_hierarchy,
                elem_id,
                elem_source_path,
                block_seq,
                assets_dir,
                max_chars,
                overlap,
            )
            records.extend(record)

        # Recurse into child divisions
        for child in list(elem):
            if _local_name(child.tag) in DIVISION_TAGS:
                walk_division(child, next_hierarchy, elem_source_path)

    def _build_block_record(
        node: ET.Element,
        hierarchy: Dict[str, str],
        division_id: str,
        source_path: str,
        block_seq: int,
        assets_dir: Path,
        max_chars: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        tag = _local_name(node.tag)
        block_type = tag if tag in BLOCK_TAGS else "block"
        block_id = _get_xml_id(node) or ""
        block_title = _get_title(node) or ""

        text, math_items = _collect_text(node, skip_tags={"title"})
        images = _extract_images(node, assets_dir)
        stack_refs = _extract_stack_refs(node)
        text = _ensure_text_fallback(text, block_type, stack_refs, images)
        if block_title and text and not text.startswith(block_title):
            text = f"{block_title}. {text}"

        if not text:
            return []

        # images and stack_refs already collected

        chunks = _chunk_text(text, max_chars=max_chars, overlap=overlap)

        results: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            source_id = block_id or division_id
            record_id = _make_record_id(
                source_path,
                division_id,
                block_id,
                str(block_seq),
                str(idx),
                chunk,
            )
            metadata = {
                "book": hierarchy.get("book", book_title),
                "chapter": hierarchy.get("chapter", ""),
                "section": hierarchy.get("section", ""),
                "subsection": hierarchy.get("subsection", ""),
                "subsubsection": hierarchy.get("subsubsection", ""),
                "division_id": division_id,
                "block_id": block_id,
                "block_type": block_type,
                "block_title": block_title,
                "source_path": source_path,
                "images": images,
                "stack": stack_refs,
                "math": math_items,
                "block_seq": block_seq,
                "chunk_index": idx,
                "chunk_total": len(chunks),
            }

            results.append(
                {
                    "id": record_id,
                    "source_id": source_id,
                    "chunk_id": idx,
                    "content": chunk,
                    "embedding_text": _build_embedding_text(chunk, hierarchy, block_type),
                    "metadata": metadata,
                }
            )

        return results

    walk_division(book_elem, {"book": book_title}, book_elem.get("_source_path") or str(main_file))
    return records


# -------------------------
# Optional S3 Asset Upload
# -------------------------

def upload_assets_to_s3(records: List[Dict[str, Any]], prefix: str = "maths_assets/") -> None:
    """
    Uploads referenced image assets to S3 and updates metadata in-place with s3_key.
    Uses AWS_S3_BUCKET_NAME from config.
    """
    if not AWS_S3_BUCKET_NAME:
        raise ValueError("AWS_S3_BUCKET_NAME is not configured.")

    for record in records:
        images = record.get("metadata", {}).get("images", [])
        for img in images:
            local_path = img.get("resolved_path")
            if not local_path:
                continue
            path = Path(local_path)
            if not path.exists():
                continue
            s3_key = f"{prefix}{path.name}"
            with open(path, "rb") as f:
                upload_file(AWS_S3_BUCKET_NAME, s3_key, f.read())
            img["s3_key"] = s3_key


# -------------------------
# Embedding Helpers
# -------------------------


def embed_texts(
    texts: List[str],
    model: str = DEFAULT_EMBED_MODEL,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 16,
) -> List[List[float]]:
    return embed_texts_gemini(texts, model=model, task_type=task_type, batch_size=batch_size)


def vectorize_records(
    records: List[Dict[str, Any]],
    model: str = DEFAULT_EMBED_MODEL,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    texts = [r["embedding_text"] for r in records]
    embeddings = embed_texts(texts, model=model, task_type=task_type, batch_size=batch_size)
    if len(embeddings) != len(records):
        raise ValueError("Embedding count mismatch")
    for record, emb in zip(records, embeddings):
        record["embedding"] = emb
    return records


# -------------------------
# Postgres (pgvector) Storage
# -------------------------

def _vector_to_pgvector(embedding: List[float]) -> str:
    return "[" + ",".join(f"{float(x):.9f}" for x in embedding) + "]"


def init_vector_store(
    table_name: str = DEFAULT_TABLE,
    embedding_dim: int = DEFAULT_EMBED_DIM,
    create_index: bool = True,
    index_type: str = "ivfflat",
    distance_metric: str = "cosine",
) -> None:
    table_name = _safe_identifier(table_name)
    execute_query("CREATE EXTENSION IF NOT EXISTS vector;", fetch=False)

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id TEXT PRIMARY KEY,
        source_id TEXT,
        chunk_id INTEGER,
        content TEXT,
        metadata JSONB,
        embedding VECTOR({int(embedding_dim)})
    );
    """
    execute_query(create_table_sql, fetch=False)

    if create_index:
        if distance_metric == "cosine":
            ops = "vector_cosine_ops"
        elif distance_metric == "l2":
            ops = "vector_l2_ops"
        else:
            ops = "vector_ip_ops"

        index_name = f"{table_name}_embedding_idx"
        index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING {index_type} (embedding {ops});"
        try:
            execute_query(index_sql, fetch=False)
        except Exception:
            # Index creation can fail if extension/privileges are missing; storage still works.
            pass


def upsert_records(
    records: List[Dict[str, Any]],
    table_name: str = DEFAULT_TABLE,
    embedding_dim: int = DEFAULT_EMBED_DIM,
    batch_size: int = 200,
) -> None:
    table_name = _safe_identifier(table_name)
    if not records:
        return
    # Deduplicate by id to avoid ON CONFLICT DO UPDATE affecting the same row twice
    unique_records: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_id = record.get("id")
        if not record_id:
            continue
        unique_records[record_id] = record
    records = list(unique_records.values())

    conn = get_connection()
    with conn.cursor() as cur:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            values = []
            for r in batch:
                emb = r.get("embedding")
                if emb is None:
                    raise ValueError("Record is missing embedding")
                if len(emb) != embedding_dim:
                    raise ValueError("Embedding dimension mismatch")
                values.append(
                    (
                        r["id"],
                        r.get("source_id", ""),
                        r.get("chunk_id", 0),
                        r.get("content", ""),
                        json.dumps(r.get("metadata", {})),
                        _vector_to_pgvector(emb),
                    )
                )

            args_str = ",".join(["(%s,%s,%s,%s,%s::jsonb,%s::vector)"] * len(values))
            sql = f"""
                INSERT INTO {table_name} (id, source_id, chunk_id, content, metadata, embedding)
                VALUES {args_str}
                ON CONFLICT (id) DO UPDATE SET
                    source_id = EXCLUDED.source_id,
                    chunk_id = EXCLUDED.chunk_id,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
            """
            flat_params: List[Any] = []
            for row in values:
                flat_params.extend(row)
            cur.execute(sql, flat_params)
        conn.commit()


def query_similar(
    query_text: str,
    top_k: int = 5,
    table_name: str = DEFAULT_TABLE,
    model: str = DEFAULT_EMBED_MODEL,
    embedding_dim: int = DEFAULT_EMBED_DIM,
) -> List[Dict[str, Any]]:
    table_name = _safe_identifier(table_name)
    query_emb = embed_texts([query_text], model=model, task_type="RETRIEVAL_QUERY")[0]
    if len(query_emb) != embedding_dim:
        raise ValueError("Query embedding dimension mismatch")
    query_vec = _vector_to_pgvector(query_emb)

    sql = f"""
        SELECT id, source_id, chunk_id, content, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM {table_name}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    rows = execute_query(sql, params=(query_vec, query_vec, top_k), fetch=True)
    return rows or []


def count_records(table_name: str = DEFAULT_TABLE) -> int:
    table_name = _safe_identifier(table_name)
    try:
        rows = execute_query(f"SELECT COUNT(*)::int AS count FROM {table_name};", fetch=True)
    except Exception:
        return 0
    if not rows:
        return 0
    return int(rows[0].get("count", 0))


def clear_records(table_name: str = DEFAULT_TABLE) -> None:
    table_name = _safe_identifier(table_name)
    execute_query(f"TRUNCATE TABLE {table_name};", fetch=False)


# -------------------------
# RAG Answer Helper
# -------------------------

def build_rag_prompt(query: str, rows: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for i, row in enumerate(rows, start=1):
        meta = row.get("metadata") or {}
        title_parts = [
            meta.get("chapter", ""),
            meta.get("section", ""),
            meta.get("subsection", ""),
            meta.get("subsubsection", ""),
        ]
        title = " > ".join([p for p in title_parts if p])
        context_blocks.append(f"[{i}] {title}\n{row.get('content','')}")

    context_text = "\n\n".join(context_blocks)
    prompt = (
        "You are a helpful tutor answering from the provided textbook excerpts. "
        "Cite sources using [#] from the context. If unsure, say you are not sure.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return prompt


def rag_answer(query: str, top_k: int = 5) -> Dict[str, Any]:
    rows = query_similar(query, top_k=top_k)
    prompt = build_rag_prompt(query, rows)
    answer = generate_answer_gemini_llm(prompt)
    return {
        "query": query,
        "answer": answer,
        "sources": rows,
    }


# -------------------------
# Export Helpers
# -------------------------

def export_records_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_and_store(
    main_file: Path = DEFAULT_MAIN_FILE,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    table_name: str = DEFAULT_TABLE,
    embedding_dim: int = DEFAULT_EMBED_DIM,
    max_chars: int = 2000,
    overlap: int = 200,
    force_reindex: bool = False,
) -> List[Dict[str, Any]]:
    """
    End-to-end helper:
    - Extract records from PreTeXt
    - Embed
    - Create vector store schema
    - Upsert into Postgres
    """
    records = extract_records(
        main_file=main_file,
        assets_dir=assets_dir,
        max_chars=max_chars,
        overlap=overlap,
    )
    vectorize_records(records)
    init_vector_store(table_name=table_name, embedding_dim=embedding_dim)
    if force_reindex:
        clear_records(table_name=table_name)
    upsert_records(records, table_name=table_name, embedding_dim=embedding_dim)
    return records


if __name__ == "__main__":
    # Example usage (manual run):
    # records = build_and_store()
    # print(f"Stored {len(records)} records")
        
    # Query
    result = rag_answer("NUMBERS AND ALGEBRA")
    for i in result["sources"]:
        print(i['content'])

    pass
