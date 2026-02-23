import base64
import json
import pathlib
import string
import wave
from functools import lru_cache
from typing import List, Optional

from google import genai
from google.genai import types

try:
    from google.auth import load_credentials_from_dict
except Exception:  # pragma: no cover
    load_credentials_from_dict = None

import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from config import (
    GEMINI_API_KEY,
    GEMINI_EMBED_MODEL,
    GEMINI_IMAGE_MODEL,
    GEMINI_MODEL,
    GEMINI_TTS_MODEL,
    GEMINI_USE_VERTEX,
    VERTEX_AI_SERVICE_ACCOUNT_JSON,
    VERTEX_AI_SERVICE_ACCOUNT_JSON_PATH,
    VERTEX_AI_SERVICE_ACCOUNT_JSON_SECRET_NAME,
    VERTEX_LOCATION,
    VERTEX_PROJECT_ID,
)

try:
    from cloud_utils.aws_utils import get_secret
except Exception:  # pragma: no cover
    get_secret = None


def _read_json_file(path_value: str) -> Optional[dict]:
    path = pathlib.Path(path_value)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_service_account_json() -> Optional[dict]:
    if VERTEX_AI_SERVICE_ACCOUNT_JSON:
        raw = VERTEX_AI_SERVICE_ACCOUNT_JSON.strip()
        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except Exception:
                pass
        file_payload = _read_json_file(raw)
        if file_payload:
            return file_payload

    if VERTEX_AI_SERVICE_ACCOUNT_JSON_PATH:
        file_payload = _read_json_file(VERTEX_AI_SERVICE_ACCOUNT_JSON_PATH)
        if file_payload:
            return file_payload

    if VERTEX_AI_SERVICE_ACCOUNT_JSON_SECRET_NAME and get_secret is not None:
        try:
            secret = get_secret(VERTEX_AI_SERVICE_ACCOUNT_JSON_SECRET_NAME)
            return json.loads(secret)
        except Exception:
            return None

    return None


@lru_cache()
def _get_client():
    if GEMINI_API_KEY and not GEMINI_USE_VERTEX:
        return genai.Client(api_key=GEMINI_API_KEY)

    service_account_json = _load_service_account_json()
    if service_account_json and load_credentials_from_dict is not None:
        credentials, project_from_credentials = load_credentials_from_dict(
            service_account_json,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        project_id = VERTEX_PROJECT_ID or project_from_credentials or service_account_json.get("project_id")
        if not project_id:
            raise RuntimeError("Vertex project id is missing. Set VERTEX_PROJECT_ID.")
        return genai.Client(
            vertexai=True,
            location=VERTEX_LOCATION,
            credentials=credentials,
            project=project_id,
        )

    if GEMINI_USE_VERTEX and VERTEX_PROJECT_ID:
        return genai.Client(vertexai=True, location=VERTEX_LOCATION, project=VERTEX_PROJECT_ID)

    if GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)

    raise RuntimeError(
        "Gemini credentials missing. Set GEMINI_API_KEY, or configure Vertex credentials."
    )


def _extract_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates and candidates[0].content.parts:
            return "".join(getattr(part, "text", "") for part in candidates[0].content.parts).strip()
    except Exception:
        pass
    raise RuntimeError("Gemini returned an empty response.")


def get_label_tag():
    return {
        "project": "AI_Lesson_Planner",
        "env": "dev",
    }


def generate_answer_gemini_llm(prompt: str, model: str = GEMINI_MODEL) -> str:
    client = _get_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return _extract_text(response)


def embed_texts_gemini(
    texts: List[str],
    model: str = GEMINI_EMBED_MODEL,
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 16,
) -> List[List[float]]:
    if not texts:
        return []

    client = _get_client()
    embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        for emb in getattr(response, "embeddings", []) or []:
            values = getattr(emb, "values", None)
            if values is None and isinstance(emb, dict):
                values = emb.get("values")
            if values is None:
                raise RuntimeError("Unexpected embedding response shape from Gemini.")
            embeddings.append([float(v) for v in values])

    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding count mismatch from Gemini API.")

    return embeddings


def generate_image_gemini(prompt: str, aspect_ratio: str = "4:3", n: int = 1, model: str = GEMINI_IMAGE_MODEL):
    client = _get_client()
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=n, aspect_ratio=aspect_ratio),
    )
    images = getattr(response, "images", None) or getattr(response, "generated_images", None) or []
    output = []
    for image in images:
        image_bytes = getattr(image, "image_bytes", None)
        if image_bytes:
            output.append(image_bytes)
    return output


def _looks_like_base64(data: bytes) -> bool:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        return False

    allowed = set(string.ascii_letters + string.digits + "+/=\n\r")
    return all(c in allowed for c in text)


def generate_text_to_speech_gemini(
    script: str,
    tone: str,
    file_path: str,
    voice: str = "Achernar",
    model: str = GEMINI_TTS_MODEL,
):
    script_with_tone = f""""[Say in a {tone} tone and take appropriate pauses for punctuations]:'{script}'"""
    tts_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        labels=get_label_tag(),
    )

    client = _get_client()
    tts_response = client.models.generate_content(model=model, contents=script_with_tone, config=tts_config)
    audio_data = tts_response.candidates[0].content.parts[0].inline_data.data

    if _looks_like_base64(audio_data[:40]):
        audio_data = base64.b64decode(audio_data)

    output_path = pathlib.Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

