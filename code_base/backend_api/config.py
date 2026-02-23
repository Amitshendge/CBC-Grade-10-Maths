from pathlib import Path
import os

import dotenv

dotenv.load_dotenv()

APP_NAME = os.getenv("APP_NAME", "AI_Lesson_Planner")

# Database configuration (primary path is DATABASE_URL for local/dev/prod portability)
DATABASE_URL = os.getenv("DATABASE_URL", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", APP_NAME.lower())
DB_SSLMODE = os.getenv("DB_SSLMODE", "prefer")
DB_ACCESS_SECRET_NAME = os.getenv("DB_ACCESS_SECRET_NAME", "")

# AWS configuration (optional unless AWS services are used)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "")

# Gemini/Vertex configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "imagen-4.0-generate-001")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_USE_VERTEX = os.getenv("GEMINI_USE_VERTEX", "false").lower() == "true"
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_AI_SERVICE_ACCOUNT_JSON = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_JSON", "")
VERTEX_AI_SERVICE_ACCOUNT_JSON_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_JSON_PATH", "")
VERTEX_AI_SERVICE_ACCOUNT_JSON_SECRET_NAME = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_JSON_SECRET_NAME", "")

# RAG configuration
RAG_TABLE_NAME = os.getenv("RAG_TABLE_NAME", "rag_chunks")
RAG_TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K_DEFAULT", "8"))
RAG_EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "768"))
RAG_CHUNK_MAX_CHARS = int(os.getenv("RAG_CHUNK_MAX_CHARS", "2000"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Textbook paths
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent.parent
MATHS_CLASS_DATA_DIR = Path(os.getenv("MATHS_CLASS_DATA_DIR", str(REPO_ROOT / "Maths_Class_Data")))
MATHS_SOURCE_MAIN_FILE = Path(
    os.getenv("MATHS_SOURCE_MAIN_FILE", str(MATHS_CLASS_DATA_DIR / "source" / "main.ptx"))
)
MATHS_ASSETS_DIR = Path(os.getenv("MATHS_ASSETS_DIR", str(MATHS_CLASS_DATA_DIR / "assets")))

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")

# Middleware configuration
MIDDLEWARE_SESSION_SECRET = os.getenv("MIDDLEWARE_SESSION_SECRET", "change-me")
ACCESS_TOKEN_EXPIRATION = int(os.getenv("ACCESS_TOKEN_EXPIRATION", "3600"))
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", "300"))
REFRESH_TOKEN_EXPIRATION = int(os.getenv("REFRESH_TOKEN_EXPIRATION", "86400"))

# Google OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
HOST_URL = os.getenv("HOST_URL", "http://localhost:8000")
