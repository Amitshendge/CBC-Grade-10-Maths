import json
import threading
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from config import (
    APP_NAME,
    DATABASE_URL,
    DB_ACCESS_SECRET_NAME,
    DB_SSLMODE,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)

try:
    from cloud_utils.aws_utils import get_secret
except Exception:  # pragma: no cover
    get_secret = None


DB_NAME = POSTGRES_DB or APP_NAME.lower()


def _load_secret_db_config() -> Optional[Dict[str, Any]]:
    if not DB_ACCESS_SECRET_NAME or get_secret is None:
        return None
    try:
        payload = json.loads(get_secret(DB_ACCESS_SECRET_NAME))
    except Exception:
        return None
    return {
        "host": payload.get("host", POSTGRES_HOST),
        "port": int(payload.get("port", POSTGRES_PORT)),
        "user": payload.get("username", POSTGRES_USER),
        "password": payload.get("password", POSTGRES_PASSWORD),
        "dbname": payload.get("dbname")
        or payload.get("database")
        or payload.get("db_name")
        or DB_NAME,
        "sslmode": payload.get("sslmode", DB_SSLMODE),
    }


def _build_pool_kwargs() -> Dict[str, Any]:
    if DATABASE_URL:
        kwargs: Dict[str, Any] = {
            "dsn": DATABASE_URL,
            "cursor_factory": RealDictCursor,
            "connect_timeout": 10,
        }
        if DB_SSLMODE and "sslmode=" not in DATABASE_URL:
            kwargs["sslmode"] = DB_SSLMODE
        return kwargs

    secret_cfg = _load_secret_db_config()
    if secret_cfg:
        secret_cfg["cursor_factory"] = RealDictCursor
        secret_cfg["connect_timeout"] = 10
        return secret_cfg

    return {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "dbname": DB_NAME,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "sslmode": DB_SSLMODE,
        "cursor_factory": RealDictCursor,
        "connect_timeout": 10,
    }


def _build_admin_kwargs() -> Optional[Dict[str, Any]]:
    if DATABASE_URL:
        parsed = urlparse(DATABASE_URL)
        if not parsed.scheme.startswith("postgres"):
            return None
        if not parsed.hostname or not parsed.username:
            return None
        return {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "dbname": "postgres",
            "user": parsed.username,
            "password": parsed.password or "",
            "sslmode": DB_SSLMODE,
            "cursor_factory": RealDictCursor,
            "connect_timeout": 10,
        }

    secret_cfg = _load_secret_db_config()
    if secret_cfg:
        secret_cfg = dict(secret_cfg)
        secret_cfg["dbname"] = "postgres"
        secret_cfg["cursor_factory"] = RealDictCursor
        secret_cfg["connect_timeout"] = 10
        return secret_cfg

    return {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "dbname": "postgres",
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "sslmode": DB_SSLMODE,
        "cursor_factory": RealDictCursor,
        "connect_timeout": 10,
    }


def _target_db_name() -> Optional[str]:
    if DATABASE_URL:
        parsed = urlparse(DATABASE_URL)
        path_db = parsed.path.lstrip("/")
        return path_db or None
    secret_cfg = _load_secret_db_config()
    if secret_cfg:
        return secret_cfg.get("dbname") or DB_NAME
    return DB_NAME


def _ensure_database_exists() -> None:
    target_db = _target_db_name()
    admin_kwargs = _build_admin_kwargs()
    if not target_db or not admin_kwargs:
        return

    admin_conn = psycopg2.connect(**admin_kwargs)
    try:
        admin_conn.autocommit = True
        with admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target_db,))
            exists = cur.fetchone()
            if not exists:
                try:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
                except psycopg2.errors.DuplicateDatabase:
                    pass
    finally:
        admin_conn.close()


def _create_tables_indexes(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            create extension if not exists vector;

            create table if not exists users (
                id text not null,
                email text not null,
                hashed_password text null,
                full_name text null,
                roles text null,
                is_active boolean null default true,
                created_at timestamp with time zone null default now(),
                updated_at timestamp with time zone null default now(),
                constraint users_pkey primary key (id),
                constraint users_email_key unique (email)
            );

            create table if not exists user_history (
                id text not null,
                hash_content text null,
                num_scenes integer null,
                topic text null,
                objective text null,
                email text null,
                file_name text null,
                file_uri text null,
                status text null,
                created_on timestamp with time zone null default now(),
                constraint user_history_pkey primary key (id)
            );

            create table if not exists rag_chunks (
                id text not null,
                source_id text null,
                chunk_id integer null,
                content text null,
                metadata jsonb null,
                embedding vector(768) null,
                created_at timestamp with time zone null default now(),
                updated_at timestamp with time zone null default now(),
                constraint rag_chunks_pkey primary key (id)
            );

            create index if not exists rag_chunks_embedding_idx
                on rag_chunks using ivfflat (embedding vector_cosine_ops);
            create index if not exists rag_chunks_source_idx
                on rag_chunks (source_id);
            """
        )
    conn.commit()


def get_connection():
    if not hasattr(get_connection, "_pool"):
        get_connection._pool = None
    if not hasattr(get_connection, "_local"):
        get_connection._local = threading.local()
    if not hasattr(get_connection, "_init_lock"):
        get_connection._init_lock = threading.Lock()
    if not hasattr(get_connection, "_schema_inited"):
        get_connection._schema_inited = False

    tl_conn = getattr(get_connection._local, "conn", None)
    if tl_conn is not None and not tl_conn.closed:
        return tl_conn

    with get_connection._init_lock:
        if get_connection._pool is None:
            _ensure_database_exists()
            kwargs = _build_pool_kwargs()
            get_connection._pool = ThreadedConnectionPool(minconn=1, maxconn=10, **kwargs)

    conn = get_connection._pool.getconn()
    get_connection._local.conn = conn

    if not get_connection._schema_inited:
        with get_connection._init_lock:
            if not get_connection._schema_inited:
                lock_key = 987654321
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_lock(%s);", (lock_key,))
                try:
                    _create_tables_indexes(conn)
                finally:
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_advisory_unlock(%s);", (lock_key,))
                    conn.commit()
                get_connection._schema_inited = True

    return conn


def create_tables_indexes():
    conn = get_connection()
    _create_tables_indexes(conn)


def execute_query(query: str, params=None, fetch=True):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            result = cur.fetchall() if fetch else None
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise

