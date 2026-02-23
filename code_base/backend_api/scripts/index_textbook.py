import argparse
from pathlib import Path

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from RAG_utils.rag_pipeline import build_and_store
from config import (
    MATHS_ASSETS_DIR,
    MATHS_SOURCE_MAIN_FILE,
    RAG_CHUNK_MAX_CHARS,
    RAG_CHUNK_OVERLAP,
    RAG_EMBEDDING_DIM,
    RAG_TABLE_NAME,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Index CBC Grade 10 Maths textbook into pgvector.")
    parser.add_argument("--main-file", type=str, default=str(MATHS_SOURCE_MAIN_FILE))
    parser.add_argument("--assets-dir", type=str, default=str(MATHS_ASSETS_DIR))
    parser.add_argument("--table-name", type=str, default=RAG_TABLE_NAME)
    parser.add_argument("--embedding-dim", type=int, default=RAG_EMBEDDING_DIM)
    parser.add_argument("--max-chars", type=int, default=RAG_CHUNK_MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=RAG_CHUNK_OVERLAP)
    parser.add_argument("--force-reindex", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    main_file = Path(args.main_file).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()

    if not main_file.exists():
        raise FileNotFoundError(f"Main PreTeXt file not found: {main_file}")
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets_dir}")

    records = build_and_store(
        main_file=main_file,
        assets_dir=assets_dir,
        table_name=args.table_name,
        embedding_dim=args.embedding_dim,
        max_chars=args.max_chars,
        overlap=args.overlap,
        force_reindex=args.force_reindex,
    )
    print(f"Indexed {len(records)} records into table '{args.table_name}'.")


if __name__ == "__main__":
    main()

