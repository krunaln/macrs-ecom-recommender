import argparse
import csv
import json
import logging
import time
import uuid
from decimal import Decimal
from typing import Iterable, List, Optional, Tuple

from langchain_ollama import OllamaEmbeddings
from pgvector import Vector
from psycopg2.extras import execute_batch

from db import get_conn


REQUIRED_COLUMNS = {
    "title",
    "brand",
    "description",
    "final_price",
    "currency",
    "categories",
    "image_url",
}


def _parse_categories(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            values = json.loads(raw)
            if isinstance(values, list):
                return [str(v).strip() for v in values if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [s.strip() for s in raw.split("|") if s.strip()]


def _stable_id(title: str, brand: str, categories: List[str]) -> uuid.UUID:
    seed = "|".join([title.strip().lower(), brand.strip().lower(), "|".join(categories).lower()])
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _build_text(title: str, brand: str, description: str, categories: List[str], max_chars: int) -> str:
    parts = [title, brand, description, " ".join(categories)]
    text = "\n".join([p.strip() for p in parts if p and p.strip()])
    return text[:max_chars]


def _to_decimal(raw: str) -> Optional[Decimal]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return Decimal(raw)
    except Exception:
        return None


def _validate_columns(header: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS - set(h.strip() for h in header)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def iter_rows(path: str) -> Iterable[tuple[int, dict]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _validate_columns(reader.fieldnames or [])
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def ingest(
    csv_path: str,
    batch_size: int,
    model: str,
    base_url: Optional[str],
    max_chars: int,
    limit: Optional[int],
) -> None:
    embedder = OllamaEmbeddings(model=model, base_url=base_url) if base_url else OllamaEmbeddings(model=model)

    rows: List[Tuple] = []
    total = 0
    skipped = 0
    batch_num = 0
    start_time = time.time()
    with get_conn() as conn:
        with conn.cursor() as cur:
            for idx, row in iter_rows(csv_path):
                if limit is not None and total >= limit:
                    logging.info("Reached limit=%d rows; stopping early", limit)
                    break
                total += 1
                logging.info("Row %d: starting", idx)
                logging.debug("Row %d raw data: %s", idx, row)
                title = (row.get("title") or "").strip()
                brand = (row.get("brand") or "").strip()
                description = (row.get("description") or "").strip()
                categories = _parse_categories(row.get("categories") or "")
                price = _to_decimal(row.get("final_price") or "")
                currency = (row.get("currency") or "").strip()
                logging.info(
                    "Row %d: title_len=%d brand_len=%d desc_len=%d categories=%d price=%s currency=%s",
                    idx,
                    len(title),
                    len(brand),
                    len(description),
                    len(categories),
                    str(price) if price is not None else "None",
                    currency or "None",
                )
                if not title:
                    logging.warning("Skipping row with empty title")
                    skipped += 1
                    continue

                product_id = _stable_id(title, brand, categories)
                embedding_text = _build_text(title, brand, description, categories, max_chars)
                logging.info(
                    "Row %d: generating embedding for id=%s (text_len=%d max_chars=%d)",
                    idx,
                    product_id,
                    len(embedding_text),
                    max_chars,
                )
                t0 = time.time()
                embedding = embedder.embed_query(embedding_text)
                t1 = time.time()
                logging.info(
                    "Row %d: embedding generated for id=%s (dim=%d) in %.2fs",
                    idx,
                    product_id,
                    len(embedding),
                    t1 - t0,
                )

                tsv_text = " ".join(
                    [
                        title,
                        brand,
                        description,
                        " ".join(categories),
                    ]
                ).strip()

                rows.append(
                    (
                        str(product_id),
                        title,
                        brand,
                        description,
                        json.dumps(categories),
                        price,
                        currency,
                        Vector(embedding),
                        tsv_text,
                    )
                )
                logging.debug("Row %d: tsv_text_len=%d", idx, len(tsv_text))

                if len(rows) >= batch_size:
                    batch_num += 1
                    logging.info("Flushing batch %d with %d rows", batch_num, len(rows))
                    _flush(cur, rows, batch_num)
                    rows = []

            if rows:
                batch_num += 1
                logging.info("Flushing final batch %d with %d rows", batch_num, len(rows))
                _flush(cur, rows, batch_num)

        conn.commit()
        elapsed = time.time() - start_time
        logging.info(
            "Ingestion complete. total=%d skipped=%d batches=%d elapsed=%.2fs",
            total,
            skipped,
            batch_num,
            elapsed,
        )


def _flush(cur, rows: List[Tuple], batch_num: int) -> None:
    logging.info("Batch %d: writing %d rows to database", batch_num, len(rows))
    t0 = time.time()
    sql = """
    INSERT INTO products (
        id,
        title,
        brand,
        description,
        categories,
        price,
        currency,
        embedding,
        tsv
    )
    VALUES (
        %(id)s,
        %(title)s,
        %(brand)s,
        %(description)s,
        %(categories)s,
        %(price)s,
        %(currency)s,
        %(embedding)s,
        to_tsvector('english', %(tsv_text)s)
    )
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title,
        brand = EXCLUDED.brand,
        description = EXCLUDED.description,
        categories = EXCLUDED.categories,
        price = EXCLUDED.price,
        currency = EXCLUDED.currency,
        embedding = EXCLUDED.embedding,
        tsv = to_tsvector('english', EXCLUDED.title || ' ' || EXCLUDED.brand || ' ' || EXCLUDED.description || ' ' || EXCLUDED.categories)
    """

    params = [
        {
            "id": r[0],
            "title": r[1],
            "brand": r[2],
            "description": r[3],
            "categories": r[4],
            "price": r[5],
            "currency": r[6],
            "embedding": r[7],
            "tsv_text": r[8],
        }
        for r in rows
    ]

    execute_batch(cur, sql, params, page_size=len(rows))
    t1 = time.time()
    logging.info("Batch %d: write complete in %.2fs", batch_num, t1 - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest products CSV into PostgreSQL")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--base-url", default=None, help="Override Ollama base URL")
    parser.add_argument("--max-chars", type=int, default=3000, help="Max chars per embedding input")
    parser.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    ingest(args.csv_path, args.batch_size, args.model, args.base_url, args.max_chars, args.limit)


if __name__ == "__main__":
    main()
