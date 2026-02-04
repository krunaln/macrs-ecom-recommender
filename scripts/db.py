import os
from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv


load_dotenv()


def get_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    if dsn.startswith("postgresql+psycopg2://"):
        dsn = dsn.replace("postgresql+psycopg2://", "postgresql://", 1)
    return dsn


@contextmanager
def get_conn():
    conn = psycopg2.connect(get_dsn())
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()
