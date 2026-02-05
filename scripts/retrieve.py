from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaEmbeddings
from pgvector import Vector

from scripts.db import get_conn


def _build_filters(params: Dict[str, Any]) -> str:
    clauses = []
    if params.get("price_min") is not None:
        clauses.append("price >= %(price_min)s")
    if params.get("price_max") is not None:
        clauses.append("price <= %(price_max)s")
    if params.get("currency"):
        clauses.append("currency = %(currency)s")
    if params.get("brand"):
        clauses.append("brand ILIKE %(brand_like)s")
    if params.get("category"):
        clauses.append("categories ILIKE %(category_like)s")
    return " AND ".join(clauses) if clauses else "TRUE"


def hybrid_search(
    query: str,
    k: int = 10,
    alpha: float = 0.5,
    beta: float = 0.5,
    dense_k: int = 50,
    sparse_k: int = 50,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    currency: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    model: str = "nomic-embed-text",
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    embedder = OllamaEmbeddings(model=model, base_url=base_url) if base_url else OllamaEmbeddings(model=model)
    query_embedding = Vector(embedder.embed_query(query))

    params: Dict[str, Any] = {
        "query": query,
        "embedding": query_embedding,
        "k": k,
        "dense_k": dense_k,
        "sparse_k": sparse_k,
        "alpha": alpha,
        "beta": beta,
        "price_min": price_min,
        "price_max": price_max,
        "currency": currency,
        "brand": brand,
        "category": category,
        "brand_like": f"%{brand}%" if brand else None,
        "category_like": f"%{category}%" if category else None,
    }

    filters_sql = _build_filters(params)

    sql = f"""
    WITH params AS (
        SELECT %(embedding)s::vector AS q_embedding,
               plainto_tsquery('english', %(query)s) AS q_tsquery
    ),
    dense AS (
        SELECT id,
               1 - (embedding <=> (SELECT q_embedding FROM params)) AS score
        FROM products
        WHERE embedding IS NOT NULL
          AND {filters_sql}
        ORDER BY embedding <=> (SELECT q_embedding FROM params)
        LIMIT %(dense_k)s
    ),
    sparse AS (
        SELECT id,
               ts_rank_cd(tsv, (SELECT q_tsquery FROM params)) AS score
        FROM products
        WHERE tsv IS NOT NULL
          AND (SELECT q_tsquery FROM params) @@ tsv
          AND {filters_sql}
        ORDER BY score DESC
        LIMIT %(sparse_k)s
    ),
    dense_norm AS (
        SELECT id,
               CASE
                   WHEN max(score) OVER () = min(score) OVER () THEN 1.0
                   ELSE (score - min(score) OVER ()) / NULLIF(max(score) OVER () - min(score) OVER (), 0)
               END AS score
        FROM dense
    ),
    sparse_norm AS (
        SELECT id,
               CASE
                   WHEN max(score) OVER () = min(score) OVER () THEN 1.0
                   ELSE (score - min(score) OVER ()) / NULLIF(max(score) OVER () - min(score) OVER (), 0)
               END AS score
        FROM sparse
    ),
    merged AS (
        SELECT COALESCE(d.id, s.id) AS id,
               COALESCE(d.score, 0) AS dense_score,
               COALESCE(s.score, 0) AS sparse_score
        FROM dense_norm d
        FULL OUTER JOIN sparse_norm s USING (id)
    )
    SELECT p.id,
           p.title,
           p.brand,
           p.description,
           p.categories,
           p.price,
           p.currency,
           m.dense_score,
           m.sparse_score,
           (%(alpha)s * m.dense_score + %(beta)s * m.sparse_score) AS final_score
    FROM merged m
    JOIN products p ON p.id = m.id
    ORDER BY final_score DESC
    LIMIT %(k)s;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for r in rows:
        results.append(
            {
                "id": r[0],
                "title": r[1],
                "brand": r[2],
                "description": r[3],
                "categories": r[4],
                "price": float(r[5]) if r[5] is not None else None,
                "currency": r[6],
                "dense_score": float(r[7]) if r[7] is not None else 0.0,
                "sparse_score": float(r[8]) if r[8] is not None else 0.0,
                "final_score": float(r[9]) if r[9] is not None else 0.0,
            }
        )

    return results


def search_products(
    query: str,
    k: int = 10,
    alpha: float = 0.5,
    beta: float = 0.5,
    dense_k: int = 50,
    sparse_k: int = 50,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    currency: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    model: str = "nomic-embed-text",
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return hybrid_search(
        query=query,
        k=k,
        alpha=alpha,
        beta=beta,
        dense_k=dense_k,
        sparse_k=sparse_k,
        price_min=price_min,
        price_max=price_max,
        currency=currency,
        brand=brand,
        category=category,
        model=model,
        base_url=base_url,
    )
