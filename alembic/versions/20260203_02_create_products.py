"""Create products table and indexes.

Revision ID: 20260203_02
Revises: 20260203_01
Create Date: 2026-02-03
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260203_02"
down_revision = "20260203_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE products (
            id UUID PRIMARY KEY,
            title TEXT,
            brand TEXT,
            description TEXT,
            categories TEXT,
            price NUMERIC,
            currency TEXT,

            embedding VECTOR(1536),
            tsv TSVECTOR
        );
        """
    )

    op.execute(
        """
        CREATE INDEX idx_products_embedding
        ON products
        USING hnsw (embedding vector_cosine_ops);
        """
    )

    op.execute(
        """
        CREATE INDEX idx_products_tsv
        ON products
        USING GIN (tsv);
        """
    )

    op.execute(
        """
        CREATE INDEX idx_products_price
        ON products (price);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_products_price;")
    op.execute("DROP INDEX IF EXISTS idx_products_tsv;")
    op.execute("DROP INDEX IF EXISTS idx_products_embedding;")
    op.execute("DROP TABLE IF EXISTS products;")
