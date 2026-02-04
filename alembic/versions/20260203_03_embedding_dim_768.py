"""Change embedding dimension to 768 for nomic-embed-text.

Revision ID: 20260203_03
Revises: 20260203_02
Create Date: 2026-02-03
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260203_03"
down_revision = "20260203_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_products_embedding;")
    op.execute("ALTER TABLE products ALTER COLUMN embedding TYPE vector(768);")
    op.execute(
        """
        CREATE INDEX idx_products_embedding
        ON products
        USING hnsw (embedding vector_cosine_ops);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_products_embedding;")
    op.execute("ALTER TABLE products ALTER COLUMN embedding TYPE vector(1536);")
    op.execute(
        """
        CREATE INDEX idx_products_embedding
        ON products
        USING hnsw (embedding vector_cosine_ops);
        """
    )
