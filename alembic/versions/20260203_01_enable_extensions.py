"""Enable pgvector and pg_trgm extensions.

Revision ID: 20260203_01
Revises: 
Create Date: 2026-02-03
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260203_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_trgm;")
    op.execute("DROP EXTENSION IF EXISTS vector;")
