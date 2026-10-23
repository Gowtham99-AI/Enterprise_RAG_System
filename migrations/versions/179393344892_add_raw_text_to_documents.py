
"""Add raw text to documents."""

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision: str = "179393344892"
down_revision: str | None = "bc330f518ba1"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Add the raw text column without recreating existing tables."""
    op.add_column(
        "documents",
        sa.Column("raw_text", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Remove the raw text column."""
    op.drop_column("documents", "raw_text")
