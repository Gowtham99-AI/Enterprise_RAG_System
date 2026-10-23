"""add document content hash"""

import sqlalchemy as sa
from alembic import op

revision = "4f38e3ddfc5d"
down_revision = "179393344892"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    op.create_unique_constraint(
        "uq_documents_content_hash",
        "documents",
        ["content_hash"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_documents_content_hash",
        "documents",
        type_="unique",
    )
    op.drop_column("documents", "content_hash")
