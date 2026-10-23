
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from enterprise_rag_platform.config import get_settings


@pytest.fixture
def db_session() -> Session:
    engine = create_engine(
        get_settings().database_url,
        pool_pre_ping=True,
    )

    connection = engine.connect()
    outer_transaction = connection.begin()

    session = Session(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield session
    finally:
        session.close()
        outer_transaction.rollback()
        connection.close()
        engine.dispose()
