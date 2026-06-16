from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

settings = get_settings()

# The engine - one per process, the connection pool itself
engine = create_async_engine(settings.DATABASE_URL, echo=False)

# The session factory - creates new sessions on demand
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """All SQLAlchemy models inherit from this."""
    pass


async def get_db():
    """FastAPI dependency - gives each request its own session, closes it after."""
    async with AsyncSessionLocal() as session:
        yield session