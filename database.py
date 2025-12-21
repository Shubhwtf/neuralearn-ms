import os
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from datetime import datetime
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

load_dotenv()


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "Dataset"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    createdAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updatedAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    userId: Mapped[str] = mapped_column(String, nullable=False, index=True)
    collaborationId: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    rows: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    rawDataPath: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cleanedDataPath: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rawUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cleanedUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    cleaningLogs: Mapped[List["CleaningLog"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    outlierLogs: Mapped[List["OutlierLog"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    featureLogs: Mapped[List["FeatureLog"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    graphs: Mapped[List["GraphMetadata"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class CleaningLog(Base):
    __tablename__ = "CleaningLog"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    datasetId: Mapped[str] = mapped_column(
        String, ForeignKey("Dataset.id", ondelete="CASCADE"), index=True
    )
    column: Mapped[str] = mapped_column(String, nullable=False)
    nullCount: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    createdAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    dataset: Mapped[Dataset] = relationship(back_populates="cleaningLogs")


class OutlierLog(Base):
    __tablename__ = "OutlierLog"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    datasetId: Mapped[str] = mapped_column(
        String, ForeignKey("Dataset.id", ondelete="CASCADE"), index=True
    )
    column: Mapped[str] = mapped_column(String, nullable=False)
    outlierCount: Mapped[int] = mapped_column(Integer, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    createdAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    dataset: Mapped[Dataset] = relationship(back_populates="outlierLogs")


class FeatureLog(Base):
    __tablename__ = "FeatureLog"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    datasetId: Mapped[str] = mapped_column(
        String, ForeignKey("Dataset.id", ondelete="CASCADE"), index=True
    )
    action: Mapped[str] = mapped_column(String, nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    createdAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    dataset: Mapped[Dataset] = relationship(back_populates="featureLogs")


class GraphMetadata(Base):
    __tablename__ = "GraphMetadata"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    datasetId: Mapped[str] = mapped_column(
        String, ForeignKey("Dataset.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String, nullable=False)
    column: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    filePath: Mapped[str] = mapped_column(String, nullable=False)
    createdAt: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    dataset: Mapped[Dataset] = relationship(back_populates="graphs")


def _build_async_url(database_url: str) -> str:
    if not database_url:
        return database_url

    if database_url.startswith("postgresql+asyncpg://"):
        parsed = urlparse(database_url)
        query = [(k, v) for k, v in parse_qsl(parsed.query) if k != "sslmode"]
        cleaned = parsed._replace(query=urlencode(query))
        return urlunparse(cleaned)

    if database_url.startswith("postgresql://"):
        base = "postgresql+asyncpg://" + database_url[len("postgresql://") :]
    elif database_url.startswith("postgres://"):
        base = "postgresql+asyncpg://" + database_url[len("postgres://") :]
    else:
        return database_url

    parsed = urlparse(base)
    query = [(k, v) for k, v in parse_qsl(parsed.query) if k != "sslmode"]
    cleaned = parsed._replace(query=urlencode(query))
    return urlunparse(cleaned)


_raw_db_url = os.getenv("DATABASE_URL", "")
if _raw_db_url.startswith("DATABASE_URL="):
    DATABASE_URL = _raw_db_url[len("DATABASE_URL="):].strip()
else:
    DATABASE_URL = _raw_db_url.strip()

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is not set or is empty. "
        "Please set DATABASE_URL to a valid PostgreSQL connection string."
    )

ASYNC_DATABASE_URL = _build_async_url(DATABASE_URL)

if not ASYNC_DATABASE_URL:
    raise ValueError(
        f"Failed to build async database URL from DATABASE_URL. "
        f"Original URL: {DATABASE_URL[:50]}..." if len(DATABASE_URL) > 50 else DATABASE_URL
    )

engine = create_async_engine(ASYNC_DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

async def connect_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def disconnect_db():
    await engine.dispose()


async def create_dataset(
    rows: int,
    columns: int,
    status: str,
    user_id: str,
    collaboration_id: str | None = None,
    raw_data_path: str | None = None,
) -> Dataset:
    from uuid import uuid4

    async with AsyncSessionLocal() as session:
        dataset = Dataset(
            id=str(uuid4()),
            userId=user_id,
            collaborationId=collaboration_id,
            rows=rows,
            columns=columns,
            status=status,
            rawDataPath=raw_data_path,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        return dataset


async def update_dataset(dataset_id: str, **kwargs) -> Dataset | None:
    async with AsyncSessionLocal() as session:
        dataset = await session.get(Dataset, dataset_id)
        if not dataset:
            return None
        for key, value in kwargs.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
        await session.commit()
        await session.refresh(dataset)
        return dataset


async def get_dataset(dataset_id: str) -> Dataset | None:
    async with AsyncSessionLocal() as session:
        return await session.get(Dataset, dataset_id)


async def list_datasets(
    user_id: str | None = None, collaboration_id: str | None = None
) -> List[Dataset]:
    async with AsyncSessionLocal() as session:
        stmt = select(Dataset)
        if collaboration_id is not None:
            stmt = stmt.where(Dataset.collaborationId == collaboration_id)
        if user_id is not None:
            stmt = stmt.where(Dataset.userId == user_id)

        result = await session.execute(stmt)
        return list(result.scalars().all())


async def create_cleaning_log(
    dataset_id: str, column: str, null_count: int, action: str, reason: str
) -> CleaningLog:
    from uuid import uuid4

    async with AsyncSessionLocal() as session:
        log = CleaningLog(
            id=str(uuid4()),
            datasetId=dataset_id,
            column=column,
            nullCount=null_count,
            action=action,
            reason=reason,
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
        return log


async def create_outlier_log(
    dataset_id: str,
    column: str,
    outlier_count: int,
    method: str,
    action: str,
) -> OutlierLog:
    from uuid import uuid4

    async with AsyncSessionLocal() as session:
        log = OutlierLog(
            id=str(uuid4()),
            datasetId=dataset_id,
            column=column,
            outlierCount=outlier_count,
            method=method,
            action=action,
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
        return log


async def create_feature_log(
    dataset_id: str, action: str, details: str
) -> FeatureLog:
    from uuid import uuid4

    async with AsyncSessionLocal() as session:
        log = FeatureLog(
            id=str(uuid4()),
            datasetId=dataset_id,
            action=action,
            details=details,
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
        return log


async def create_graph_metadata(
    dataset_id: str,
    graph_type: str,
    column: str | None,
    file_path: str,
) -> GraphMetadata:
    from uuid import uuid4

    async with AsyncSessionLocal() as session:
        graph = GraphMetadata(
            id=str(uuid4()),
            datasetId=dataset_id,
            type=graph_type,
            column=column,
            filePath=file_path,
        )
        session.add(graph)
        await session.commit()
        await session.refresh(graph)
        return graph


async def get_cleaning_logs(dataset_id: str) -> List[CleaningLog]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(CleaningLog).where(CleaningLog.datasetId == dataset_id)
        )
        return list(result.scalars().all())


async def get_outlier_logs(dataset_id: str) -> List[OutlierLog]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(OutlierLog).where(OutlierLog.datasetId == dataset_id)
        )
        return list(result.scalars().all())


async def get_feature_logs(dataset_id: str) -> List[FeatureLog]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(FeatureLog).where(FeatureLog.datasetId == dataset_id)
        )
        return list(result.scalars().all())


async def get_graphs_metadata(dataset_id: str) -> List[GraphMetadata]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(GraphMetadata).where(GraphMetadata.datasetId == dataset_id)
        )
        return list(result.scalars().all())


async def get_graph_metadata_by_id(graph_id: str) -> GraphMetadata | None:
    async with AsyncSessionLocal() as session:
        return await session.get(GraphMetadata, graph_id)
