import structlog
import asyncio
from typing import Optional
from logging_config import configure_logging
from database import connect_db, disconnect_db
from main import process_dataset

configure_logging()
logger = structlog.get_logger()

def process_dataset_job(dataset_id: str, mode: str = "fast", **kwargs) -> Optional[bool]:
    """
    Synchronous wrapper to run the async EDA/cleaning pipeline.

    This is intended to be executed by an RQ worker process.
    
    Args:
        dataset_id: ID of the dataset to process
        mode: Processing mode (fast, smart, or deep)
        **kwargs: Additional arguments (ignored, but allows RQ job options to be passed)
    """

    async def _run() -> Optional[bool]:
        structlog.contextvars.bind_contextvars(dataset_id=dataset_id, mode=mode)
        await connect_db()
        logger.info("job_started", dataset_id=dataset_id, mode=mode)
        try:
            await process_dataset(dataset_id=dataset_id, mode=mode)
            logger.info("job_completed_success", dataset_id=dataset_id)
            return True
        except Exception as e:
            logger.error("job_failed_error", exc_info=True, dataset_id=dataset_id)
            raise e
        finally:
            await disconnect_db()
            structlog.contextvars.clear_contextvars()

    return asyncio.run(_run())

