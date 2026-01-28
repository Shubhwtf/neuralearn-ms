import asyncio
from typing import Optional

from database import connect_db, disconnect_db
from main import process_dataset


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
        await connect_db()
        try:
            await process_dataset(dataset_id=dataset_id, mode=mode)
            return True
        finally:
            await disconnect_db()

    return asyncio.run(_run())

