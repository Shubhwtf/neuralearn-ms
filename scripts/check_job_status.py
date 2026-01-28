#!/usr/bin/env python3
"""
Check actual job completion status by querying the database.
"""

import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_dataset, list_datasets


async def check_job_status():
    """Check dataset status from database to see if jobs are completing."""
    datasets = await list_datasets()
    
    status_counts = {"uploaded": 0, "processing": 0, "completed": 0, "failed": 0}
    
    # Sample recent datasets
    recent = sorted(datasets, key=lambda x: x.createdAt, reverse=True)[:100]
    
    for ds in recent:
        status_counts[ds.status] = status_counts.get(ds.status, 0) + 1
    
    print("\n📊 DATASET STATUS FROM DATABASE (last 100):")
    print("=" * 60)
    print(f"  📤 Uploaded:   {status_counts.get('uploaded', 0)}")
    print(f"  ⚙️  Processing: {status_counts.get('processing', 0)}")
    print(f"  ✅ Completed:  {status_counts.get('completed', 0)}")
    print(f"  ❌ Failed:     {status_counts.get('failed', 0)}")
    print("=" * 60)
    
    # Check processing time
    processing = [ds for ds in recent if ds.status == "processing"]
    if processing:
        print(f"\n⚠️  {len(processing)} datasets stuck in 'processing' state")
        oldest = min(processing, key=lambda x: x.updatedAt)
        print(f"   Oldest processing: {oldest.id} (updated {oldest.updatedAt})")
    
    completed = [ds for ds in recent if ds.status == "completed"]
    if completed:
        print(f"\n✅ {len(completed)} datasets completed successfully")
        newest = max(completed, key=lambda x: x.updatedAt)
        print(f"   Most recent: {newest.id} (completed {newest.updatedAt})")


if __name__ == "__main__":
    from database import connect_db, disconnect_db
    
    async def run():
        await connect_db()
        try:
            await check_job_status()
        finally:
            await disconnect_db()
    
    asyncio.run(run())
