#!/usr/bin/env python3
"""
Load testing script for Redis queue system.

Generates multiple concurrent dataset uploads to test Redis queue scalability.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import time
import sys
from pathlib import Path
from typing import List, Dict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE_URL = "http://localhost:8000"


def generate_test_csv(rows: int = 1000, columns: int = 10) -> str:
    """Generate a test CSV file in memory."""
    data = {}
    for i in range(columns):
        if i % 3 == 0:
            # Simple numeric column with a bit of noise
            base = np.arange(rows)
            noise = np.random.normal(loc=0.0, scale=1.0, size=rows)
            data[f"numeric_col_{i}"] = base + noise
        elif i % 3 == 1:
            data[f"categorical_col_{i}"] = pd.Series(["A", "B", "C", "D"] * (rows // 4 + 1))[:rows]
        else:
            # Monotonic float column
            data[f"float_col_{i}"] = np.linspace(0, rows * 1.5, rows)
    
    df = pd.DataFrame(data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


async def upload_dataset(
    session: aiohttp.ClientSession,
    csv_content: str,
    user_id: str,
    mode: str = "fast",
    dataset_num: int = 0
) -> Dict:
    """Upload a single dataset and return response."""
    start_time = time.time()
    
    data = aiohttp.FormData()
    data.add_field('file', csv_content.encode('utf-8'), filename=f'test_dataset_{dataset_num}.csv', content_type='text/csv')
    data.add_field('user_id', user_id)
    data.add_field('mode', mode)
    
    try:
        async with session.post(f"{API_BASE_URL}/dataset/upload", data=data) as response:
            result = await response.json()
            elapsed = time.time() - start_time
            return {
                "success": response.status == 200,
                "status_code": response.status,
                "dataset_id": result.get("dataset_id"),
                "elapsed_time": elapsed,
                "response": result
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed
        }


async def check_status(session: aiohttp.ClientSession, dataset_id: str) -> Dict:
    """Check the status of a dataset."""
    try:
        async with session.get(f"{API_BASE_URL}/dataset/{dataset_id}/status") as response:
            if response.status == 200:
                return await response.json()
            return {"error": f"Status {response.status}"}
    except Exception as e:
        return {"error": str(e)}


async def run_load_test(
    num_datasets: int = 50,
    concurrent_uploads: int = 10,
    mode: str = "fast",
    user_id: str = "load_test_user"
):
    """Run load test with concurrent uploads."""
    print(f"\n🚀 Starting load test:")
    print(f"   - Datasets to upload: {num_datasets}")
    print(f"   - Concurrent uploads: {concurrent_uploads}")
    print(f"   - Mode: {mode}")
    print(f"   - API: {API_BASE_URL}\n")
    
    csv_content = generate_test_csv(rows=1000, columns=10)
    
    results: List[Dict] = []
    dataset_ids: List[str] = []
    
    async with aiohttp.ClientSession() as session:
        # Phase 1: Upload datasets
        print("📤 Uploading datasets...")
        upload_start = time.time()
        
        semaphore = asyncio.Semaphore(concurrent_uploads)
        
        async def upload_with_limit(dataset_num: int):
            async with semaphore:
                result = await upload_dataset(session, csv_content, user_id, mode, dataset_num)
                results.append(result)
                if result.get("success") and result.get("dataset_id"):
                    dataset_ids.append(result["dataset_id"])
                return result
        
        upload_tasks = [upload_with_limit(i) for i in range(num_datasets)]
        await asyncio.gather(*upload_tasks)
        
        upload_elapsed = time.time() - upload_start
        
        # Phase 2: Monitor job processing
        print(f"\n✅ Upload phase complete in {upload_elapsed:.2f}s")
        print(f"   - Successful uploads: {len(dataset_ids)}/{num_datasets}")
        print(f"\n📊 Monitoring job processing (checking every 5s, max 5min)...")
        
        if not dataset_ids:
            print("❌ No successful uploads to monitor!")
            return
        
        max_wait_time = 300  # 5 minutes
        check_interval = 5
        start_monitor = time.time()
        
        status_counts = {"uploaded": len(dataset_ids), "processing": 0, "completed": 0, "failed": 0}
        
        while time.time() - start_monitor < max_wait_time:
            await asyncio.sleep(check_interval)
            
            # Check all datasets, but in batches to avoid overwhelming the API
            batch_size = 50
            all_statuses = []
            
            for i in range(0, len(dataset_ids), batch_size):
                batch_ids = dataset_ids[i:i + batch_size]
                status_checks = [check_status(session, did) for did in batch_ids]
                batch_statuses = await asyncio.gather(*status_checks)
                all_statuses.extend(batch_statuses)
                # Small delay between batches to avoid rate limiting
                if i + batch_size < len(dataset_ids):
                    await asyncio.sleep(0.5)
            
            # Count actual statuses
            current_counts = {"uploaded": 0, "processing": 0, "completed": 0, "failed": 0}
            for status_data in all_statuses:
                if "status" in status_data:
                    status = status_data["status"]
                    current_counts[status] = current_counts.get(status, 0) + 1
            
            status_counts = current_counts
            
            elapsed = time.time() - start_monitor
            total_processed = status_counts["completed"] + status_counts["failed"]
            print(f"   [{elapsed:.0f}s] Status: uploaded={status_counts['uploaded']}, "
                  f"processing={status_counts['processing']}, "
                  f"completed={status_counts['completed']}, "
                  f"failed={status_counts['failed']} "
                  f"({total_processed}/{len(dataset_ids)} processed)")
            
            if total_processed >= len(dataset_ids) * 0.9:
                print("\n✅ Most jobs completed!")
                break
        
        # Summary
        print("\n" + "="*60)
        print("📈 LOAD TEST SUMMARY")
        print("="*60)
        
        successful_uploads = sum(1 for r in results if r.get("success"))
        failed_uploads = num_datasets - successful_uploads
        
        avg_upload_time = sum(r.get("elapsed_time", 0) for r in results) / len(results) if results else 0
        
        print(f"\nUpload Phase:")
        print(f"  ✅ Successful: {successful_uploads}/{num_datasets}")
        print(f"  ❌ Failed: {failed_uploads}")
        print(f"  ⏱️  Total time: {upload_elapsed:.2f}s")
        print(f"  ⚡ Avg upload time: {avg_upload_time:.3f}s")
        print(f"  📊 Throughput: {successful_uploads/upload_elapsed:.2f} uploads/sec")
        
        print(f"\nProcessing Phase (sampled):")
        print(f"  📤 Uploaded: {status_counts['uploaded']}")
        print(f"  ⚙️  Processing: {status_counts['processing']}")
        print(f"  ✅ Completed: {status_counts['completed']}")
        print(f"  ❌ Failed: {status_counts['failed']}")
        
        if failed_uploads > 0:
            print(f"\n⚠️  Failed uploads:")
            for i, r in enumerate(results):
                if not r.get("success"):
                    print(f"   - Dataset {i}: {r.get('error', r.get('status_code'))}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test Redis queue system")
    parser.add_argument("--num", type=int, default=50, help="Number of datasets to upload (default: 50)")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrent uploads (default: 10)")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "smart", "deep"], help="Processing mode")
    parser.add_argument("--user", type=str, default="load_test_user", help="User ID for testing")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_load_test(
            num_datasets=args.num,
            concurrent_uploads=args.concurrent,
            mode=args.mode,
            user_id=args.user
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Load test failed: {e}")
        import traceback
        traceback.print_exc()
