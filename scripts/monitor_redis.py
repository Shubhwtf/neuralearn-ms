#!/usr/bin/env python3
"""
Redis queue monitoring script.

Monitors Redis queues, worker status, and job metrics in real-time.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from rq import Queue, Worker, Connection
from rq.job import Job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry


def get_redis_connection():
    """Get Redis connection."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(redis_url)


def get_queue_stats(queue_name: str, conn: redis.Redis) -> Dict:
    """Get statistics for a specific queue."""
    queue = Queue(queue_name, connection=conn)
    
    # Get job counts
    queued = len(queue)
    started_registry = StartedJobRegistry(queue_name, connection=conn)
    finished_registry = FinishedJobRegistry(queue_name, connection=conn)
    failed_registry = FailedJobRegistry(queue_name, connection=conn)
    
    started = len(started_registry)
    finished = len(finished_registry)
    failed = len(failed_registry)
    
    # Get worker count
    workers = Worker.all(queue=queue)
    worker_count = len(workers)
    
    # Get oldest job in queue
    oldest_job_id = None
    oldest_job_age = None
    if queued > 0:
        job_ids = queue.job_ids
        if job_ids:
            oldest_job_id = job_ids[0]
            try:
                job = Job.fetch(oldest_job_id, connection=conn)
                if job.created_at:
                    oldest_job_age = (datetime.now() - job.created_at).total_seconds()
            except:
                pass
    
    return {
        "queue_name": queue_name,
        "queued": queued,
        "started": started,
        "finished": finished,
        "failed": failed,
        "workers": worker_count,
        "oldest_job_age": oldest_job_age,
        "total": queued + started + finished + failed
    }


def get_worker_info(conn: redis.Redis) -> List[Dict]:
    """Get information about all workers."""
    workers = Worker.all(connection=conn)
    worker_info = []
    
    for worker in workers:
        worker_info.append({
            "name": worker.name,
            "queues": ", ".join([q.name for q in worker.queues]),
            "state": worker.get_state(),
            "current_job": worker.get_current_job_id(),
            "birth_date": worker.birth_date.isoformat() if worker.birth_date else None,
        })
    
    return worker_info


def get_redis_info(conn: redis.Redis) -> Dict:
    """Get Redis server information."""
    try:
        info = conn.info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "N/A"),
            "used_memory_peak_human": info.get("used_memory_peak_human", "N/A"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
        }
    except:
        return {}


def print_stats(stats: Dict, worker_info: List[Dict], redis_info: Dict):
    """Print formatted statistics."""
    print("\n" + "="*70)
    print(f"📊 REDIS QUEUE MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Queue statistics
    print("\n📋 QUEUE STATISTICS:")
    print("-" * 70)
    for queue_name in ["eda_fast", "eda_deep"]:
        if queue_name in stats:
            q = stats[queue_name]
            print(f"\n  Queue: {q['queue_name']}")
            print(f"    📤 Queued:      {q['queued']:>5}")
            print(f"    ⚙️  Started:     {q['started']:>5}")
            print(f"    ✅ Finished:    {q['finished']:>5}")
            print(f"    ❌ Failed:      {q['failed']:>5}")
            print(f"    👷 Workers:     {q['workers']:>5}")
            if q['oldest_job_age']:
                print(f"    ⏰ Oldest job:  {q['oldest_job_age']:.1f}s ago")
            print(f"    📊 Total jobs:  {q['total']:>5}")
    
    # Worker information
    print("\n👷 WORKER INFORMATION:")
    print("-" * 70)
    if worker_info:
        for w in worker_info:
            print(f"\n  Worker: {w['name']}")
            print(f"    Queues:     {w['queues']}")
            print(f"    State:      {w['state']}")
            print(f"    Current job: {w['current_job'] or 'idle'}")
            if w['birth_date']:
                print(f"    Started:    {w['birth_date']}")
    else:
        print("  ⚠️  No active workers found!")
    
    # Redis server info
    if redis_info:
        print("\n🔧 REDIS SERVER:")
        print("-" * 70)
        print(f"  Connected clients:  {redis_info.get('connected_clients', 'N/A')}")
        print(f"  Memory used:        {redis_info.get('used_memory_human', 'N/A')}")
        print(f"  Memory peak:        {redis_info.get('used_memory_peak_human', 'N/A')}")
        print(f"  Commands processed: {redis_info.get('total_commands_processed', 'N/A')}")
        hits = redis_info.get('keyspace_hits', 0)
        misses = redis_info.get('keyspace_misses', 0)
        if hits + misses > 0:
            hit_rate = hits / (hits + misses) * 100
            print(f"  Cache hit rate:    {hit_rate:.1f}%")
    
    print("\n" + "="*70)


def monitor_continuous(interval: int = 5):
    """Continuously monitor Redis queues."""
    print(f"🔄 Starting continuous monitoring (refresh every {interval}s)")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            conn = get_redis_connection()
            
            stats = {}
            for queue_name in ["eda_fast", "eda_deep"]:
                try:
                    stats[queue_name] = get_queue_stats(queue_name, conn)
                except Exception as e:
                    print(f"⚠️  Error getting stats for {queue_name}: {e}")
            
            worker_info = get_worker_info(conn)
            redis_info = get_redis_info(conn)
            
            print_stats(stats, worker_info, redis_info)
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


def monitor_once():
    """Monitor once and exit."""
    conn = get_redis_connection()
    
    stats = {}
    for queue_name in ["eda_fast", "eda_deep"]:
        try:
            stats[queue_name] = get_queue_stats(queue_name, conn)
        except Exception as e:
            print(f"⚠️  Error getting stats for {queue_name}: {e}")
    
    worker_info = get_worker_info(conn)
    redis_info = get_redis_info(conn)
    
    print_stats(stats, worker_info, redis_info)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Redis queues and workers")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode (continuous)")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    try:
        if args.watch:
            monitor_continuous(args.interval)
        else:
            monitor_once()
    except redis.ConnectionError:
        print("❌ Error: Could not connect to Redis!")
        print("   Make sure Redis is running and REDIS_URL is set correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
