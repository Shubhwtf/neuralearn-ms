#!/usr/bin/env python3
"""
Redis Queue Autoscaler

Monitors queue lengths and automatically starts/stops workers based on thresholds.
Uses subprocess to manage worker processes.
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis
from rq import Queue


class WorkerManager:
    """Manages worker processes for a queue."""
    
    def __init__(self, queue_name: str, min_workers: int = 1, max_workers: int = 10):
        self.queue_name = queue_name
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.workers: List[subprocess.Popen] = []
    
    def start_worker(self) -> bool:
        """Start a new worker process."""
        if len(self.workers) >= self.max_workers:
            return False
        
        try:
            proc = subprocess.Popen(
                ["rq", "worker", self.queue_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            self.workers.append(proc)
            print(f"  ✅ Started worker for {self.queue_name} (PID: {proc.pid}, total: {len(self.workers)})")
            return True
        except Exception as e:
            print(f"  ❌ Failed to start worker for {self.queue_name}: {e}")
            return False
    
    def stop_worker(self) -> bool:
        """Stop the oldest worker process."""
        if len(self.workers) <= self.min_workers:
            return False
        
        if not self.workers:
            return False
        
        try:
            proc = self.workers.pop(0)
            # Kill the process group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            print(f"  🛑 Stopped worker for {self.queue_name} (PID: {proc.pid}, remaining: {len(self.workers)})")
            return True
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            print(f"  🛑 Force-killed worker for {self.queue_name} (PID: {proc.pid})")
            return True
        except Exception as e:
            print(f"  ❌ Failed to stop worker for {self.queue_name}: {e}")
            return False
    
    def cleanup_dead_workers(self):
        """Remove dead worker processes from the list."""
        self.workers = [w for w in self.workers if w.poll() is None]
    
    def get_active_count(self) -> int:
        """Get number of active workers."""
        self.cleanup_dead_workers()
        return len(self.workers)
    
    def shutdown_all(self):
        """Stop all workers."""
        for proc in self.workers:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass
        for proc in self.workers:
            try:
                proc.wait(timeout=5)
            except:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        self.workers.clear()


class Autoscaler:
    """Autoscales workers based on queue length."""
    
    def __init__(
        self,
        scale_up_threshold: int = 50,      # Start new worker when queue > this
        scale_down_threshold: int = 10,    # Stop worker when queue < this
        workers_per_scale: int = 1,        # How many workers to add/remove
        check_interval: int = 10,          # Check every N seconds
    ):
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.workers_per_scale = workers_per_scale
        self.check_interval = check_interval
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.conn = redis.from_url(redis_url)
        
        # Initialize worker managers
        self.managers = {
            "eda_fast": WorkerManager("eda_fast", min_workers=1, max_workers=20),
            "eda_deep": WorkerManager("eda_deep", min_workers=0, max_workers=5),
        }
        
        self.running = True
    
    def get_queue_length(self, queue_name: str) -> int:
        """Get current queue length."""
        try:
            queue = Queue(queue_name, connection=self.conn)
            return len(queue)
        except Exception as e:
            print(f"  ⚠️  Error getting queue length for {queue_name}: {e}")
            return 0
    
    def scale_queue(self, queue_name: str):
        """Scale workers for a specific queue."""
        manager = self.managers[queue_name]
        queue_length = self.get_queue_length(queue_name)
        active_workers = manager.get_active_count()
        
        # Scale up logic
        if queue_length > self.scale_up_threshold:
            needed = min(
                (queue_length // self.scale_up_threshold) * self.workers_per_scale,
                manager.max_workers - active_workers
            )
            if needed > 0:
                print(f"  📈 {queue_name}: Queue={queue_length}, Workers={active_workers} → Scaling UP by {needed}")
                for _ in range(needed):
                    manager.start_worker()
        
        # Scale down logic (only if queue is low and we have more than min)
        elif queue_length < self.scale_down_threshold and active_workers > manager.min_workers:
            excess = active_workers - manager.min_workers
            to_remove = min(self.workers_per_scale, excess)
            if to_remove > 0:
                print(f"  📉 {queue_name}: Queue={queue_length}, Workers={active_workers} → Scaling DOWN by {to_remove}")
                for _ in range(to_remove):
                    manager.stop_worker()
    
    def run(self):
        """Main autoscaler loop."""
        print("🚀 Starting Redis Queue Autoscaler")
        print(f"   Scale up threshold: {self.scale_up_threshold} jobs")
        print(f"   Scale down threshold: {self.scale_down_threshold} jobs")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Workers per scale: {self.workers_per_scale}")
        print("\n   Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Checking queues...")
                
                for queue_name in self.managers.keys():
                    self.scale_queue(queue_name)
                
                # Show summary
                print("\n📊 Current Status:")
                for queue_name, manager in self.managers.items():
                    queue_length = self.get_queue_length(queue_name)
                    active_workers = manager.get_active_count()
                    print(f"  {queue_name}: {queue_length} queued, {active_workers} workers")
                
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down autoscaler...")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown all workers."""
        print("Stopping all workers...")
        for manager in self.managers.values():
            manager.shutdown_all()
        print("✅ All workers stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Autoscale Redis queue workers")
    parser.add_argument("--scale-up", type=int, default=50, help="Scale up when queue > this (default: 50)")
    parser.add_argument("--scale-down", type=int, default=10, help="Scale down when queue < this (default: 10)")
    parser.add_argument("--workers-per-scale", type=int, default=1, help="Workers to add/remove per scale (default: 1)")
    parser.add_argument("--interval", type=int, default=10, help="Check interval in seconds (default: 10)")
    
    args = parser.parse_args()
    
    autoscaler = Autoscaler(
        scale_up_threshold=args.scale_up,
        scale_down_threshold=args.scale_down,
        workers_per_scale=args.workers_per_scale,
        check_interval=args.interval,
    )
    
    autoscaler.run()
