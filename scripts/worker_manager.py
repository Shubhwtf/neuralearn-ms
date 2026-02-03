#!/usr/bin/env python3
"""
Production Worker Manager for NeuraLearn

Manages multiple RQ worker processes within a single container.
"""

import os
import sys
import time
import signal
import subprocess
import structlog
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure structlog for JSON output
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

@dataclass
class WorkerConfig:
    queue_name: str
    count: int

class WorkerManager:
    def __init__(self, configs: List[WorkerConfig]):
        self.configs = configs
        self.workers: Dict[int, Dict] = {}  # pid -> {process, queue_name}
        self.running = True
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        if not self.shutdown_requested:
            logger.info("shutdown_signal_received", signal=sig_name)
            self.shutdown_requested = True
            self.running = False

    def start_worker(self, queue_name: str) -> bool:
        try:
            # Use os.setsid to create a process group for easier cleanup if needed
            proc = subprocess.Popen(
                ["rq", "worker", queue_name],
                stdout=sys.stdout,
                stderr=sys.stderr,
                preexec_fn=os.setsid
            )
            self.workers[proc.pid] = {
                "process": proc,
                "queue": queue_name,
                "started_at": time.time()
            }
            logger.info("worker_started", queue=queue_name, pid=proc.pid)
            return True
        except Exception as e:
            logger.error("worker_start_failed", queue=queue_name, error=str(e))
            return False

    def stop_all_workers(self):
        logger.info("stopping_all_workers", count=len(self.workers))
        
        # Send SIGTERM to all workers
        for pid, info in self.workers.items():
            try:
                proc = info["process"]
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as e:
                logger.error("worker_stop_error", pid=pid, error=str(e))

        # Wait for them to exit
        start_wait = time.time()
        while self.workers and time.time() - start_wait < 10:
            pids_to_remove = []
            for pid, info in self.workers.items():
                if info["process"].poll() is not None:
                    pids_to_remove.append(pid)
            
            for pid in pids_to_remove:
                del self.workers[pid]
            
            if self.workers:
                time.sleep(0.5)

        # Force kill remaining
        if self.workers:
            logger.warning("force_killing_workers", count=len(self.workers))
            for pid, info in self.workers.items():
                try:
                    os.killpg(os.getpgid(info["process"].pid), signal.SIGKILL)
                except Exception:
                    pass
            self.workers.clear()

    def run(self):
        logger.info("worker_manager_started")
        
        # Initial startup
        for config in self.configs:
            for _ in range(config.count):
                self.start_worker(config.queue_name)

        while self.running:
            try:
                # Check for dead workers
                dead_pids = []
                for pid, info in self.workers.items():
                    if info["process"].poll() is not None:
                        dead_pids.append(pid)
                        exit_code = info["process"].poll()
                        logger.warning(
                            "worker_died", 
                            pid=pid, 
                            queue=info["queue"], 
                            exit_code=exit_code
                        )

                # Cleanup dead workers and restart
                for pid in dead_pids:
                    info = self.workers.pop(pid)
                    if not self.shutdown_requested:
                        # Add a small delay/backoff could be good here but keeping it simple
                        logger.info("restarting_worker", queue=info["queue"])
                        self.start_worker(info["queue"])

                time.sleep(1)

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.error("manager_loop_error", error=str(e))
                time.sleep(1)

        self.stop_all_workers()
        logger.info("worker_manager_stopped")

if __name__ == "__main__":
    # Configuration from environment or defaults
    # WORKER_COUNTS="eda_fast:2,eda_deep:1"
    env_counts = os.getenv("WORKER_COUNTS", "eda_fast:2,eda_deep:2")
    
    configs = []
    try:
        for entry in env_counts.split(","):
            queue, count = entry.split(":")
            configs.append(WorkerConfig(queue_name=queue.strip(), count=int(count)))
    except ValueError as e:
        logger.error("invalid_configuration", error=str(e))
        sys.exit(1)

    manager = WorkerManager(configs)
    manager.run()
