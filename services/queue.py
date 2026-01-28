import os
from typing import Dict

import redis
from rq import Queue


_queues: Dict[str, Queue] = {}


def get_redis_connection() -> redis.Redis:
    """
    Create a Redis connection using REDIS_URL env variable.

    Defaults to redis://localhost:6379/0 if not provided.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(redis_url)


def get_queue(name: str = "eda") -> Queue:
    """
    Lazily initialized RQ queue for EDA / processing jobs.

    Supports multiple queues (e.g. eda_fast, eda_deep) sharing a
    single Redis connection.
    """
    if name not in _queues:
        conn = get_redis_connection()
        _queues[name] = Queue(name, connection=conn)
    return _queues[name]

