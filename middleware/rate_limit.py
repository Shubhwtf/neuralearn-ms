"""
Rate Limiting Middleware for NeuraLearn Microservices.

Provides per-user rate limiting using Redis for distributed rate limiting
or in-memory fallback for development.
"""

import os
import time
from typing import Optional, Dict
from dataclasses import dataclass
from collections import defaultdict
import asyncio

from fastapi import Request, HTTPException, Depends

from .auth import get_current_user, JWTPayload, AuthenticationError


# Rate limit configuration from environment
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds

# Redis URL for distributed rate limiting (optional)
REDIS_URL = os.getenv("REDIS_URL")


class RateLimitExceeded(HTTPException):
    """Exception raised when rate limit is exceeded."""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )


@dataclass
class RateLimitInfo:
    """Information about current rate limit status."""
    remaining: int
    reset: int  # Unix timestamp when limit resets
    limit: int


class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.
    Suitable for single-instance deployments or development.
    """
    
    def __init__(self, requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.requests = requests
        self.window = window
        self._timestamps: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self, user_id: str) -> RateLimitInfo:
        """
        Check if a user has exceeded their rate limit.
        
        Args:
            user_id: The user's ID
            
        Returns:
            RateLimitInfo with current status
            
        Raises:
            RateLimitExceeded if limit is exceeded
        """
        async with self._lock:
            now = time.time()
            window_start = now - self.window
            
            # Clean old timestamps
            self._timestamps[user_id] = [
                ts for ts in self._timestamps[user_id]
                if ts > window_start
            ]
            
            current_count = len(self._timestamps[user_id])
            reset_time = int(now + self.window)
            
            if current_count >= self.requests:
                # Find when the oldest request in window will expire
                oldest = min(self._timestamps[user_id]) if self._timestamps[user_id] else now
                retry_after = int(oldest + self.window - now) + 1
                raise RateLimitExceeded(retry_after=max(1, retry_after))
            
            # Record this request
            self._timestamps[user_id].append(now)
            
            return RateLimitInfo(
                remaining=self.requests - current_count - 1,
                reset=reset_time,
                limit=self.requests
            )


class RedisRateLimiter:
    """
    Redis-based rate limiter for distributed deployments.
    Uses sliding window with Redis sorted sets.
    """
    
    def __init__(self, redis_url: str, requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.requests = requests
        self.window = window
        self.redis_url = redis_url
        self._redis = None
    
    async def _get_redis(self):
        """Lazy initialization of Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
            except ImportError:
                raise RuntimeError("redis package not installed. Install with: pip install redis")
        return self._redis
    
    async def check_rate_limit(self, user_id: str) -> RateLimitInfo:
        """
        Check if a user has exceeded their rate limit using Redis.
        
        Uses sorted sets with timestamps as scores for sliding window.
        """
        redis_client = await self._get_redis()
        key = f"ratelimit:ms:{user_id}"
        now = time.time()
        window_start = now - self.window
        
        pipe = redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current entries
        pipe.zcard(key)
        # Add new entry
        pipe.zadd(key, {str(now): now})
        # Set expiry on the key
        pipe.expire(key, self.window + 1)
        
        results = await pipe.execute()
        current_count = results[1]
        
        reset_time = int(now + self.window)
        
        if current_count >= self.requests:
            # Get oldest timestamp to calculate retry-after
            oldest = await redis_client.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry_after = int(oldest[0][1] + self.window - now) + 1
            else:
                retry_after = self.window
            raise RateLimitExceeded(retry_after=max(1, retry_after))
        
        return RateLimitInfo(
            remaining=self.requests - current_count - 1,
            reset=reset_time,
            limit=self.requests
        )


class RateLimiter:
    """
    Main rate limiter class that selects the appropriate backend.
    Uses Redis if REDIS_URL is configured, otherwise falls back to in-memory.
    """
    
    _instance: Optional['RateLimiter'] = None
    
    def __init__(self):
        if REDIS_URL:
            self._backend = RedisRateLimiter(REDIS_URL)
        else:
            self._backend = InMemoryRateLimiter()
    
    @classmethod
    def get_instance(cls) -> 'RateLimiter':
        """Get singleton instance of rate limiter."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def check_rate_limit(self, user_id: str) -> RateLimitInfo:
        """Check rate limit for a user."""
        return await self._backend.check_rate_limit(user_id)


async def rate_limit_dependency(
    request: Request,
    user: JWTPayload = Depends(get_current_user),
) -> JWTPayload:
    """
    FastAPI dependency that enforces rate limiting per authenticated user.
    
    Usage:
        @app.post("/upload")
        async def upload(user: JWTPayload = Depends(rate_limit_dependency)):
            # user is authenticated and rate-limited
            pass
    
    This dependency:
    1. Authenticates the user (via get_current_user)
    2. Checks rate limit for that user
    3. Adds rate limit headers to response
    4. Returns the user payload
    """
    rate_limiter = RateLimiter.get_instance()
    
    try:
        info = await rate_limiter.check_rate_limit(user.user_id)
        
        # Store rate limit info for response headers middleware
        request.state.rate_limit_info = info
        
    except RateLimitExceeded:
        raise
    
    return user


def add_rate_limit_headers(request: Request, response):
    """
    Add rate limit headers to response.
    Call this in middleware or after route handler.
    """
    if hasattr(request.state, 'rate_limit_info'):
        info: RateLimitInfo = request.state.rate_limit_info
        response.headers["X-RateLimit-Limit"] = str(info.limit)
        response.headers["X-RateLimit-Remaining"] = str(info.remaining)
        response.headers["X-RateLimit-Reset"] = str(info.reset)
    return response
