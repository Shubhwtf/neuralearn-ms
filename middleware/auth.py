"""
JWT Authentication Middleware for NeuraLearn Microservices.

This middleware validates JWT tokens issued by the neuralearn-backend.
It uses the same JWT_SECRET to ensure compatibility.
"""

import os
from typing import Optional
from dataclasses import dataclass

import jwt
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# JWT Configuration - must match the backend's JWT_SECRET
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """Custom exception for authentication failures."""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=401, detail=detail)


@dataclass
class JWTPayload:
    """JWT payload structure matching the backend."""
    user_id: str
    role: Optional[str] = None
    scope: Optional[str] = None


def decode_jwt_token(token: str) -> Optional[JWTPayload]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: The JWT token string
        
    Returns:
        JWTPayload if valid, None otherwise
    """
    if not JWT_SECRET:
        raise AuthenticationError("JWT_SECRET not configured on server")
    
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["userId"]}
        )
        
        return JWTPayload(
            user_id=payload.get("userId"),
            role=payload.get("role"),
            scope=payload.get("scope"),
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> JWTPayload:
    """
    FastAPI dependency to get the current authenticated user from JWT token.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: JWTPayload = Depends(get_current_user)):
            return {"user_id": user.user_id}
    
    Args:
        request: The FastAPI request object
        credentials: The HTTP Authorization credentials
        
    Returns:
        JWTPayload containing user information
        
    Raises:
        AuthenticationError: If token is missing, invalid, or expired
    """
    # Check for token in Authorization header
    if credentials is None:
        # Also check for token in query params (for backward compatibility during migration)
        token = request.query_params.get("token")
        if not token:
            raise AuthenticationError("Missing authentication token")
    else:
        token = credentials.credentials
    
    if not token:
        raise AuthenticationError("Missing authentication token")
    
    return decode_jwt_token(token)


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[JWTPayload]:
    """
    FastAPI dependency to optionally get the current user.
    Returns None if no valid token is provided instead of raising an error.
    
    Useful for endpoints that work both authenticated and unauthenticated.
    """
    try:
        return await get_current_user(request, credentials)
    except AuthenticationError:
        return None
