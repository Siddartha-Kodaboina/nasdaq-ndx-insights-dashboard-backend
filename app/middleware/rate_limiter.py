"""
Rate limiting middleware for the FastAPI application.

This middleware implements rate limiting to prevent API abuse,
using a token bucket algorithm to control request rates.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Tuple, Optional, Callable
import time
import logging
import asyncio
from collections import defaultdict
from starlette.types import ASGIApp, Receive, Scope, Send

# Configure logger
logger = logging.getLogger(__name__)

class TokenBucket:
    """
    Implementation of the Token Bucket algorithm for rate limiting.
    
    The token bucket algorithm allows for bursts of requests up to a certain limit,
    while still enforcing an average rate limit over time.
    """
    
    def __init__(self, tokens: int, refill_rate: float):
        """
        Initialize a token bucket.
        
        Args:
            tokens: The maximum number of tokens the bucket can hold
            refill_rate: The rate at which tokens are refilled (tokens per second)
        """
        self.tokens = tokens
        self.capacity = tokens
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: The number of tokens to consume (default: 1)
            
        Returns:
            bool: True if tokens were consumed, False if not enough tokens
        """
        async with self.lock:
            # Refill tokens based on time elapsed
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # Check if enough tokens are available
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    async def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate the wait time until enough tokens are available.
        
        Args:
            tokens: The number of tokens needed (default: 1)
            
        Returns:
            float: The wait time in seconds
        """
        async with self.lock:
            # If enough tokens are already available
            if tokens <= self.tokens:
                return 0
            
            # Calculate time needed to refill enough tokens
            needed_tokens = tokens - self.tokens
            wait_time = needed_tokens / self.refill_rate
            return wait_time

class RateLimiter:
    """
    Rate limiter for API endpoints.
    
    This class manages rate limits for different clients and endpoints,
    using the token bucket algorithm to enforce limits.
    """
    
    def __init__(self):
        """Initialize the rate limiter."""
        # Store token buckets for each client and endpoint
        # Format: {client_id: {endpoint: TokenBucket}}
        self.buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        
        # Default rate limits
        self.default_limit = 100  # requests
        self.default_period = 60  # seconds
        
        # Endpoint-specific rate limits
        # Format: {endpoint_pattern: (limit, period)}
        self.endpoint_limits: Dict[str, Tuple[int, int]] = {
            # High-frequency endpoints
            "/tasks/": (200, 60),  # 200 requests per minute
            "/data/": (300, 60),   # 300 requests per minute
            
            # Low-frequency endpoints
            "/tasks/create": (30, 60),  # 30 requests per minute
            "/tasks/aggregated": (50, 60),  # 50 requests per minute
        }
    
    def get_client_id(self, request: Request) -> str:
        """
        Get a unique identifier for the client.
        
        This method tries to identify the client using various headers,
        falling back to IP address if no other identifier is available.
        
        Args:
            request: The FastAPI request object
            
        Returns:
            str: A unique identifier for the client
        """
        # Try to get API key from header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"
        
        # Fall back to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def get_endpoint_pattern(self, path: str) -> str:
        """
        Get the endpoint pattern for a given path.
        
        This method matches the path to the most specific endpoint pattern
        defined in the endpoint_limits dictionary.
        
        Args:
            path: The request path
            
        Returns:
            str: The matching endpoint pattern, or "default" if none match
        """
        # Check for exact matches first
        if path in self.endpoint_limits:
            return path
        
        # Check for prefix matches
        for pattern in self.endpoint_limits:
            if path.startswith(pattern):
                return pattern
        
        # Fall back to default
        return "default"
    
    def get_bucket(self, client_id: str, endpoint: str) -> TokenBucket:
        """
        Get or create a token bucket for a client and endpoint.
        
        Args:
            client_id: The client identifier
            endpoint: The endpoint pattern
            
        Returns:
            TokenBucket: The token bucket for this client and endpoint
        """
        # Check if bucket already exists
        if endpoint in self.buckets[client_id]:
            return self.buckets[client_id][endpoint]
        
        # Get rate limit for this endpoint
        limit, period = self.endpoint_limits.get(endpoint, (self.default_limit, self.default_period))
        
        # Create a new bucket
        bucket = TokenBucket(tokens=limit, refill_rate=limit/period)
        self.buckets[client_id][endpoint] = bucket
        
        return bucket
    
    async def is_rate_limited(self, request: Request) -> Tuple[bool, Optional[float]]:
        """
        Check if a request is rate limited.
        
        Args:
            request: The FastAPI request object
            
        Returns:
            Tuple[bool, Optional[float]]: (is_limited, retry_after)
                - is_limited: True if the request is rate limited
                - retry_after: Seconds to wait before retrying, or None if not limited
        """
        # Get client ID and endpoint pattern
        client_id = self.get_client_id(request)
        endpoint = self.get_endpoint_pattern(request.url.path)
        
        # Get token bucket
        bucket = self.get_bucket(client_id, endpoint)
        
        # Try to consume a token
        if await bucket.consume(1):
            # Request is allowed
            return False, None
        else:
            # Request is rate limited
            wait_time = await bucket.get_wait_time(1)
            return True, wait_time

class RateLimitMiddleware:
    """
    Middleware for enforcing rate limits on API requests.
    
    This middleware checks each request against the rate limiter,
    and returns a 429 Too Many Requests response if the client
    has exceeded their rate limit.
    """
    
    def __init__(self, app: ASGIApp):
        """Initialize the middleware with the FastAPI application."""
        self.app = app
        self.limiter = RateLimiter()
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """Process the request and enforce rate limits."""
        if scope["type"] != "http":
            # Pass through non-HTTP requests (like WebSockets)
            await self.app(scope, receive, send)
            return
            
        # Create a request object
        request = Request(scope)
        
        # Check if request is rate limited
        is_limited, retry_after = await self.limiter.is_rate_limited(request)
        
        if is_limited:
            # Log rate limit exceeded
            client_id = self.limiter.get_client_id(request)
            logger.warning(f"Rate limit exceeded for {client_id} on {request.url.path}")
            
            # Create rate limit response
            headers = {}
            if retry_after is not None:
                headers["Retry-After"] = str(int(retry_after))
            
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate Limit Exceeded",
                    "detail": "You have exceeded the rate limit for this endpoint",
                    "retry_after": retry_after
                },
                headers=headers
            )
            
            # Send the response
            await response(scope, receive, send)
        else:
            # Process the request normally
            await self.app(scope, receive, send)

# Function to create a rate limit decorator for specific endpoints
def create_rate_limit_decorator(
    limit: int = 100,
    period: int = 60,
    key_func: Optional[Callable[[Request], str]] = None
):
    """
    Create a decorator for rate limiting specific endpoints.
    
    Args:
        limit: Maximum number of requests allowed in the period
        period: Time period in seconds
        key_func: Function to extract a key from the request for rate limiting
            (defaults to client IP address)
            
    Returns:
        Callable: A decorator function for FastAPI endpoints
    """
    limiter = RateLimiter()
    
    # Default key function uses client IP
    if key_func is None:
        key_func = lambda request: request.client.host if request.client else "unknown"
    
    async def rate_limit_decorator(request: Request):
        """
        Check if the request exceeds the rate limit.
        
        Args:
            request: The FastAPI request object
            
        Raises:
            HTTPException: If the rate limit is exceeded
        """
        from fastapi import HTTPException
        
        # Get client key
        key = key_func(request)
        
        # Create a unique endpoint identifier
        endpoint = f"custom:{request.url.path}"
        
        # Create a custom bucket for this endpoint
        if endpoint not in limiter.endpoint_limits:
            limiter.endpoint_limits[endpoint] = (limit, period)
        
        # Check rate limit
        is_limited, retry_after = await limiter.is_rate_limited(request)
        
        if is_limited:
            # Log rate limit exceeded
            logger.warning(f"Rate limit exceeded for {key} on {request.url.path}")
            
            # Raise exception
            headers = {}
            if retry_after is not None:
                headers["Retry-After"] = str(int(retry_after))
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers=headers
            )
    
    return rate_limit_decorator
