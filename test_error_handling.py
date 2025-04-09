import unittest
from datetime import datetime, timedelta
from fastapi import HTTPException
from app.utils.validators import (
    validate_numeric_range, validate_date_range, validate_pagination,
    validate_query_parameters
)
from app.middleware.rate_limiter import TokenBucket, RateLimiter
import asyncio

# Helper function to run async tests
def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

class TestValidators(unittest.TestCase):
    """Test the validation decorators directly"""
    
    def test_numeric_range_validator(self):
        """Test numeric range validation"""
        # Create a test decorator
        decorator = validate_numeric_range('test_param', min_value=1, max_value=100)
        
        # Create a test function
        @decorator
        async def test_func(test_param=None):
            return test_param
        
        # Test valid value
        result = run_async(test_func(test_param=50))
        self.assertEqual(result, 50)
        
        # Test below minimum
        with self.assertRaises(HTTPException) as context:
            run_async(test_func(test_param=0))
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("must be at least 1", context.exception.detail)
        
        # Test above maximum
        with self.assertRaises(HTTPException) as context:
            run_async(test_func(test_param=101))
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("must be at most 100", context.exception.detail)
    
    def test_date_range_validator(self):
        """Test date range validation"""
        # Create a test function
        @validate_date_range
        async def test_func(from_date=None, to_date=None):
            return from_date, to_date
        
        # Test valid dates
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        result = run_async(test_func(from_date=yesterday, to_date=today))
        self.assertEqual(result, (yesterday, today))
        
        # Test from_date after to_date
        with self.assertRaises(HTTPException) as context:
            run_async(test_func(from_date=today, to_date=yesterday))
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("from_date must be before or equal to to_date", context.exception.detail)
        
        # Test future date
        future = today + timedelta(days=10)
        with self.assertRaises(HTTPException) as context:
            run_async(test_func(from_date=future, to_date=future))
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("from_date cannot be in the future", context.exception.detail)
    
    def test_pagination_validator(self):
        """Test pagination validation"""
        # Create a test decorator
        decorator = validate_pagination(max_limit=200, default_limit=50)
        
        # Create a test function
        @decorator
        async def test_func(limit=None, offset=None):
            return limit, offset
        
        # Test default values
        result = run_async(test_func())
        self.assertEqual(result, (50, 0))
        
        # Test valid values
        result = run_async(test_func(limit=100, offset=10))
        self.assertEqual(result, (100, 10))
        
        # Test negative limit
        result = run_async(test_func(limit=-10))
        self.assertEqual(result, (50, 0))
        
        # Test exceeding max limit
        result = run_async(test_func(limit=300))
        self.assertEqual(result, (200, 0))
        
        # Test negative offset
        result = run_async(test_func(offset=-5))
        self.assertEqual(result, (50, 0))

class TestRateLimiter(unittest.TestCase):
    """Test the rate limiter directly"""
    
    def test_token_bucket(self):
        """Test the token bucket algorithm"""
        # Create a token bucket with 10 tokens and 1 token per second refill rate
        bucket = TokenBucket(10, 1)
        
        # Test initial state
        self.assertEqual(bucket.tokens, 10)
        self.assertEqual(bucket.capacity, 10)
        self.assertEqual(bucket.refill_rate, 1)
        
        # Test consuming tokens
        result = run_async(bucket.consume(5))
        self.assertTrue(result)
        self.assertAlmostEqual(bucket.tokens, 5, places=1)
        
        # Test consuming too many tokens
        result = run_async(bucket.consume(6))
        self.assertFalse(result)
        self.assertAlmostEqual(bucket.tokens, 5, places=1)
        
        # Test wait time
        wait_time = run_async(bucket.get_wait_time(6))
        self.assertGreater(wait_time, 0)

if __name__ == "__main__":
    unittest.main()
