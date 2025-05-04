#!/usr/bin/env python3
"""
Test script for Brave API throughput with Premium plan.
This script sends multiple concurrent requests to measure throughput.
"""

import os
import time
import asyncio
import logging
import argparse
import requests
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import uuid
from collections import deque

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter from the source_validator module
class RateLimiter:
    """
    Simple rate limiter for API calls.
    """
    
    def __init__(self, max_calls_per_second: int = 1, buffer_time: float = 0.1):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls_per_second: Maximum calls allowed per second
            buffer_time: Additional buffer time in seconds
        """
        self.max_calls_per_second = max_calls_per_second
        self.buffer_time = buffer_time
        self.call_timestamps = deque()
    
    async def wait_if_needed(self):
        """Wait if necessary to stay within rate limits."""
        now = time.time()
        
        # Remove timestamps older than 1 second
        while self.call_timestamps and now - self.call_timestamps[0] > 1:
            self.call_timestamps.popleft()
        
        # Check if we need to wait (over the limit)
        current_rate = len(self.call_timestamps)
        
        if current_rate >= self.max_calls_per_second:
            # Calculate wait time
            earliest_timestamp = self.call_timestamps[0] if self.call_timestamps else now - 1
            wait_time = 1.0 - (now - earliest_timestamp) + self.buffer_time
            if wait_time > 0:
                logging.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        else:
            # Even if under the limit, add a small delay for more reliable spacing
            await asyncio.sleep(self.buffer_time)
        
        # Add current timestamp (after waiting)
        self.call_timestamps.append(time.time())


async def search_brave(query: str, brave_api_key: str, rate_limiter: RateLimiter) -> Dict[str, Any]:
    """
    Perform a Brave search API call with rate limiting.
    
    Args:
        query: Search query
        brave_api_key: Brave API key
        rate_limiter: Rate limiter instance
        
    Returns:
        Dict with status information and results if successful
    """
    try:
        # Apply rate limiting
        await rate_limiter.wait_if_needed()
        
        # Make the API request
        headers = {
            'Accept': 'application/json',
            'X-Subscription-Token': brave_api_key
        }
        
        params = {
            'q': query,
            'count': 5,
            'search_lang': 'en'
        }
        
        start_time = time.time()
        
        response = requests.get(
            'https://api.search.brave.com/res/v1/web/search',
            headers=headers,
            params=params,
            timeout=10  # Add timeout to prevent hanging requests
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Handle response
        if response.status_code == 200:
            results = response.json().get('web', {}).get('results', [])
            return {
                'status': 'success',
                'elapsed': elapsed,
                'results_count': len(results),
                'status_code': response.status_code
            }
        elif response.status_code == 429:
            # Rate limit error
            return {
                'status': 'rate_limited',
                'elapsed': elapsed,
                'status_code': response.status_code,
                'message': response.text
            }
        else:
            # Other error
            return {
                'status': 'error',
                'elapsed': elapsed,
                'status_code': response.status_code,
                'message': response.text
            }
            
    except Exception as e:
        return {
            'status': 'exception',
            'message': str(e),
            'exception_type': type(e).__name__
        }


async def run_test(num_requests: int, calls_per_second: int, brave_api_key: str) -> List[Dict[str, Any]]:
    """
    Run a throughput test with the specified number of requests and rate limit.
    
    Args:
        num_requests: Number of requests to make
        calls_per_second: Target calls per second
        brave_api_key: Brave API key
        
    Returns:
        List of result dictionaries
    """
    # Create a rate limiter
    rate_limiter = RateLimiter(max_calls_per_second=calls_per_second)
    
    # Generate test queries
    test_queries = [
        f"test query {i} {uuid.uuid4().hex[:8]}" for i in range(num_requests)
    ]
    
    # Create tasks for all queries
    tasks = []
    for i, query in enumerate(test_queries):
        logging.info(f"Creating task for query {i+1}/{num_requests}")
        # Create a task for each query
        task = asyncio.create_task(
            search_brave(query, brave_api_key, rate_limiter)
        )
        tasks.append((i, query, task))
    
    # Wait for all tasks to complete
    results = []
    for i, query, task in tasks:
        try:
            start_time = time.time()
            result = await task
            end_time = time.time()
            
            # Add timing information
            result['query_index'] = i
            result['query'] = query
            result['start_time'] = start_time
            result['end_time'] = end_time
            
            results.append(result)
            logging.info(f"Completed query {i+1}/{num_requests}")
        except Exception as e:
            logging.error(f"Error processing query {i+1}: {e}")
            results.append({
                'status': 'exception',
                'message': str(e),
                'exception_type': type(e).__name__,
                'query_index': i,
                'query': query,
                'start_time': start_time,
                'end_time': time.time()
            })
    
    # Sort results by query index
    results.sort(key=lambda x: x.get('query_index', 0))
    
    return results


async def main():
    # Load API key from environment
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        print("Error: BRAVE_API_KEY environment variable not set")
        return
    
    print("\n=== Brave API Throughput Test ===")
    print(f"Testing with API key: {brave_api_key[:5]}...")
    
    # Test parameters
    calls_per_second = 20  # Updated to 20 calls per second for Premium plan
    num_requests = 60      # Increased to 60 requests to better test the throughput
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    start_time = time.time()
    results = await run_test(num_requests, calls_per_second, brave_api_key)
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r.get('status') == 'success']
    rate_limited = [r for r in results if r.get('status') == 'rate_limited']
    other_errors = [r for r in results if r.get('status') not in ['success', 'rate_limited']]
    
    success_rate = len(successful) / len(results) * 100
    actual_throughput = len(successful) / elapsed_time if elapsed_time > 0 else 0
    
    # Calculate detailed timing stats
    request_times = [r.get('elapsed', 0) for r in successful]
    avg_request_time = sum(request_times) / len(request_times) if request_times else 0
    min_request_time = min(request_times) if request_times else 0
    max_request_time = max(request_times) if request_times else 0
    
    # Calculate throughput over time
    if len(successful) > 1:
        first_success_time = min(r.get('start_time', float('inf')) for r in successful)
        last_success_time = max(r.get('end_time', 0) for r in successful)
        effective_time = last_success_time - first_success_time
        peak_throughput = len(successful) / effective_time if effective_time > 0 else 0
    else:
        peak_throughput = 0
    
    # Print results
    print("\n=== Results ===")
    print(f"Test completed in {elapsed_time:.2f} seconds")
    print(f"Total requests: {len(results)}")
    print(f"Successful requests: {len(successful)}")
    print(f"Rate limit errors: {len(rate_limited)}")
    print(f"Other errors: {len(other_errors)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Actual throughput: {actual_throughput:.2f} requests/second")
    
    print("\n=== Timing Statistics ===")
    print(f"Average request time: {avg_request_time:.4f} seconds")
    print(f"Minimum request time: {min_request_time:.4f} seconds")
    print(f"Maximum request time: {max_request_time:.4f} seconds")
    print(f"Peak throughput: {peak_throughput:.2f} requests/second")
    
    # Print concurrency analysis
    print("\n=== Concurrency Analysis ===")
    concurrency_bins = {}
    for r in successful:
        start_time = r.get('start_time', 0)
        end_time = r.get('end_time', 0)
        # Create bins for each second
        for t in range(int(start_time), int(end_time) + 1):
            concurrency_bins[t] = concurrency_bins.get(t, 0) + 1
    
    if concurrency_bins:
        max_concurrency = max(concurrency_bins.values())
        avg_concurrency = sum(concurrency_bins.values()) / len(concurrency_bins)
        print(f"Maximum concurrent requests: {max_concurrency}")
        print(f"Average concurrent requests: {avg_concurrency:.2f}")
    
    if len(rate_limited) > 0:
        print("\n⚠️ Rate limit errors detected. Your Brave API key may not support the requested throughput.")
        print("Consider using a lower rate limit or upgrading your plan.")
    
    # Sample of errors if any
    errors = [r for r in results if r.get("status") != 'success']
    if errors:
        print("\n=== Sample Errors ===")
        for i, error in enumerate(errors[:3]):
            print(f"Error {i+1}: {error.get('message', 'Unknown error')}")
            print(f"Status Code: {error.get('status_code', 'N/A')}")
            print(f"Query: {error.get('query', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main()) 