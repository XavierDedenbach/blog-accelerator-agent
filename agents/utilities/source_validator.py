"""
Source validation and blacklist system for the Blog Accelerator Agent.

This utility:
1. Validates sources against credibility criteria
2. Maintains a blacklist of unreliable domains
3. Provides credibility scoring for research sources
"""

import os
import re
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import requests
from urllib.parse import urlparse
from collections import deque
import asyncio
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SourceValidationError(Exception):
    """Exception raised for errors in source validation."""
    pass


class RateLimiter:
    """
    Simple rate limiter for API calls.
    
    Limits calls to a specified rate per second.
    """
    
    def __init__(self, max_calls_per_second: int = 1, buffer_time: float = 0.1):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls_per_second: Maximum calls allowed per second
            buffer_time: Additional buffer time in seconds for more reliability
        """
        self.max_calls_per_second = max_calls_per_second
        self.buffer_time = buffer_time
        self.call_timestamps = deque()
        # Adaptive rate limiting variables
        self.rate_limit_errors = 0
        self.last_rate_limit_error = 0
        self.adaptive_mode = False
        self.original_rate = max_calls_per_second
    
    async def wait_if_needed(self):
        """
        Wait if necessary to stay within rate limits.
        """
        now = time.time()
        
        # Remove timestamps older than 1 second
        while self.call_timestamps and now - self.call_timestamps[0] > 1:
            self.call_timestamps.popleft()
        
        # Check if we need to adapt the rate limit
        if self.rate_limit_errors >= 3 and now - self.last_rate_limit_error < 10:
            if not self.adaptive_mode:
                # Enter adaptive mode - reduce rate
                new_rate = max(1, self.max_calls_per_second // 2)
                logger.warning(f"Too many rate limit errors. Adapting rate limit from {self.max_calls_per_second} to {new_rate} req/s")
                self.adaptive_mode = True
                self.max_calls_per_second = new_rate
        
        # If no recent rate limit errors, gradually recover
        elif self.adaptive_mode and now - self.last_rate_limit_error > 60:
            # Try to recover by increasing the rate limit
            new_rate = min(self.original_rate, self.max_calls_per_second * 2)
            if new_rate > self.max_calls_per_second:
                logger.info(f"Recovering: increasing rate limit from {self.max_calls_per_second} to {new_rate} req/s")
                self.max_calls_per_second = new_rate
                # Reset if we've recovered fully
                if self.max_calls_per_second >= self.original_rate:
                    self.adaptive_mode = False
                    self.rate_limit_errors = 0
                    logger.info(f"Returned to original rate limit: {self.original_rate} req/s")
        
        # Check if we need to wait (over the limit)
        current_rate = len(self.call_timestamps)
        if current_rate >= self.max_calls_per_second:
            # Calculate wait time
            earliest_timestamp = self.call_timestamps[0] if self.call_timestamps else now - 1
            wait_time = 1.0 - (now - earliest_timestamp) + self.buffer_time
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        else:
            # Even if under the limit, add a small delay for more reliable spacing
            await asyncio.sleep(self.buffer_time)
        
        # Add current timestamp (after waiting)
        self.call_timestamps.append(time.time())
    
    def record_rate_limit_error(self):
        """Record a rate limit error for adaptive rate limiting."""
        self.rate_limit_errors += 1
        self.last_rate_limit_error = time.time()
        logger.warning(f"Rate limit error recorded ({self.rate_limit_errors} in total)")


class SourceValidator:
    """
    Source validator for checking credibility of research sources.
    
    Features:
    - Domain blacklisting
    - Credibility scoring
    - Academic source validation
    - Recency checks
    """
    
    def __init__(
        self,
        blacklist_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        max_brave_calls_per_second: int = 2  # More conservative rate limit based on testing
    ):
        """
        Initialize the source validator.
        
        Args:
            blacklist_path: Path to JSON file containing blacklisted domains
            cache_dir: Directory to cache validation results
            brave_api_key: API key for Brave Search (for enhanced validation)
            max_brave_calls_per_second: Maximum Brave API calls per second (default: 2 - conservative rate)
        """
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        
        # Set default paths if not provided
        self.blacklist_path = blacklist_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../../data/blacklist.json"
        )
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../../data/validation_cache"
        )
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(max_calls_per_second=max_brave_calls_per_second)
        
        # Initialize search cache
        self.search_cache = {}
        
        # Load blacklist
        self.blacklist = self._load_blacklist()
        
        # Define credibility tiers
        self.credibility_tiers = {
            "academic": ["edu", "gov", "org"],
            "high": ["nature.com", "science.org", "nejm.org", "thelancet.com"],
            "news": ["nytimes.com", "washingtonpost.com", "bbc.co.uk", "reuters.com"],
            "medium": ["medium.com", "substack.com"],
            "low": ["wordpress.com", "blogspot.com"],
            "untrusted": []  # Populated from blacklist
        }
        
        # Add blacklisted domains to untrusted tier
        self.credibility_tiers["untrusted"].extend(self.blacklist)
    
    def _load_blacklist(self) -> List[str]:
        """
        Load blacklisted domains from JSON file.
        
        Returns:
            List of blacklisted domains
        """
        try:
            if os.path.exists(self.blacklist_path):
                with open(self.blacklist_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default blacklist if not exists
                default_blacklist = [
                    "example.com",
                    "untrusted-source.com",
                    "fake-news.org",
                    "conspiracy-theories.net"
                ]
                with open(self.blacklist_path, 'w') as f:
                    json.dump(default_blacklist, f, indent=2)
                return default_blacklist
        except Exception as e:
            logger.error(f"Error loading blacklist: {e}")
            return []
    
    def save_blacklist(self) -> None:
        """Save the current blacklist to disk."""
        try:
            with open(self.blacklist_path, 'w') as f:
                json.dump(self.blacklist, f, indent=2)
            logger.info(f"Blacklist saved to {self.blacklist_path}")
        except Exception as e:
            logger.error(f"Error saving blacklist: {e}")
            raise SourceValidationError(f"Failed to save blacklist: {e}")
    
    def add_to_blacklist(self, domain: str) -> None:
        """
        Add a domain to the blacklist.
        
        Args:
            domain: Domain to blacklist
        """
        # Extract domain if full URL is provided
        parsed = urlparse(domain)
        domain_only = parsed.netloc or parsed.path
        
        # Remove www. prefix if present
        domain_only = re.sub(r'^www\.', '', domain_only)
        
        if domain_only not in self.blacklist:
            self.blacklist.append(domain_only)
            self.credibility_tiers["untrusted"].append(domain_only)
            self.save_blacklist()
            logger.info(f"Added {domain_only} to blacklist")
    
    def remove_from_blacklist(self, domain: str) -> bool:
        """
        Remove a domain from the blacklist.
        
        Args:
            domain: Domain to remove
            
        Returns:
            True if domain was removed, False if not in blacklist
        """
        # Handle www. prefix and full URLs
        parsed = urlparse(domain)
        domain_only = parsed.netloc or parsed.path
        domain_only = re.sub(r'^www\.', '', domain_only)
        
        if domain_only in self.blacklist:
            self.blacklist.remove(domain_only)
            if domain_only in self.credibility_tiers["untrusted"]:
                self.credibility_tiers["untrusted"].remove(domain_only)
            self.save_blacklist()
            logger.info(f"Removed {domain_only} from blacklist")
            return True
        return False
    
    def is_blacklisted(self, url: str) -> bool:
        """
        Check if a URL's domain is blacklisted.
        
        Args:
            url: URL to check
            
        Returns:
            True if domain is blacklisted, False otherwise
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            domain = re.sub(r'^www\.', '', domain)
            
            # Check if domain or any parent domain is blacklisted
            domain_parts = domain.split('.')
            for i in range(len(domain_parts) - 1):
                check_domain = '.'.join(domain_parts[i:])
                if check_domain in self.blacklist:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking blacklist for {url}: {e}")
            return False
    
    def get_domain_tier(self, url: str) -> str:
        """
        Determine the credibility tier of a domain based on TLD and specific domains.

        Args:
            url: The URL to check

        Returns:
            Credibility tier (e.g., 'high', 'medium', 'low', 'news', 'academic')
        """
        try:
            domain = urlparse(url).netloc.lower()
            
            # Check if domain is empty after parsing (e.g., for invalid URLs)
            if not domain:
                return "unknown"

            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]

            # Define domain lists (can be expanded)
            high_credibility_domains = [
                "nature.com", "science.org", "thelancet.com", "nejm.org", 
                "cell.com", "ieee.org", "acm.org", "arxiv.org",
                "pubmed.ncbi.nlm.nih.gov", "fda.gov", "cdc.gov", "who.int",
                "nih.gov"
            ]
            
            news_domains = [
                "bbc.co.uk", "bbc.com", "reuters.com", "apnews.com", 
                "nytimes.com", "wsj.com", "washingtonpost.com", "theguardian.com",
                "npr.org", "pbs.org"
            ]
            
            medium_credibility_domains = [
                "wikipedia.org", "webmd.com", "mayoclinic.org", "techcrunch.com",
                "wired.com", "arstechnica.com", "stackoverflow.com", "github.com",
                "medium.com", "substack.com" # Added substack
            ]
            
            low_credibility_domains = [
                "reddit.com", "4chan.org", "quora.com", "facebook.com",
                "twitter.com", "instagram.com", "tiktok.com", "tumblr.com",
                "livejournal.com", "blogspot.com" # Added blogspot.com
            ]
            
            academic_tlds = [".edu", ".ac.uk", ".ac.jp", ".ac.cn", ".edu.au"]

            # Check specific domain lists first
            if domain in high_credibility_domains:
                return "high"
            elif domain in news_domains:
                return "news"
            elif any(domain.endswith(tld) for tld in academic_tlds):
                return "academic"
            elif domain in medium_credibility_domains:
                return "medium"
            elif domain in low_credibility_domains:
                return "low"
                
            # Default tier
            return "default"
            
        except Exception as e:
            logger.error(f"Error parsing URL {url} for domain tier: {e}")
            return "unknown"
    
    def get_credibility_score(self, url: str) -> int:
        """
        Get a numeric credibility score for a URL based on the domain tier.
        
        Args:
            url: URL to score
            
        Returns:
            Score from 0-10 where 10 is highest credibility
        """
        # Check blacklist first
        if self.is_blacklisted(url):
            return 0
            
        tier = self.get_domain_tier(url)
        
        # Map tier to score
        tier_scores = {
            "academic": 10,
            "high": 9,
            "news": 8,
            "medium": 6,
            "low": 4,
            "default": 5,
            "untrusted": 0, # Should already be caught by is_blacklisted check
            "unknown": 3
        }
        
        return tier_scores.get(tier, 5) # Default to 5 (default score) if tier is not in map
    
    def validate_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a source and enrich with credibility information.
        
        Args:
            source: Source dict with url, title, etc.
            
        Returns:
            Enriched source with validation data
        
        Raises:
            SourceValidationError: If the source is blacklisted.
        """
        url = source.get('url')
        if not url:
            return source

        # Raise error if blacklisted
        if self.is_blacklisted(url):
            raise SourceValidationError(f"Source URL is blacklisted: {url}")

        # Get credibility score and tier
        score = self.get_credibility_score(url)
        tier = self.get_domain_tier(url)
        
        # Parse domain
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if domain.startswith("www."):
            domain = domain[4:]
            
        # Get publication date if available
        publication_date = source.get('date')
        
        # Add validation data
        source['validation'] = {
            'domain': domain,
            'credibility_score': score,
            'credibility_tier': tier,
            'blacklisted': False, # Already checked above
            'publication_date': publication_date, # Add publication date
            'validated_at': datetime.now().isoformat()
        }
        
        return source
    
    async def search_web(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using the Brave Search API.

        Args:
            query: Search query
            count: Number of results to fetch
            
        Returns:
            List of validated search results
            
        Raises:
            SourceValidationError: If API key is missing or API call fails.
        """
        if not self.brave_api_key:
            raise SourceValidationError("Brave Search API key is not configured.")

        cache_key = f"{query}_{count}"
        if cache_key in self.search_cache:
            logger.info(f"Using cached search results for query: {query}")
            return self.search_cache[cache_key]

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.brave_api_key
        }
        params = {
            "q": query,
            "count": count,
            "safesearch": "moderate" # Options: off, moderate, strict
        }
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Ensure rate limit is respected
                await self.rate_limiter.wait_if_needed()
                
                # Use async client correctly
                async with httpx.AsyncClient(timeout=10) as client: # Use context manager
                    response = await client.get( # Await the get call on the client instance
                        "https://api.search.brave.com/res/v1/web/search",
                        headers=headers,
                        params=params
                    )

                # Handle rate limiting (429)
                if response.status_code == 429:
                    logger.error(f"Brave API rate limit hit (429) for query: {query}")
                    retry_count += 1
                    # Record for adaptive rate limiting
                    self.rate_limiter.record_rate_limit_error()
                    if retry_count < max_retries:
                        # Less steep backoff for Premium plan
                        wait_time = 0.5 + (retry_count * 0.5)  # Start with 1s, then 1.5s, 2s, 2.5s
                        logger.warning(f"Brave API rate limit hit, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Brave API rate limit exceeded after {max_retries} retries")
                        # Fall back to mock results when rate limit persists
                        mock_results = [
                            {
                                'title': f'Rate-limited fallback for "{query}" #{i}',
                                'url': f'https://example.com/ratelimited{i}',
                                'description': f'This is a fallback result #{i} due to API rate limits for query: {query}',
                                'source': 'Rate-Limited Fallback',
                                'date': datetime.now().isoformat()
                            } for i in range(1, count + 1)
                        ]
                        # Cache the fallback results
                        self.search_cache[cache_key] = mock_results
                        return mock_results
                
                elif response.status_code != 200:
                    logger.error(f"Error searching Brave API: {response.status_code} - {response.text}")
                    retry_count += 1
                    if retry_count < max_retries and response.status_code >= 500:  # Only retry server errors
                        wait_time = 0.5 + (retry_count * 0.5)  # Less steep backoff
                        logger.warning(f"Server error, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    raise SourceValidationError(f"Brave Search API error: {response.status_code} - {response.text}") # Include response text
                
                # Process results
                results = []
                try:
                    data = await response.json() # Await the JSON parsing
                    web_results = data.get('web', {}).get('results', [])
                except ValueError as e:
                    logger.error(f"Error parsing JSON response: {e} - Response text: {response.text}") # Log response text
                    raise SourceValidationError(f"Failed to parse API response: {e}")
                
                for result in web_results:
                    url = result.get('url')
                    
                    # Skip blacklisted sources
                    if url and self.is_blacklisted(url):
                        logger.info(f"Skipping blacklisted source: {url}")
                        continue
                    
                    source = {
                        'title': result.get('title'),
                        'url': url,
                        'description': result.get('description'),
                        'source': result.get('profile', {}).get('name', 'Unknown Source'), # Use profile name if available
                        'date': result.get('page_age') # Try to get page_age as date
                    }
                    
                    # Add validation data only if URL exists
                    if url:
                        try:
                            source = self.validate_source(source)
                            results.append(source)
                        except SourceValidationError as sve:
                            logger.warning(f"Skipping source due to validation error: {sve}")
                            continue # Skip this source
                    
                    # Stop when we have enough
                    if len(results) >= count:
                        break
                
                # Cache the results
                self.search_cache[cache_key] = results
                logger.info(f"Successfully found {len(results)} search results for query: {query}")
                return results
                
            except httpx.RequestError as exc: # Catch httpx specific request errors
                logger.error(f"HTTP Request Error searching web: {exc.__class__.__name__} - {exc}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 0.5 + (retry_count * 0.5)
                    logger.warning(f"HTTP error, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Web search failed after {max_retries} retries due to HTTP errors: {exc}")
                    # Fallback results
                    mock_results = self._create_fallback_results(query, count, str(exc))
                    self.search_cache[cache_key] = mock_results
                    return mock_results
                    
            except Exception as e:
                logger.error(f"Unexpected error searching web: {e}", exc_info=True) # Log traceback
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 0.5 + (retry_count * 0.5)
                    logger.warning(f"Unexpected error, retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Web search failed after {max_retries} retries due to unexpected error: {e}")
                    # Fallback results
                    mock_results = self._create_fallback_results(query, count, str(e))
                    self.search_cache[cache_key] = mock_results
                    return mock_results
                    
    def _create_fallback_results(self, query: str, count: int, error_message: str) -> List[Dict[str, Any]]:
        """Helper function to create fallback results."""
        return [
            {
                'title': f'Error fallback for "{query}" #{i}',
                'url': f'https://example.com/error{i}',
                'description': f'This is a fallback result #{i} due to errors for query: {query}. Error: {error_message}',
                'source': 'Error Fallback',
                'date': datetime.now().isoformat()
            } for i in range(1, count + 1)
        ]
    
    async def find_supporting_contradicting_sources(
        self, 
        claim: str, 
        count: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Find sources that support or contradict a claim.
        
        Args:
            claim: Claim to search for
            count: Number of sources to find of each type
            
        Returns:
            Tuple of (supporting_sources, contradicting_sources)
        """
        # Create search queries
        supporting_query = f"evidence supporting {claim}"
        contradicting_query = f"evidence against {claim}"
        
        # Search for supporting and contradicting sources
        supporting_results = await self.search_web(supporting_query, count)
        contradicting_results = await self.search_web(contradicting_query, count)
        
        return supporting_results, contradicting_results
    
    def calculate_consensus_score(
        self,
        supporting_sources: List[Dict[str, Any]],
        contradicting_sources: List[Dict[str, Any]]
    ) -> int:
        """
        Calculate a consensus score based on supporting and contradicting sources.
        
        Args:
            supporting_sources: List of supporting sources
            contradicting_sources: List of contradicting sources
            
        Returns:
            Consensus score from 1-10:
            - 10: Strong consensus supporting the claim
            - 1: Strong consensus against the claim
            - 5: Evenly divided evidence
        """
        if not supporting_sources and not contradicting_sources:
            return 5  # Neutral when no sources
        
        # Calculate weighted scores based on credibility
        supporting_score = sum(
            source.get('validation', {}).get('credibility_score', 5)
            for source in supporting_sources
        )
        
        contradicting_score = sum(
            source.get('validation', {}).get('credibility_score', 5)
            for source in contradicting_sources
        )
        
        total_score = supporting_score + contradicting_score
        if total_score == 0:
            return 5  # Avoid division by zero
        
        # Calculate percentage supporting (0 to 1)
        supporting_percentage = supporting_score / total_score
        
        # Map to 1-10 scale
        consensus_score = int(1 + supporting_percentage * 9)
        
        return consensus_score 