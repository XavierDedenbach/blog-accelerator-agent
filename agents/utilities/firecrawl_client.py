"""
Firecrawl MCP client for visual asset collection and processing.

This utility:
1. Handles communication with Firecrawl MCP server
2. Collects visual assets based on search queries
3. Processes and validates images
4. Categorizes visual content by type and relevance
"""

import os
import base64
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse
import hashlib
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FirecrawlError(Exception):
    """Exception raised for errors in Firecrawl operations."""
    pass


class FirecrawlClient:
    """
    Client for Firecrawl MCP API to retrieve and process visual assets.
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        brave_api_key: Optional[str] = None
    ):
        """
        Initialize the Firecrawl client.
        
        Args:
            server_url: Firecrawl MCP server URL
            api_key: API key for Firecrawl API
            cache_dir: Directory to cache downloaded assets
            brave_api_key: Brave Search API key (fallback for images)
        """
        # Set API info from args or environment variables
        self.server_url = server_url or os.environ.get("FIRECRAWL_SERVER", "http://firecrawl:4000")
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        
        # Set up cache directory
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../../data/firecrawl_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize session
        self.session = requests.Session()
        
        # If API key is provided, add to session headers
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})
        
        # Supported image formats
        self.supported_formats = ["jpg", "jpeg", "png", "gif", "webp", "svg"]
        
        # Track request rate to avoid hitting rate limits
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "GET", 
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry_count: int = 3,
        retry_wait: float = 2.0
    ) -> Dict[str, Any]:
        """
        Make a request to the Firecrawl API with retry logic.
        
        Args:
            endpoint: API endpoint (without base URL)
            method: HTTP method (GET, POST, etc.)
            params: URL parameters
            json_data: JSON data for POST/PUT requests
            retry_count: Number of retries on failure
            retry_wait: Wait time between retries (with exponential backoff)
            
        Returns:
            Response data as dictionary
            
        Raises:
            FirecrawlError: If the request fails after all retries
        """
        url = f"{self.server_url}/api/v1/{endpoint.lstrip('/')}"
        
        for attempt in range(retry_count):
            try:
                # Enforce rate limiting
                self._rate_limit()
                
                if method.upper() == "GET":
                    response = self.session.get(url, params=params)
                elif method.upper() == "POST":
                    response = self.session.post(url, params=params, json=json_data)
                else:
                    raise FirecrawlError(f"Unsupported HTTP method: {method}")
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited by Firecrawl MCP. Retrying in {retry_wait} seconds.")
                    time.sleep(retry_wait * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Firecrawl API error: {response.status_code} - {response.text}")
                    if attempt < retry_count - 1:
                        logger.info(f"Retrying in {retry_wait} seconds...")
                        time.sleep(retry_wait * (2 ** attempt))
                        continue
                    raise FirecrawlError(f"API request failed: {response.status_code} - {response.text}")
            
            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.error(f"Request error: {e}")
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {retry_wait} seconds...")
                    time.sleep(retry_wait * (2 ** attempt))
                    continue
                raise FirecrawlError(f"Request failed after {retry_count} attempts: {e}")
        
        raise FirecrawlError(f"Request failed after {retry_count} attempts")
    
    def search_images(
        self, 
        query: str, 
        count: int = 10, 
        image_type: Optional[str] = None,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for images using Firecrawl MCP.
        
        Args:
            query: Search query
            count: Number of images to return
            image_type: Filter by image type (photo, illustration, diagram)
            min_width: Minimum image width
            min_height: Minimum image height
            
        Returns:
            List of image metadata dictionaries
        """
        try:
            # Prepare request parameters
            params = {
                "query": query,
                "count": count
            }
            
            if image_type:
                params["type"] = image_type
                
            if min_width:
                params["min_width"] = min_width
                
            if min_height:
                params["min_height"] = min_height
            
            # Make the request
            response = self._make_request("images/search", params=params)
            
            if "results" not in response:
                logger.warning(f"No results found for query: {query}")
                return []
            
            return response["results"]
        
        except FirecrawlError as e:
            logger.error(f"Firecrawl search error: {e}")
            
            # If Firecrawl fails, try to fall back to Brave Search API
            if self.brave_api_key:
                logger.info(f"Falling back to Brave Search API for images")
                return self._brave_image_search_fallback(query, count)
            
            return []
    
    def _brave_image_search_fallback(self, query: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Fallback to Brave Search API for images if Firecrawl fails.
        
        Args:
            query: Search query
            count: Number of images to return
            
        Returns:
            List of image metadata formatted like Firecrawl results
        """
        try:
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': self.brave_api_key
            }
            
            params = {
                'q': query,
                'count': count,
                'search_lang': 'en'
            }
            
            url = 'https://api.search.brave.com/res/v1/images/search'
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"Brave API error: {response.status_code} - {response.text}")
                return []
            
            brave_results = response.json().get('images', {}).get('results', [])
            
            # Convert Brave format to Firecrawl format
            formatted_results = []
            for result in brave_results:
                formatted = {
                    "url": result.get("image", {}).get("url"),
                    "title": result.get("title"),
                    "source_url": result.get("source"),
                    "width": result.get("image", {}).get("width"),
                    "height": result.get("image", {}).get("height"),
                    "format": self._guess_format_from_url(result.get("image", {}).get("url")),
                    "source": "brave_fallback"
                }
                formatted_results.append(formatted)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Brave fallback search error: {e}")
            return []
    
    def _guess_format_from_url(self, url: str) -> str:
        """Guess image format from URL extension."""
        if not url:
            return "unknown"
            
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        for fmt in self.supported_formats:
            if path.endswith(f".{fmt}"):
                return fmt
        
        return "jpg"  # Default assumption
    
    def download_image(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Download an image and convert to base64.
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Dict with base64 data and metadata, or None if download fails
        """
        try:
            # Create a cache key from the URL
            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{url_hash}.json")
            
            # Check cache first
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    # Cache file corrupted, continue with download
                    pass
            
            # Enforce rate limiting
            self._rate_limit()
            
            # Download the image
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Failed to download image {image_url}: {response.status_code}")
                return None
            
            # Get content type and determine format
            content_type = response.headers.get('Content-Type', '')
            img_format = 'jpg'  # Default
            if 'image/' in content_type:
                img_format = content_type.split('/')[-1].lower()
                if img_format == 'jpeg':
                    img_format = 'jpg'
            
            # Convert to base64
            img_data = base64.b64encode(response.content).decode('utf-8')
            
            # Get dimensions (approximate from content length)
            # This is a rough estimate - actual dimensions would require
            # parsing the image with PIL or similar
            content_length = len(response.content)
            estimated_width = 800  # Default estimate
            estimated_height = 600  # Default estimate
            
            result = {
                'url': image_url,
                'base64': img_data,
                'format': img_format,
                'estimated_width': estimated_width,
                'estimated_height': estimated_height,
                'content_length': content_length,
                'downloaded_at': datetime.now().isoformat()
            }
            
            # Save to cache
            try:
                with open(cache_path, 'w') as f:
                    json.dump(result, f)
            except IOError as e:
                logger.warning(f"Failed to cache image: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading image {image_url}: {e}")
            return None
    
    def collect_visual_assets(
        self,
        topic: str,
        count: int = 50,
        categories: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect visual assets for a topic across multiple categories.
        
        Args:
            topic: Main topic to find visuals for
            count: Total number of visuals to collect
            categories: List of categories to search for
                        (defaults to ["diagram", "illustration", "photo", "infographic"])
            
        Returns:
            Dictionary mapping categories to lists of visual assets
        """
        if not categories:
            categories = ["diagram", "illustration", "photo", "infographic"]
        
        # Calculate images per category
        images_per_category = max(5, count // len(categories))
        
        result = {}
        total_collected = 0
        
        for category in categories:
            # Generate search queries for this category
            queries = [
                f"{topic} {category}",
                f"{category} of {topic}",
                f"{topic} visualization"
            ]
            
            category_results = []
            
            # Try each query until we get enough images
            for query in queries:
                if len(category_results) >= images_per_category:
                    break
                    
                # Search for images
                search_results = self.search_images(
                    query,
                    count=images_per_category - len(category_results),
                    image_type=category
                )
                
                # Download and process each image
                for img in search_results:
                    # Skip if we already have enough
                    if len(category_results) >= images_per_category:
                        break
                        
                    img_url = img.get("url")
                    if not img_url:
                        continue
                    
                    # Download image
                    downloaded = self.download_image(img_url)
                    if not downloaded:
                        continue
                    
                    # Add metadata
                    asset = {
                        **img,
                        **downloaded,
                        "category": category,
                        "topic": topic,
                        "search_query": query
                    }
                    
                    category_results.append(asset)
                    total_collected += 1
                
                # Add some delay between queries
                time.sleep(1.0 + random.random())
            
            result[category] = category_results
        
        logger.info(f"Collected {total_collected} visual assets for {topic} across {len(categories)} categories")
        return result
    
    def scrape_visuals_from_url(
        self,
        url: str,
        min_width: int = 400,
        min_height: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Scrape visual assets from a specific URL.
        
        Args:
            url: URL to scrape
            min_width: Minimum image width to include
            min_height: Minimum image height to include
            
        Returns:
            List of visual assets found on the page
        """
        try:
            # Prepare request
            params = {
                "url": url,
                "min_width": min_width,
                "min_height": min_height
            }
            
            # Make the request
            response = self._make_request("scrape/visuals", params=params)
            
            if "visuals" not in response:
                logger.warning(f"No visuals found for URL: {url}")
                return []
            
            visuals = response["visuals"]
            
            # Download and enrich each visual
            enriched_visuals = []
            for visual in visuals:
                img_url = visual.get("url")
                if not img_url:
                    continue
                
                # Download image
                downloaded = self.download_image(img_url)
                if not downloaded:
                    continue
                
                # Add metadata
                asset = {
                    **visual,
                    **downloaded,
                    "source_url": url,
                    "scraped_at": datetime.now().isoformat()
                }
                
                enriched_visuals.append(asset)
            
            return enriched_visuals
            
        except FirecrawlError as e:
            logger.error(f"Error scraping visuals from {url}: {e}")
            return [] 