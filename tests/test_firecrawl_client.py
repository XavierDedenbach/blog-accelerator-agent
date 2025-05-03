"""
Tests for the Firecrawl client component.
"""

import os
import pytest
import json
import base64
import time
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from agents.utilities.firecrawl_client import FirecrawlClient, FirecrawlError


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Fixture to create a temporary cache directory."""
    cache_dir = tmp_path / "firecrawl_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def firecrawl_client(mock_cache_dir):
    """Fixture to create a Firecrawl client with mocked cache directory."""
    return FirecrawlClient(
        server_url="http://firecrawl:4000",
        cache_dir=mock_cache_dir
    )


def test_initialization():
    """Test initialization with different parameters."""
    # Default initialization
    client = FirecrawlClient()
    assert client.server_url == "http://firecrawl:4000"
    
    # Custom server URL
    client = FirecrawlClient(server_url="http://custom-firecrawl:5000")
    assert client.server_url == "http://custom-firecrawl:5000"
    
    # With API key
    client = FirecrawlClient(api_key="test_api_key")
    assert client.api_key == "test_api_key"
    assert "X-API-Key" in client.session.headers
    assert client.session.headers["X-API-Key"] == "test_api_key"


def test_rate_limiting(firecrawl_client):
    """Test rate limiting between requests."""
    firecrawl_client.min_request_interval = 0.1  # Set a small interval for testing
    
    # First request shouldn't be delayed
    start_time = time.time()
    firecrawl_client._rate_limit()
    elapsed = time.time() - start_time
    assert elapsed < 0.05  # Should be very quick
    
    # Second request should be delayed by min_request_interval
    start_time = time.time()
    firecrawl_client._rate_limit()
    elapsed = time.time() - start_time
    assert elapsed >= 0.1  # Should wait at least min_request_interval


@patch('agents.utilities.firecrawl_client.requests.get')
def test_make_request_success(mock_get, firecrawl_client):
    """Test successful API request."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": ["test result"]}
    mock_get.return_value = mock_response
    
    # Make request
    result = firecrawl_client._make_request("test/endpoint", params={"query": "test"})
    
    # Check if the request was made correctly
    mock_get.assert_called_once_with(
        "http://firecrawl:4000/api/v1/test/endpoint",
        params={"query": "test"}
    )
    
    # Check if result was parsed correctly
    assert result == {"results": ["test result"]}


@patch('agents.utilities.firecrawl_client.requests.get')
def test_make_request_rate_limit(mock_get, firecrawl_client):
    """Test handling rate limiting in API requests."""
    # First response with rate limit
    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    rate_limit_response.text = "Rate limit exceeded"
    
    # Second response success
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"results": ["test result"]}
    
    # Set up mock to return rate limit then success
    mock_get.side_effect = [rate_limit_response, success_response]
    
    # Override retry wait time for faster testing
    firecrawl_client._make_request(
        "test/endpoint", 
        params={"query": "test"},
        retry_wait=0.1
    )
    
    # Should have made two requests
    assert mock_get.call_count == 2


@patch('agents.utilities.firecrawl_client.requests.get')
def test_make_request_error(mock_get, firecrawl_client):
    """Test handling API errors in requests."""
    # Mock error response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal server error"
    mock_get.return_value = mock_response
    
    # Make request (should raise exception after retries)
    with pytest.raises(FirecrawlError):
        firecrawl_client._make_request(
            "test/endpoint", 
            params={"query": "test"},
            retry_count=2,
            retry_wait=0.1
        )
    
    # Should have tried twice
    assert mock_get.call_count == 2


@patch('agents.utilities.firecrawl_client.FirecrawlClient._make_request')
def test_search_images(mock_make_request, firecrawl_client):
    """Test searching for images."""
    # Mock successful search
    mock_make_request.return_value = {
        "results": [
            {
                "url": "https://example.com/image1.jpg",
                "title": "Test Image 1",
                "source_url": "https://example.com/page1",
                "width": 800,
                "height": 600,
                "format": "jpg"
            },
            {
                "url": "https://example.com/image2.png",
                "title": "Test Image 2",
                "source_url": "https://example.com/page2",
                "width": 1024,
                "height": 768,
                "format": "png"
            }
        ]
    }
    
    # Search for images
    results = firecrawl_client.search_images(
        query="test query",
        count=2,
        image_type="photo",
        min_width=800,
        min_height=600
    )
    
    # Check if the request was made correctly
    mock_make_request.assert_called_once_with(
        "images/search",
        params={
            "query": "test query",
            "count": 2,
            "type": "photo",
            "min_width": 800,
            "min_height": 600
        }
    )
    
    # Check if results were parsed correctly
    assert len(results) == 2
    assert results[0]["url"] == "https://example.com/image1.jpg"
    assert results[1]["url"] == "https://example.com/image2.png"


@patch('agents.utilities.firecrawl_client.FirecrawlClient._make_request')
@patch('agents.utilities.firecrawl_client.requests.get')
def test_brave_fallback(mock_get, mock_make_request, firecrawl_client):
    """Test fallback to Brave Search API when Firecrawl fails."""
    # Mock Firecrawl failure
    mock_make_request.side_effect = FirecrawlError("API error")
    
    # Mock Brave API success
    brave_response = MagicMock()
    brave_response.status_code = 200
    brave_response.json.return_value = {
        "images": {
            "results": [
                {
                    "title": "Brave Image 1",
                    "source": "https://example.com/page1",
                    "image": {
                        "url": "https://example.com/brave1.jpg",
                        "width": 800,
                        "height": 600
                    }
                }
            ]
        }
    }
    mock_get.return_value = brave_response
    
    # Set Brave API key for fallback
    firecrawl_client.brave_api_key = "test_brave_key"
    
    # Search for images (should fall back to Brave)
    results = firecrawl_client.search_images(query="test query", count=1)
    
    # Check if Brave fallback was called
    mock_get.assert_called_once()
    assert "api.search.brave.com" in mock_get.call_args[0][0]
    
    # Check if results were parsed correctly
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/brave1.jpg"
    assert results[0]["source"] == "brave_fallback"


@patch('agents.utilities.firecrawl_client.requests.get')
def test_download_image(mock_get, firecrawl_client, mock_cache_dir):
    """Test downloading an image."""
    # Sample image content (a very small GIF)
    gif_content = base64.b64decode(
        "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
    )
    
    # Mock successful download
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = gif_content
    mock_response.headers = {"Content-Type": "image/gif"}
    mock_get.return_value = mock_response
    
    # Download image
    image_url = "https://example.com/test.gif"
    result = firecrawl_client.download_image(image_url)
    
    # Check if request was made correctly
    mock_get.assert_called_once_with(image_url, timeout=10)
    
    # Check if result has expected fields
    assert result["url"] == image_url
    assert result["format"] == "gif"
    assert "base64" in result
    assert result["base64"] == base64.b64encode(gif_content).decode('utf-8')
    
    # Check if image was cached
    url_hash = hashlib.md5(image_url.encode()).hexdigest()
    cache_path = Path(mock_cache_dir) / f"{url_hash}.json"
    assert cache_path.exists()
    
    # Test cache retrieval on subsequent request
    mock_get.reset_mock()
    cached_result = firecrawl_client.download_image(image_url)
    assert not mock_get.called  # Should not make a new request
    assert cached_result["url"] == image_url  # Should have same data


@patch('agents.utilities.firecrawl_client.FirecrawlClient.search_images')
@patch('agents.utilities.firecrawl_client.FirecrawlClient.download_image')
def test_collect_visual_assets(mock_download, mock_search, firecrawl_client):
    """Test collecting visual assets across categories."""
    # Mock search results
    mock_search.return_value = [
        {"url": "https://example.com/image1.jpg", "title": "Image 1"},
        {"url": "https://example.com/image2.png", "title": "Image 2"}
    ]
    
    # Mock download results
    mock_download.return_value = {
        "url": "https://example.com/image1.jpg",
        "base64": "test_base64",
        "format": "jpg",
        "estimated_width": 800,
        "estimated_height": 600,
        "content_length": 1000,
        "downloaded_at": "2023-01-01T00:00:00Z"
    }
    
    # Collect assets for two categories
    result = firecrawl_client.collect_visual_assets(
        topic="test topic",
        count=4,
        categories=["photo", "diagram"]
    )
    
    # Should have searched for each category
    assert mock_search.call_count == 2  # At least one query per category
    
    # Should have downloaded images
    assert mock_download.called
    
    # Check if result has expected categories
    assert "photo" in result
    assert "diagram" in result
    
    # Check if assets were collected
    assert len(result["photo"]) > 0
    assert len(result["diagram"]) > 0
    
    # Verify total collected count
    total_count = sum(len(assets) for assets in result.values())
    assert total_count <= 4  # Should not exceed requested count


@patch('agents.utilities.firecrawl_client.FirecrawlClient._make_request')
@patch('agents.utilities.firecrawl_client.FirecrawlClient.download_image')
def test_scrape_visuals_from_url(mock_download, mock_make_request, firecrawl_client):
    """Test scraping visuals from a URL."""
    # Mock scrape results
    mock_make_request.return_value = {
        "visuals": [
            {"url": "https://example.com/img1.jpg", "width": 800, "height": 600},
            {"url": "https://example.com/img2.png", "width": 1024, "height": 768}
        ]
    }
    
    # Mock download results
    mock_download.return_value = {
        "url": "https://example.com/img1.jpg",
        "base64": "test_base64",
        "format": "jpg",
        "estimated_width": 800,
        "estimated_height": 600,
        "content_length": 1000,
        "downloaded_at": "2023-01-01T00:00:00Z"
    }
    
    # Scrape visuals
    url = "https://example.com/page"
    result = firecrawl_client.scrape_visuals_from_url(url, min_width=400, min_height=300)
    
    # Check if the request was made correctly
    mock_make_request.assert_called_once_with(
        "scrape/visuals",
        params={"url": url, "min_width": 400, "min_height": 300}
    )
    
    # Check if images were downloaded
    assert mock_download.call_count == 2
    
    # Check if result has expected data
    assert len(result) == 2
    assert "source_url" in result[0]
    assert result[0]["source_url"] == url
    assert "base64" in result[0]
    assert "scraped_at" in result[0] 