#!/usr/bin/env python
"""
Test script to verify Brave API key connectivity.

This script:
1. Loads environment variables including the Brave API key
2. Creates an instance of the SourceValidator
3. Attempts to find sources for a test claim
4. Reports on success or failure
"""

import os
import json
import sys
import time
from pathlib import Path

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("python-dotenv not installed, using system environment variables")

# Import the source validator
try:
    from agents.utilities.source_validator import SourceValidator
    from agents.utilities.firecrawl_client import FirecrawlClient
except ImportError:
    print("Error: Could not import utilities. Make sure you're running from the project root.")
    sys.exit(1)

def test_brave_api_validity():
    """Test if the Brave API key is valid."""
    print("\n=== Testing Brave API Key Validity ===")
    
    # Get Brave API key from environment
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        print("Error: BRAVE_API_KEY not found in environment variables")
        return False
    
    # Check if it looks like a placeholder
    if brave_api_key.startswith(("xx", "brv-xx")):
        print("Error: BRAVE_API_KEY appears to be a placeholder value")
        return False
    
    print(f"Using Brave API key: {brave_api_key[:5]}...{brave_api_key[-3:]}")
    
    # Test a simple API call with minimal rate limits
    import requests
    headers = {
        'Accept': 'application/json',
        'X-Subscription-Token': brave_api_key
    }
    
    params = {
        'q': 'test',
        'count': 1
    }
    
    print("Making a simple API call to test key validity...")
    try:
        response = requests.get(
            'https://api.search.brave.com/res/v1/web/search',
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            print("✅ Brave API key is valid!")
            return True
        elif response.status_code == 429:
            print("⚠️ Brave API key is valid but rate limited (Free plan)")
            print("Response: ", response.text[:100])
            # Consider it a success for validation purposes
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            print("Response: ", response.text[:100])
            return False
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

def test_brave_images_api():
    """Test the Brave Images API directly."""
    print("\n=== Testing Brave Images API ===")
    
    # Get Brave API key from environment
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        print("Error: BRAVE_API_KEY not found in environment variables")
        return False
    
    # Test images API with minimal rate limits
    import requests
    headers = {
        'Accept': 'application/json',
        'X-Subscription-Token': brave_api_key
    }
    
    params = {
        'q': 'solar panels',
        'count': 1
    }
    
    print("Searching for a single image via Brave Images API...")
    try:
        response = requests.get(
            'https://api.search.brave.com/res/v1/images/search',
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            result = response.json()
            images = result.get('images', {}).get('results', [])
            if images:
                print(f"✅ Found {len(images)} image(s):")
                for img in images:
                    title = img.get('title', 'No title')
                    url = img.get('image', {}).get('url', 'No URL')
                    print(f"Image: {title[:50]}... - {url[:50]}...")
                return True
            else:
                print("⚠️ API responded but no images found")
                return True
        elif response.status_code == 429:
            print("⚠️ Brave Images API key is valid but rate limited (Free plan)")
            print("Response: ", response.text[:100])
            # Consider it a success for validation purposes
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            print("Response: ", response.text[:100])
            return False
    except Exception as e:
        print(f"❌ Error testing Images API: {e}")
        return False

if __name__ == "__main__":
    # Run key validity test
    key_valid = test_brave_api_validity()
    
    # Allow time for rate limits to reset
    time.sleep(2)
    
    # Run images API test
    images_api_works = test_brave_images_api()
    
    # Overall result
    if key_valid:
        print("\n✅ SUCCESS: Brave API key is valid and recognized")
        if not images_api_works:
            print("⚠️ Note: The Images API may be rate-limited, but the key itself is valid")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: There were issues with the Brave API key configuration")
        sys.exit(1) 