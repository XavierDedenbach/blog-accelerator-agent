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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SourceValidationError(Exception):
    """Exception raised for errors in source validation."""
    pass


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
        brave_api_key: Optional[str] = None
    ):
        """
        Initialize the source validator.
        
        Args:
            blacklist_path: Path to JSON file containing blacklisted domains
            cache_dir: Directory to cache validation results
            brave_api_key: API key for Brave Search (for enhanced validation)
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
        Get the credibility tier for a domain.
        
        Args:
            url: URL to check
            
        Returns:
            Credibility tier ('academic', 'high', 'news', 'medium', 'low', 'untrusted')
        """
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = re.sub(r'^www\.', '', domain)
        
        # Check if blacklisted
        if self.is_blacklisted(url):
            return "untrusted"
        
        # Check for academic/government domains
        domain_parts = domain.split('.')
        tld = domain_parts[-1] if len(domain_parts) > 1 else ""
        
        if any(domain.endswith(f".{suffix}") for suffix in self.credibility_tiers["academic"]):
            return "academic"
        
        # Check other tiers
        for tier, domains in self.credibility_tiers.items():
            if tier == "untrusted" or tier == "academic":
                continue
            
            if any(domain.endswith(trusted_domain) for trusted_domain in domains):
                return tier
        
        # Default to low if not found in other tiers
        return "low"
    
    def get_credibility_score(self, url: str) -> int:
        """
        Get a numeric credibility score for a URL.
        
        Args:
            url: URL to score
            
        Returns:
            Score from 0-10 where 10 is highest credibility
        """
        tier = self.get_domain_tier(url)
        
        # Map tiers to scores
        tier_scores = {
            "academic": 10,
            "high": 9,
            "news": 8,
            "medium": 6,
            "low": 4,
            "untrusted": 0
        }
        
        return tier_scores.get(tier, 5)
    
    def validate_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a source and enrich with credibility information.
        
        Args:
            source: Source dict with url, title, etc.
            
        Returns:
            Enriched source with validation data
            
        Raises:
            SourceValidationError: If source is invalid or fails validation
        """
        url = source.get('url')
        if not url:
            raise SourceValidationError("Source missing URL")
        
        # Check if blacklisted
        if self.is_blacklisted(url):
            raise SourceValidationError(f"Source domain is blacklisted: {url}")
        
        # Get basic domain info
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = re.sub(r'^www\.', '', domain)
        
        # Get credibility info
        tier = self.get_domain_tier(url)
        score = self.get_credibility_score(url)
        
        # Try to determine publication date
        pub_date = source.get('date')
        
        # Enrich source with validation data
        enriched = {
            **source,
            'validation': {
                'domain': domain,
                'credibility_tier': tier,
                'credibility_score': score,
                'blacklisted': False,
                'publication_date': pub_date,
                'validated_at': datetime.now().isoformat()
            }
        }
        
        return enriched
    
    def find_supporting_contradicting_sources(
        self, 
        claim: str, 
        count: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Find sources that support or contradict a claim.
        
        Args:
            claim: The claim to find sources for
            count: Number of sources to find for each category
            
        Returns:
            Tuple of (supporting_sources, contradicting_sources)
        """
        if not self.brave_api_key:
            logger.warning("Brave API key not available. Using mock data.")
            # Return mock data
            return (
                [{'title': f'Supporting source for "{claim}"', 
                  'url': 'https://example.com/support',
                  'description': 'This is a mock supporting source.',
                  'source': 'Mock Source',
                  'date': datetime.now().isoformat()}] * count,
                [{'title': f'Contradicting source for "{claim}"', 
                  'url': 'https://example.com/contradict',
                  'description': 'This is a mock contradicting source.',
                  'source': 'Mock Source',
                  'date': datetime.now().isoformat()}] * count
            )
        
        # Query for supporting sources
        supporting_query = f"evidence supporting {claim}"
        supporting = []
        
        # Query for contradicting sources
        contradicting_query = f"evidence against {claim}"
        contradicting = []
        
        try:
            # Fetch supporting sources
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': self.brave_api_key
            }
            
            for query, results_list in [
                (supporting_query, supporting),
                (contradicting_query, contradicting)
            ]:
                params = {
                    'q': query,
                    'count': count * 2,  # Request more to filter blacklisted
                    'search_lang': 'en'
                }
                
                response = requests.get(
                    'https://api.search.brave.com/res/v1/web/search',
                    headers=headers,
                    params=params
                )
                
                if response.status_code != 200:
                    logger.error(f"Error searching Brave API: {response.status_code} - {response.text}")
                    raise SourceValidationError(f"Brave Search API error: {response.status_code}")
                
                results = response.json().get('web', {}).get('results', [])
                
                for result in results:
                    url = result.get('url')
                    
                    # Skip blacklisted sources
                    if self.is_blacklisted(url):
                        continue
                    
                    source = {
                        'title': result.get('title'),
                        'url': url,
                        'description': result.get('description'),
                        'source': result.get('extra_snippets', {}).get('source', 'Unknown Source'),
                        'date': datetime.now().isoformat()
                    }
                    
                    # Add validation data
                    source = self.validate_source(source)
                    results_list.append(source)
                    
                    # Stop when we have enough
                    if len(results_list) >= count:
                        break
            
            # If we didn't get enough sources, pad with placeholders
            while len(supporting) < count:
                supporting.append({
                    'title': f'Supporting source for "{claim}" (placeholder)',
                    'url': 'https://example.com/support',
                    'description': 'Source not found. Please try a different query.',
                    'source': 'Placeholder',
                    'date': datetime.now().isoformat(),
                    'validation': {
                        'domain': 'example.com',
                        'credibility_tier': 'low',
                        'credibility_score': 3,
                        'blacklisted': False,
                        'publication_date': datetime.now().isoformat(),
                        'validated_at': datetime.now().isoformat()
                    }
                })
            
            while len(contradicting) < count:
                contradicting.append({
                    'title': f'Contradicting source for "{claim}" (placeholder)',
                    'url': 'https://example.com/contradict',
                    'description': 'Source not found. Please try a different query.',
                    'source': 'Placeholder',
                    'date': datetime.now().isoformat(),
                    'validation': {
                        'domain': 'example.com',
                        'credibility_tier': 'low',
                        'credibility_score': 3,
                        'blacklisted': False,
                        'publication_date': datetime.now().isoformat(),
                        'validated_at': datetime.now().isoformat()
                    }
                })
            
            return supporting, contradicting
            
        except Exception as e:
            logger.error(f"Error finding sources: {e}")
            raise SourceValidationError(f"Failed to find sources: {e}")
    
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
        
        # Calculate total possible score
        total_sources = len(supporting_sources) + len(contradicting_sources)
        max_possible = total_sources * 10  # If all sources had max credibility
        
        if max_possible == 0:
            return 5  # Avoid division by zero
        
        # Calculate percentage supporting
        supporting_percentage = supporting_score / (supporting_score + contradicting_score)
        
        # Map percentage to 1-10 scale
        # 0% supporting -> 1
        # 50% supporting -> 5
        # 100% supporting -> 10
        consensus_score = round(1 + supporting_percentage * 9)
        
        return consensus_score 