"""
Tests for the source validation component.
"""

import os
import pytest
import json
from unittest.mock import patch, MagicMock
from agents.utilities.source_validator import SourceValidator, SourceValidationError


@pytest.fixture
def mock_blacklist_path(tmp_path):
    """Fixture to create a temporary blacklist file."""
    blacklist_file = tmp_path / "test_blacklist.json"
    blacklist_file.write_text(json.dumps([
        "fake-news.com",
        "untrusted-source.net",
        "conspiracy-site.org"
    ]))
    return str(blacklist_file)


@pytest.fixture
def source_validator(mock_blacklist_path):
    """Fixture to create a source validator with mocked blacklist."""
    return SourceValidator(blacklist_path=mock_blacklist_path)


def test_load_blacklist(source_validator):
    """Test loading blacklist from file."""
    assert len(source_validator.blacklist) == 3
    assert "fake-news.com" in source_validator.blacklist
    assert "untrusted-source.net" in source_validator.blacklist
    assert "conspiracy-site.org" in source_validator.blacklist


def test_is_blacklisted(source_validator):
    """Test checking if a domain is blacklisted."""
    # Blacklisted domains
    assert source_validator.is_blacklisted("https://fake-news.com/article")
    assert source_validator.is_blacklisted("http://www.untrusted-source.net/page")
    assert source_validator.is_blacklisted("https://conspiracy-site.org/theory")
    
    # Non-blacklisted domains
    assert not source_validator.is_blacklisted("https://example.com")
    assert not source_validator.is_blacklisted("https://legitimate-news.org")


def test_add_to_blacklist(source_validator, mock_blacklist_path):
    """Test adding a domain to the blacklist."""
    # Add a new domain
    source_validator.add_to_blacklist("new-fake-site.com")
    
    # Check if it was added
    assert "new-fake-site.com" in source_validator.blacklist
    assert source_validator.is_blacklisted("https://new-fake-site.com/article")
    
    # Check if it was saved to the file
    with open(mock_blacklist_path, 'r') as f:
        saved_blacklist = json.load(f)
    
    assert "new-fake-site.com" in saved_blacklist


def test_remove_from_blacklist(source_validator, mock_blacklist_path):
    """Test removing a domain from the blacklist."""
    # Remove an existing domain
    result = source_validator.remove_from_blacklist("fake-news.com")
    
    # Check if it was removed
    assert result is True
    assert "fake-news.com" not in source_validator.blacklist
    assert not source_validator.is_blacklisted("https://fake-news.com/article")
    
    # Check if it was saved to the file
    with open(mock_blacklist_path, 'r') as f:
        saved_blacklist = json.load(f)
    
    assert "fake-news.com" not in saved_blacklist
    
    # Try to remove a non-existent domain
    result = source_validator.remove_from_blacklist("nonexistent-domain.com")
    assert result is False


def test_get_domain_tier(source_validator):
    """Test getting the credibility tier for a domain."""
    # Academic domains
    assert source_validator.get_domain_tier("https://harvard.edu") == "academic"
    assert source_validator.get_domain_tier("https://nih.gov") == "academic"
    
    # High credibility domains
    assert source_validator.get_domain_tier("https://nature.com/article") == "high"
    
    # News domains
    assert source_validator.get_domain_tier("https://nytimes.com/news") == "news"
    
    # Medium credibility domains
    assert source_validator.get_domain_tier("https://medium.com/post") == "medium"
    
    # Low credibility domains
    assert source_validator.get_domain_tier("https://wordpress.com/blog") == "low"
    
    # Blacklisted domains
    assert source_validator.get_domain_tier("https://fake-news.com") == "untrusted"
    
    # Unknown domain defaults to low
    assert source_validator.get_domain_tier("https://some-random-site.com") == "low"


def test_get_credibility_score(source_validator):
    """Test getting a credibility score for a domain."""
    # Academic domains (score: 10)
    assert source_validator.get_credibility_score("https://stanford.edu") == 10
    
    # High credibility domains (score: 9)
    assert source_validator.get_credibility_score("https://science.org") == 9
    
    # News domains (score: 8)
    assert source_validator.get_credibility_score("https://bbc.co.uk") == 8
    
    # Medium credibility domains (score: 6)
    assert source_validator.get_credibility_score("https://substack.com") == 6
    
    # Low credibility domains (score: 4)
    assert source_validator.get_credibility_score("https://blogspot.com") == 4
    
    # Blacklisted domains (score: 0)
    assert source_validator.get_credibility_score("https://fake-news.com") == 0


def test_validate_source(source_validator):
    """Test validating and enriching a source."""
    # Valid source
    source = {
        "url": "https://nature.com/article",
        "title": "Research Article",
        "description": "Scientific study",
        "date": "2023-01-01T00:00:00Z"
    }
    
    enriched = source_validator.validate_source(source)
    
    assert "validation" in enriched
    assert enriched["validation"]["domain"] == "nature.com"
    assert enriched["validation"]["credibility_tier"] == "high"
    assert enriched["validation"]["credibility_score"] == 9
    assert enriched["validation"]["blacklisted"] is False
    assert enriched["validation"]["publication_date"] == "2023-01-01T00:00:00Z"
    
    # Blacklisted source
    blacklisted_source = {
        "url": "https://fake-news.com/article",
        "title": "Fake News",
        "description": "Unreliable information"
    }
    
    with pytest.raises(SourceValidationError):
        source_validator.validate_source(blacklisted_source)


@patch('agents.utilities.source_validator.requests.get')
def test_find_supporting_contradicting_sources(mock_get, source_validator):
    """Test finding supporting and contradicting sources."""
    # Mock the Brave API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "Supporting Article 1",
                    "url": "https://example.org/support1",
                    "description": "Evidence supporting the claim"
                },
                {
                    "title": "Supporting Article 2",
                    "url": "https://example.org/support2",
                    "description": "More evidence supporting the claim"
                },
                {
                    "title": "Supporting Article 3",
                    "url": "https://example.org/support3",
                    "description": "Additional evidence supporting the claim"
                }
            ]
        }
    }
    mock_get.return_value = mock_response
    
    # Set API key for testing
    source_validator.brave_api_key = "test_api_key"
    
    # Test finding sources
    supporting, contradicting = source_validator.find_supporting_contradicting_sources(
        "Climate change is real", count=3
    )
    
    # Check if we got the expected number of sources
    assert len(supporting) == 3
    assert len(contradicting) == 3
    
    # Check if sources were validated
    assert "validation" in supporting[0]
    assert "validation" in contradicting[0]


def test_calculate_consensus_score(source_validator):
    """Test calculating a consensus score from sources."""
    # Create some test sources with validation data
    supporting_sources = [
        {"validation": {"credibility_score": 10}},
        {"validation": {"credibility_score": 8}},
        {"validation": {"credibility_score": 9}}
    ]
    
    contradicting_sources = [
        {"validation": {"credibility_score": 6}},
        {"validation": {"credibility_score": 4}},
        {"validation": {"credibility_score": 5}}
    ]
    
    # Calculate consensus score
    score = source_validator.calculate_consensus_score(supporting_sources, contradicting_sources)
    
    # Score should be higher than 5 (neutral) since supporting sources have higher credibility
    assert score > 5
    
    # Test with no sources
    neutral_score = source_validator.calculate_consensus_score([], [])
    assert neutral_score == 5  # Should be neutral 