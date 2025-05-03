"""
Tests for the Industry Analyzer component.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
from agents.research.industry_analysis import IndustryAnalyzer, IndustryAnalysisError


@pytest.fixture
def mock_source_validator():
    """Mock SourceValidator for testing."""
    mock_validator = MagicMock()
    
    # Mock find_supporting_contradicting_sources
    mock_validator.find_supporting_contradicting_sources.return_value = (
        [{"url": "https://example.com/1", "title": "Source 1", "description": "Description 1"}],
        [{"url": "https://example.com/2", "title": "Source 2", "description": "Description 2"}]
    )
    
    return mock_validator


@pytest.fixture
def industry_analyzer(mock_source_validator):
    """IndustryAnalyzer instance with mocked dependencies."""
    with patch('agents.research.industry_analysis.ChatOpenAI') as mock_openai:
        # Mock LLM instance
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        analyzer = IndustryAnalyzer(
            openai_api_key="test_key",
            source_validator=mock_source_validator
        )
        
        # Store mock LLM for assertions
        analyzer._mock_llm = mock_llm
        
        return analyzer


@pytest.fixture
def mock_challenges():
    """Sample challenges for testing."""
    return [
        {
            "name": "Challenge 1",
            "description": "Description of challenge 1",
            "criticality": "High criticality",
            "search_terms": ["term1", "term2"]
        },
        {
            "name": "Challenge 2",
            "description": "Description of challenge 2",
            "criticality": "Medium criticality",
            "search_terms": ["term3", "term4"]
        }
    ]


@pytest.mark.asyncio
async def test_identify_challenges(industry_analyzer):
    """Test identifying challenges for a topic."""
    # Mock LLM response
    industry_analyzer._mock_llm.arun = AsyncMock(
        return_value=json.dumps([
            {
                "name": "Challenge 1",
                "description": "Description of challenge 1",
                "criticality": "High criticality",
                "search_terms": ["term1", "term2"]
            },
            {
                "name": "Challenge 2",
                "description": "Description of challenge 2",
                "criticality": "Medium criticality",
                "search_terms": ["term3", "term4"]
            }
        ])
    )
    
    # Call the method
    challenges = await industry_analyzer.identify_challenges("test topic")
    
    # Check if LLM was called with correct parameters
    industry_analyzer._mock_llm.arun.assert_called_once()
    args, kwargs = industry_analyzer._mock_llm.arun.call_args
    assert "test topic" in kwargs.get("topic", "")
    assert industry_analyzer.min_challenges == kwargs.get("min_challenges", 0)
    
    # Check result
    assert len(challenges) == 2
    assert challenges[0]["name"] == "Challenge 1"
    assert challenges[1]["name"] == "Challenge 2"


@pytest.mark.asyncio
async def test_find_sources_for_challenge(industry_analyzer, mock_source_validator):
    """Test finding sources for a challenge."""
    challenge = {
        "name": "Test Challenge",
        "description": "Description",
        "search_terms": ["term1", "term2"]
    }
    
    # Call the method
    sources = await industry_analyzer.find_sources_for_challenge(challenge)
    
    # Check if source validator was called
    assert mock_source_validator.find_supporting_contradicting_sources.call_count > 0
    
    # Check result
    assert len(sources) > 0


@pytest.mark.asyncio
async def test_analyze_challenge_components(industry_analyzer):
    """Test analyzing components of a challenge."""
    # Mock LLM response
    industry_analyzer._mock_llm.arun = AsyncMock(
        return_value=json.dumps({
            "risk_factors": ["Risk 1", "Risk 2"],
            "slowdown_factors": ["Slowdown 1", "Slowdown 2"],
            "cost_factors": ["Cost 1", "Cost 2"],
            "inefficiency_factors": ["Inefficiency 1", "Inefficiency 2"]
        })
    )
    
    challenge = {
        "name": "Test Challenge",
        "description": "Description of test challenge"
    }
    
    sources = [
        {"url": "https://example.com/1", "title": "Source 1", "description": "Description 1"},
        {"url": "https://example.com/2", "title": "Source 2", "description": "Description 2"}
    ]
    
    # Call the method
    components = await industry_analyzer.analyze_challenge_components(challenge, sources)
    
    # Check if LLM was called with correct parameters
    industry_analyzer._mock_llm.arun.assert_called_once()
    args, kwargs = industry_analyzer._mock_llm.arun.call_args
    assert challenge["name"] in kwargs.get("challenge", "")
    assert challenge["description"] in kwargs.get("description", "")
    assert "Source 1" in kwargs.get("sources", "")
    assert "Source 2" in kwargs.get("sources", "")
    
    # Check result
    assert "risk_factors" in components
    assert "slowdown_factors" in components
    assert "cost_factors" in components
    assert "inefficiency_factors" in components
    assert len(components["risk_factors"]) == 2
    assert len(components["slowdown_factors"]) == 2
    assert len(components["cost_factors"]) == 2
    assert len(components["inefficiency_factors"]) == 2


@pytest.mark.asyncio
async def test_analyze_industry_integration(industry_analyzer, mock_challenges):
    """Test the complete industry analysis process."""
    # Mock identify_challenges
    industry_analyzer.identify_challenges = AsyncMock(return_value=mock_challenges)
    
    # Mock find_sources_for_challenge
    industry_analyzer.find_sources_for_challenge = AsyncMock(
        return_value=[
            {"url": "https://example.com/1", "title": "Source 1", "description": "Description 1"},
            {"url": "https://example.com/2", "title": "Source 2", "description": "Description 2"}
        ]
    )
    
    # Mock analyze_challenge_components
    industry_analyzer.analyze_challenge_components = AsyncMock(
        return_value={
            "risk_factors": ["Risk 1", "Risk 2"],
            "slowdown_factors": ["Slowdown 1", "Slowdown 2"],
            "cost_factors": ["Cost 1", "Cost 2"],
            "inefficiency_factors": ["Inefficiency 1", "Inefficiency 2"]
        }
    )
    
    # Call the method
    result = await industry_analyzer.analyze_industry("test topic")
    
    # Check if component methods were called
    industry_analyzer.identify_challenges.assert_called_once_with("test topic")
    assert industry_analyzer.find_sources_for_challenge.call_count == len(mock_challenges)
    assert industry_analyzer.analyze_challenge_components.call_count == len(mock_challenges)
    
    # Check result structure
    assert "topic" in result
    assert "challenges" in result
    assert "stats" in result
    
    # Check stats
    assert "challenges_count" in result["stats"]
    assert "sources_count" in result["stats"]
    assert "analysis_duration_seconds" in result["stats"]
    assert "timestamp" in result["stats"]
    
    # Check challenges
    assert len(result["challenges"]) == len(mock_challenges)
    for challenge in result["challenges"]:
        assert "sources" in challenge
        assert "components" in challenge


@pytest.mark.asyncio
async def test_error_handling(industry_analyzer):
    """Test error handling in industry analyzer."""
    # Mock LLM to raise an exception
    industry_analyzer._mock_llm.arun = AsyncMock(side_effect=Exception("Test error"))
    
    # Test identify_challenges error handling
    with pytest.raises(IndustryAnalysisError):
        await industry_analyzer.identify_challenges("test topic")
    
    # Test analyze_industry error handling
    industry_analyzer.identify_challenges = AsyncMock(side_effect=Exception("Test error"))
    
    with pytest.raises(IndustryAnalysisError):
        await industry_analyzer.analyze_industry("test topic") 