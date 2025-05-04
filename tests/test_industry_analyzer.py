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
    
    # Mock find_supporting_contradicting_sources as AsyncMock to properly handle await
    async_mock = AsyncMock()
    async_mock.find_supporting_contradicting_sources = AsyncMock(return_value=(
        [{"url": "https://example.com/1", "title": "Source 1", "description": "Description 1"}],
        [{"url": "https://example.com/2", "title": "Source 2", "description": "Description 2"}]
    ))
    
    mock_validator.find_supporting_contradicting_sources = async_mock.find_supporting_contradicting_sources
    
    return mock_validator


@pytest.fixture
def industry_analyzer(mock_source_validator):
    """IndustryAnalyzer instance with mocked dependencies for integration testing."""
    with patch('agents.research.industry_analysis.ChatOpenAI') as mock_openai, \
         patch('agents.research.industry_analysis.LLMChain') as mock_chain:
        
        # Create a proper mock that simulates the behavior we need
        mock_chain_instance = MagicMock()
        mock_chain_instance.arun = AsyncMock()
        mock_chain.return_value = mock_chain_instance
        
        analyzer = IndustryAnalyzer(
            openai_api_key="test_key",
            source_validator=mock_source_validator
        )
        
        # Store mock chain for direct access in tests
        analyzer._mock_chain = mock_chain_instance
        
        return analyzer


@pytest.fixture
def mock_challenges():
    """Sample challenges for testing."""
    return [
        {
            "name": "Challenge 1",
            "description": "Description of challenge 1 for policy makers",
            "criticality": "High criticality for policy makers",
            "search_terms": ["term1", "term2"]
        },
        {
            "name": "Challenge 2",
            "description": "Description of challenge 2 for developers",
            "criticality": "Medium criticality for developers",
            "search_terms": ["term3", "term4"]
        }
    ]


# Primary test that verifies the full workflow - this is most important for real-world use
@pytest.mark.asyncio
async def test_analyze_industry_integration(industry_analyzer, mock_challenges):
    """Test the complete industry analysis process with focus on user personas."""
    # Mock identify_challenges to return user-centric challenges
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
            "risk_factors": ["Risk 1 for policy makers", "Risk 2 for developers"],
            "slowdown_factors": ["Slowdown 1 affecting decision-making", "Slowdown 2 affecting implementation"],
            "cost_factors": ["Cost 1 for government budgets", "Cost 2 for small teams"],
            "inefficiency_factors": ["Inefficiency 1 in workflow", "Inefficiency 2 in adoption"]
        }
    )
    
    # Call the method with a topic that includes user personas
    topic = "Challenges of electronic health record adoption for rural healthcare providers and policy makers"
    result = await industry_analyzer.analyze_industry(topic)
    
    # Check if component methods were called with the correct topic
    industry_analyzer.identify_challenges.assert_called_once_with(topic)
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
    
    # Check challenges have user persona references
    assert len(result["challenges"]) == len(mock_challenges)
    for challenge in result["challenges"]:
        assert "sources" in challenge
        assert "components" in challenge
        # Check for user persona mentions
        combined_text = (challenge.get("description", "") + " " + challenge.get("criticality", "")).lower()
        assert any(persona in combined_text for persona in ["policy makers", "developers", "healthcare providers", "users"])


# This test focuses specifically on the prompt enhancements for user personas
@pytest.mark.asyncio
async def test_user_persona_prompt_content(industry_analyzer):
    """Test that the identify_challenges_prompt properly focuses on user personas."""
    # Verify that the prompt template contains the key user persona elements
    prompt_template = industry_analyzer.identify_challenges_prompt.template
    
    # Check for user persona specific content
    assert "specific USER TYPES or ROLES mentioned in the topic" in prompt_template
    assert "SPECIFIC CONSTRAINTS do these user types face" in prompt_template
    assert "UNIQUELY AFFECTED by limitations" in prompt_template
    assert "challenge SPECIFICALLY FOR THE USER TYPES" in prompt_template
    assert "critical for specific user types" in prompt_template


@pytest.mark.asyncio
async def test_error_handling(industry_analyzer):
    """Test error handling in industry analyzer."""
    # Mock methods to raise exceptions
    industry_analyzer.identify_challenges = AsyncMock(side_effect=Exception("Test error"))
    
    # Test analyze_industry error handling
    with pytest.raises(IndustryAnalysisError):
        await industry_analyzer.analyze_industry("test topic") 