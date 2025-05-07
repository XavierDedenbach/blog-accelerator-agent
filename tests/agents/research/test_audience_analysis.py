import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio

# Langchain imports needed for mocks
from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError as OpenAIRateLimitError # For simulating errors

# Import the class to be tested
from agents.research.audience_analysis import AudienceAnalyzer, AudienceAnalysisError
from agents.utilities.source_validator import SourceValidator


# --- Mock Fixtures ---

@pytest.fixture
def mock_llm():
    """Creates a mock LLM that returns predefined JSON responses."""
    llm = MagicMock(spec=BaseChatModel)
    llm.arun = AsyncMock() # Use AsyncMock for async methods like arun
    return llm

@pytest.fixture
def mock_llm_failing():
    """Creates a mock LLM that raises a RateLimitError."""
    llm = MagicMock(spec=BaseChatModel)
    llm.arun = AsyncMock(side_effect=OpenAIRateLimitError("Mock rate limit error"))
    return llm

@pytest.fixture
def mock_source_validator():
    """Creates a mock SourceValidator."""
    validator = MagicMock(spec=SourceValidator)
    # Mock the source finding method used by AudienceAnalyzer
    validator.find_supporting_contradicting_sources = AsyncMock(return_value=([], []))
    return validator

@pytest.fixture
def audience_analyzer(mock_llm, mock_source_validator):
    """Creates an instance of AudienceAnalyzer with mock dependencies."""
    analyzer = AudienceAnalyzer(llm=mock_llm, source_validator=mock_source_validator)
    # Mock the default segments if necessary, or assume they are static class data
    return analyzer


# --- Test Cases ---

# Helper function to create mock JSON responses
def create_mock_json_response(key, data):
    return json.dumps({key: data})

# Sample data for tests
test_topic = "AI Ethics in Hiring"
mock_segment = {"name": "HR Professionals", "description": "HR managers using AI tools.", "search_terms": ["HR AI ethics"]}
mock_needs = {"information_needs": ["Bias detection methods"], "key_questions": ["How to ensure fairness?"]}
mock_knowledge = {"assumed_knowledge": ["Basic HR practices"], "likely_knowledge_gaps": ["Specific AI algorithms"]}

# --- Tests for identify_segments ---

@pytest.mark.asyncio
async def test_identify_segments_success(audience_analyzer, mock_llm):
    """Test successful identification of audience segments."""
    expected_segments = [mock_segment]
    mock_response = create_mock_json_response("audience_segments", expected_segments)
    mock_llm.arun.return_value = mock_response

    segments = await audience_analyzer.identify_segments(test_topic)

    assert segments == expected_segments
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert call_kwargs['topic'] == test_topic
    assert call_kwargs['min_segments'] == 3
    assert "STEM Students" in call_kwargs['existing_segments'] # Check if default segments are passed

@pytest.mark.asyncio
async def test_identify_segments_override_success(audience_analyzer, mock_llm):
    """Test successful segment identification using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_segments = [{"name": "Regulators", "description": "Policymakers overseeing AI."}]
    mock_response = create_mock_json_response("audience_segments", expected_segments)
    override_llm.arun.return_value = mock_response

    segments = await audience_analyzer.identify_segments(test_topic, llm_override=override_llm)

    assert segments == expected_segments
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_identify_segments_llm_error(audience_analyzer, mock_llm_failing):
    """Test segment identification when the LLM fails."""
    analyzer = AudienceAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(AudienceAnalysisError, match="Failed to identify audience segments"):
        await analyzer.identify_segments(test_topic)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for analyze_segment_needs ---

@pytest.mark.asyncio
async def test_analyze_segment_needs_success(audience_analyzer, mock_llm):
    """Test successful analysis of segment needs."""
    mock_response = json.dumps(mock_needs)
    mock_llm.arun.return_value = mock_response

    needs = await audience_analyzer.analyze_segment_needs(mock_segment, test_topic)

    assert needs == mock_needs
    mock_llm.arun.assert_called_once_with(
        segment_name=mock_segment['name'],
        segment_description=mock_segment['description'],
        topic=test_topic
    )

@pytest.mark.asyncio
async def test_analyze_segment_needs_override_success(audience_analyzer, mock_llm):
    """Test successful needs analysis using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_needs = {"key_questions": ["What are the legal risks?"]}
    mock_response = json.dumps(expected_needs)
    override_llm.arun.return_value = mock_response

    needs = await audience_analyzer.analyze_segment_needs(mock_segment, test_topic, llm_override=override_llm)

    assert needs == expected_needs
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_segment_needs_llm_error(audience_analyzer, mock_llm_failing):
    """Test needs analysis when the LLM fails."""
    analyzer = AudienceAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(AudienceAnalysisError, match="Failed to analyze segment needs"):
        await analyzer.analyze_segment_needs(mock_segment, test_topic)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for evaluate_segment_knowledge ---

@pytest.mark.asyncio
async def test_evaluate_segment_knowledge_success(audience_analyzer, mock_llm):
    """Test successful evaluation of segment knowledge."""
    mock_response = json.dumps(mock_knowledge)
    mock_llm.arun.return_value = mock_response

    knowledge = await audience_analyzer.evaluate_segment_knowledge(mock_segment, test_topic)

    assert knowledge == mock_knowledge
    mock_llm.arun.assert_called_once_with(
        segment_name=mock_segment['name'],
        segment_description=mock_segment['description'],
        topic=test_topic
    )

@pytest.mark.asyncio
async def test_evaluate_segment_knowledge_override_success(audience_analyzer, mock_llm):
    """Test successful knowledge evaluation using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_knowledge = {"potential_misconceptions": ["AI is always objective"]}
    mock_response = json.dumps(expected_knowledge)
    override_llm.arun.return_value = mock_response

    knowledge = await audience_analyzer.evaluate_segment_knowledge(mock_segment, test_topic, llm_override=override_llm)

    assert knowledge == expected_knowledge
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_evaluate_segment_knowledge_llm_error(audience_analyzer, mock_llm_failing):
    """Test knowledge evaluation when the LLM fails."""
    analyzer = AudienceAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(AudienceAnalysisError, match="Failed to evaluate segment knowledge"):
        await analyzer.evaluate_segment_knowledge(mock_segment, test_topic)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for recommend_content_strategies ---

@pytest.mark.asyncio
async def test_recommend_content_strategies_success(audience_analyzer, mock_llm):
    """Test successful recommendation of content strategies."""
    expected_strategies = [{"title": "Case Study Analysis", "description": "Analyze real-world cases."}]
    mock_response = json.dumps(expected_strategies)
    mock_llm.arun.return_value = mock_response

    strategies = await audience_analyzer.recommend_content_strategies(mock_segment, test_topic, mock_needs, mock_knowledge)

    assert strategies == expected_strategies
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert call_kwargs['segment_name'] == mock_segment['name']
    assert call_kwargs['topic'] == test_topic
    assert json.loads(call_kwargs['needs_analysis']) == mock_needs
    assert json.loads(call_kwargs['knowledge_evaluation']) == mock_knowledge

@pytest.mark.asyncio
async def test_recommend_content_strategies_override_success(audience_analyzer, mock_llm):
    """Test successful strategy recommendation using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_strategies = [{"title": "Expert Interview Series"}]
    mock_response = json.dumps(expected_strategies)
    override_llm.arun.return_value = mock_response

    strategies = await audience_analyzer.recommend_content_strategies(
        mock_segment, test_topic, mock_needs, mock_knowledge, llm_override=override_llm
    )

    assert strategies == expected_strategies
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_recommend_content_strategies_llm_error(audience_analyzer, mock_llm_failing):
    """Test strategy recommendation when the LLM fails."""
    analyzer = AudienceAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(AudienceAnalysisError, match="Failed to recommend content strategies"):
        await analyzer.recommend_content_strategies(mock_segment, test_topic, mock_needs, mock_knowledge)
    mock_llm_failing.arun.assert_called_once()

@pytest.mark.asyncio
async def test_recommend_content_strategies_rate_limit_fallback(audience_analyzer, mock_llm_failing):
    """Test strategy recommendation fallback when a rate limit error occurs."""
    # Ensure the mock LLM raises a rate limit error specifically
    mock_llm_failing.arun.side_effect = OpenAIRateLimitError("429 - Rate limit exceeded")
    analyzer = AudienceAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    # It should not raise an error, but return fallback strategies
    strategies = await analyzer.recommend_content_strategies(mock_segment, test_topic, mock_needs, mock_knowledge)

    assert isinstance(strategies, list)
    assert len(strategies) > 0 # Check that fallback strategies were returned
    assert strategies[0]["title"] == "Comprehensive Guide" # Check content of fallback
    mock_llm_failing.arun.assert_called_once() # LLM was still called once


# --- Tests for find_sources_for_segment ---

@pytest.mark.asyncio
async def test_find_sources_for_segment_success(audience_analyzer, mock_source_validator):
    """Test finding sources for an audience segment."""
    mock_sources = [{"url": "http://hrtech.com/ai", "title": "AI in HR Report"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_sources, [])

    sources = await audience_analyzer.find_sources_for_segment(mock_segment, count=1)

    assert sources == mock_sources
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()
    call_args, call_kwargs = mock_source_validator.find_supporting_contradicting_sources.call_args
    assert "HR AI ethics audience data HR Professionals" in call_args[0] # Check query
    assert call_kwargs['count'] == 1


# --- Tests for analyze_audience (Full Workflow) ---

@pytest.mark.asyncio
async def test_analyze_audience_success(audience_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_audience workflow using the initial LLM."""
    # Mock responses for each stage
    segments_resp = [mock_segment]
    needs_resp = mock_needs
    knowledge_resp = mock_knowledge
    strategies_resp = [{"title": "Strategy 1"}]

    # Configure side_effect for mock LLM
    mock_llm_responses = [
        create_mock_json_response("audience_segments", segments_resp),
        json.dumps(needs_resp),
        json.dumps(knowledge_resp),
        json.dumps(strategies_resp)
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    # Mock source validator
    mock_source = [{"url": "http://audience.com/data", "title": "Audience Data"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source, [])

    result = await audience_analyzer.analyze_audience(test_topic)

    assert result["topic"] == test_topic
    assert len(result["audience_segments"]) == 1
    segment_result = result["audience_segments"][0]
    assert segment_result["name"] == mock_segment["name"]
    assert segment_result["sources"] == mock_source
    assert segment_result["needs_analysis"] == needs_resp
    assert segment_result["knowledge_evaluation"] == knowledge_resp
    assert segment_result["content_strategies"] == strategies_resp
    assert result["stats"]["segments_count"] == 1
    assert result["stats"]["sources_count"] == 1

    # Verify LLM calls (identify, needs, knowledge, strategies)
    assert mock_llm.arun.call_count == 4
    # Verify source validator called once (for the one segment)
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_audience_override_success(audience_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_audience workflow using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    # Mock responses for override LLM
    segments_resp = [{"name": "Override Segment", "search_terms": ["override search"]}]
    needs_resp = {"information_needs": ["Override need"]}
    knowledge_resp = {"likely_knowledge_gaps": ["Override gap"]}
    strategies_resp = [{"title": "Override Strategy"}]
    override_responses = [
        create_mock_json_response("audience_segments", segments_resp),
        json.dumps(needs_resp),
        json.dumps(knowledge_resp),
        json.dumps(strategies_resp)
    ]
    override_llm.arun.side_effect = override_responses

    # Mock source validator
    mock_source = [{"url": "http://override.com/source", "title": "Override Source"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source, [])

    result = await audience_analyzer.analyze_audience(test_topic, llm_override=override_llm)

    assert len(result["audience_segments"]) == 1
    segment_result = result["audience_segments"][0]
    assert segment_result["name"] == "Override Segment"
    assert segment_result["sources"] == mock_source
    assert segment_result["needs_analysis"]["information_needs"] == ["Override need"]
    assert segment_result["knowledge_evaluation"]["likely_knowledge_gaps"] == ["Override gap"]
    assert segment_result["content_strategies"][0]["title"] == "Override Strategy"

    # Verify override LLM was called, initial was not
    assert override_llm.arun.call_count == 4
    initial_llm.arun.assert_not_called()
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_audience_llm_error_in_needs_analysis(audience_analyzer, mock_llm, mock_source_validator):
    """Test analyze_audience when LLM fails during needs analysis."""
    # Mock responses: identify succeeds, needs fails
    segments_resp = [mock_segment]

    mock_llm_responses = [
        create_mock_json_response("audience_segments", segments_resp), # Success for identify
        AsyncMock(side_effect=Exception("LLM needs analysis failed")) # Failure for needs
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    # Mock source validator (called for identify)
    mock_source_validator.find_supporting_contradicting_sources.return_value = ([], [])

    # The overall function should raise the error from the failed stage
    with pytest.raises(AudienceAnalysisError, match="Failed to analyze segment needs"):
        await audience_analyzer.analyze_audience(test_topic)

    # Verify LLM calls: identify (1), needs (1 failed)
    assert mock_llm.arun.call_count == 2
    # Verify source validator called once (for the segment)
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()


# Add more tests for source finding errors, JSON parse errors, multiple segments, etc. 