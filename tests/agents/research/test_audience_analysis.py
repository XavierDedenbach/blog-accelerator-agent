import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio
import httpx

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
    mock_response_str = create_mock_json_response("audience_segments", expected_segments)

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str
        segments = await audience_analyzer.identify_segments(test_topic)

    assert segments == expected_segments
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['topic'] == test_topic
    assert called_kwargs['min_segments'] == 3
    assert "STEM Students" in called_kwargs['existing_segments'] # Check if default segments are passed

@pytest.mark.asyncio
async def test_identify_segments_override_success(audience_analyzer, mock_llm):
    """Test successful segment identification using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)

    expected_segments = [{"name": "Regulators", "description": "Policymakers overseeing AI."}]
    mock_response_str_override = create_mock_json_response("audience_segments", expected_segments)

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str_override
        segments = await audience_analyzer.identify_segments(test_topic, llm_override=override_llm)

    assert segments == expected_segments
    mock_arun.assert_called_once()
    initial_llm.arun.assert_not_called()

    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['topic'] == test_topic
    assert called_kwargs['min_segments'] == 3
    assert "STEM Students" in called_kwargs['existing_segments']

@pytest.mark.asyncio
async def test_identify_segments_llm_error(audience_analyzer, mock_source_validator):
    """Test segment identification when the LLM (via LLMChain) fails."""
    simulated_error_message = "LLMChain.arun failed for identify_segments"
    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        
        with pytest.raises(AudienceAnalysisError, match=f"Failed to identify audience segments: {simulated_error_message}"):
            await audience_analyzer.identify_segments(test_topic)
        
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['topic'] == test_topic
        assert called_kwargs['min_segments'] == 3


# --- Tests for analyze_segment_needs ---

@pytest.mark.asyncio
async def test_analyze_segment_needs_success(audience_analyzer):
    """Test successful analysis of segment needs."""
    mock_response_str = json.dumps(mock_needs)
    
    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str
        needs = await audience_analyzer.analyze_segment_needs(mock_segment, test_topic)

    assert needs == mock_needs
    mock_arun.assert_called_once_with(
        segment_name=mock_segment['name'],
        segment_description=mock_segment['description'],
        topic=test_topic
    )

@pytest.mark.asyncio
async def test_analyze_segment_needs_override_success(audience_analyzer, mock_llm):
    """Test successful needs analysis using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    # No need to mock override_llm.arun, as we patch LLMChain.arun

    expected_needs_override = {"key_questions": ["What are the legal risks?"]}
    mock_response_str_override = json.dumps(expected_needs_override)

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str_override
        needs = await audience_analyzer.analyze_segment_needs(mock_segment, test_topic, llm_override=override_llm)

    assert needs == expected_needs_override
    mock_arun.assert_called_once_with(
        segment_name=mock_segment['name'],
        segment_description=mock_segment['description'],
        topic=test_topic
    )
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_segment_needs_llm_error(audience_analyzer, mock_source_validator):
    """Test needs analysis when the LLM (via LLMChain) fails."""
    # Standard audience_analyzer is fine, its LLM is not directly used.

    simulated_error_message = "LLMChain.arun failed for analyze_segment_needs"
    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)

        with pytest.raises(AudienceAnalysisError, match=f"Failed to analyze segment needs: {simulated_error_message}"):
            await audience_analyzer.analyze_segment_needs(mock_segment, test_topic)
        
        mock_arun.assert_called_once_with(
            segment_name=mock_segment['name'],
            segment_description=mock_segment['description'],
            topic=test_topic
        )


# --- Tests for evaluate_segment_knowledge ---

@pytest.mark.asyncio
async def test_evaluate_segment_knowledge_success(audience_analyzer):
    """Test successful evaluation of segment knowledge."""
    mock_response_str = json.dumps(mock_knowledge)
    
    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str
        knowledge = await audience_analyzer.evaluate_segment_knowledge(mock_segment, test_topic)

    assert knowledge == mock_knowledge
    mock_arun.assert_called_once_with(
        segment_name=mock_segment['name'],
        segment_description=mock_segment['description'],
        topic=test_topic
    )

@pytest.mark.asyncio
async def test_evaluate_segment_knowledge_override_success(audience_analyzer, mock_llm):
    """Test successful knowledge evaluation using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    # No need to mock override_llm.arun

    expected_knowledge_override = {"potential_misconceptions": ["AI is always objective"]}
    mock_response_str_override = json.dumps(expected_knowledge_override)

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str_override
        knowledge = await audience_analyzer.evaluate_segment_knowledge(mock_segment, test_topic, llm_override=override_llm)

    assert knowledge == expected_knowledge_override
    mock_arun.assert_called_once_with(
        segment_name=mock_segment['name'],
        segment_description=mock_segment['description'],
        topic=test_topic
    )
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_evaluate_segment_knowledge_llm_error(audience_analyzer, mock_source_validator):
    """Test knowledge evaluation when the LLM (via LLMChain) fails."""
    # Standard audience_analyzer is fine.

    simulated_error_message = "LLMChain.arun failed for evaluate_segment_knowledge"
    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)

        with pytest.raises(AudienceAnalysisError, match=f"Failed to evaluate segment knowledge: {simulated_error_message}"):
            await audience_analyzer.evaluate_segment_knowledge(mock_segment, test_topic)
        
        mock_arun.assert_called_once_with(
            segment_name=mock_segment['name'],
            segment_description=mock_segment['description'],
            topic=test_topic
        )


# --- Tests for recommend_content_strategies ---

@pytest.mark.asyncio
async def test_recommend_content_strategies_success(audience_analyzer):
    """Test successful recommendation of content strategies."""
    expected_strategies = [{"title": "Case Study Analysis", "description": "Analyze real-world cases."}]
    mock_response_str = json.dumps(expected_strategies)
    
    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str
        strategies = await audience_analyzer.recommend_content_strategies(mock_segment, test_topic, mock_needs, mock_knowledge)

    assert strategies == expected_strategies
    mock_arun.assert_called_once()
    call_args, call_kwargs = mock_arun.call_args
    assert call_kwargs['segment_name'] == mock_segment['name']
    assert call_kwargs['topic'] == test_topic
    assert json.loads(call_kwargs['needs_analysis']) == mock_needs
    assert json.loads(call_kwargs['knowledge_evaluation']) == mock_knowledge

@pytest.mark.asyncio
async def test_recommend_content_strategies_override_success(audience_analyzer, mock_llm):
    """Test successful strategy recommendation using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    # No need to mock override_llm.arun

    expected_strategies_override = [{"title": "Expert Interview Series"}]
    mock_response_str_override = json.dumps(expected_strategies_override)

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_response_str_override
        strategies = await audience_analyzer.recommend_content_strategies(
            mock_segment, test_topic, mock_needs, mock_knowledge, llm_override=override_llm
        )

    assert strategies == expected_strategies_override
    mock_arun.assert_called_once()
    call_args, call_kwargs = mock_arun.call_args
    assert call_kwargs['segment_name'] == mock_segment['name']
    # Add other assertions for call_kwargs as in the success test if needed
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_recommend_content_strategies_llm_error(audience_analyzer, mock_source_validator):
    """Test strategy recommendation when the LLM (via LLMChain) fails."""
    # Standard audience_analyzer is fine.
    simulated_error_message = "LLMChain.arun failed for recommend_content_strategies"
    
    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)

        # This call should raise the AudienceAnalysisError due to the exception in LLMChain.arun
        with pytest.raises(AudienceAnalysisError, match=f"Failed to recommend content strategies: {simulated_error_message}"):
            await audience_analyzer.recommend_content_strategies(mock_segment, test_topic, mock_needs, mock_knowledge)
        
        mock_arun.assert_called_once()
        # Assert call_kwargs if necessary

@pytest.mark.asyncio
async def test_recommend_content_strategies_rate_limit_fallback(audience_analyzer, mock_source_validator):
    """Test strategy recommendation fallback when a rate limit error occurs from LLMChain.arun."""
    # Standard audience_analyzer is fine.
    rate_limit_error_message = "429 - Rate limit exceeded"

    # ADDED: Mock response and body for OpenAIRateLimitError
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 429
    mock_http_response.request = MagicMock(spec=httpx.Request)
    mock_http_response.headers = MagicMock(spec=httpx.Headers) # ADDED spec for headers
    mock_http_response.headers.get.return_value = "mock-request-id" # Ensure .get() call on headers works

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        # Simulate the LLMChain.arun call raising a rate limit error
        mock_arun.side_effect = OpenAIRateLimitError(
            message=rate_limit_error_message, 
            response=mock_http_response, # ADDED
            body={}
        ) 
        
        # The method should catch OpenAIRateLimitError and return fallback strategies
        strategies = await audience_analyzer.recommend_content_strategies(mock_segment, test_topic, mock_needs, mock_knowledge)

    assert isinstance(strategies, list)
    assert len(strategies) > 0 # Check that fallback strategies were returned
    assert strategies[0]["title"] == "Comprehensive Guide" # Check content of fallback
    assert mock_arun.call_count == 3 # CORRECTED: Called 3 times due to retry logic


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
async def test_analyze_audience_success(audience_analyzer, mock_source_validator):
    """Test the full analyze_audience workflow using the initial LLM (via patched LLMChain.arun)."""
    # Mock responses for each stage (identify_segments, analyze_segment_needs, evaluate_segment_knowledge, recommend_content_strategies)
    segments_resp_data = [mock_segment]
    needs_resp_data = mock_needs
    knowledge_resp_data = mock_knowledge
    strategies_resp_data = [{"title": "Strategy 1"}]

    # These will be the return values of the patched LLMChain.arun, in order.
    mock_chain_arun_side_effects = [
        create_mock_json_response("audience_segments", segments_resp_data), # For identify_segments
        json.dumps(needs_resp_data),                                   # For analyze_segment_needs
        json.dumps(knowledge_resp_data),                               # For evaluate_segment_knowledge
        json.dumps(strategies_resp_data)                               # For recommend_content_strategies
    ]

    # Mock source validator (this is not LLMChain based, so direct mock is fine)
    mock_source = [{"url": "http://audience.com/data", "title": "Audience Data"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source, [])

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_arun_side_effects
        result = await audience_analyzer.analyze_audience(test_topic)

    assert result["topic"] == test_topic
    assert len(result["audience_segments"]) == 1
    segment_result = result["audience_segments"][0]
    assert segment_result["name"] == mock_segment["name"]
    assert segment_result["sources"] == mock_source
    assert segment_result["needs_analysis"] == needs_resp_data
    assert segment_result["knowledge_evaluation"] == knowledge_resp_data
    assert segment_result["content_strategies"] == strategies_resp_data
    assert result["stats"]["segments_count"] == 1
    assert result["stats"]["sources_count"] == 1

    # Verify LLMChain.arun calls (identify, needs, knowledge, strategies)
    assert mock_arun.call_count == 4 
    # Example: Check args of the first call (identify_segments)
    first_call_kwargs = mock_arun.call_args_list[0].kwargs
    assert first_call_kwargs['topic'] == test_topic
    # Example: Check args of the second call (analyze_segment_needs for the first segment)
    second_call_kwargs = mock_arun.call_args_list[1].kwargs
    assert second_call_kwargs['segment_name'] == segments_resp_data[0]['name']

    # Verify source validator called once (for the one segment)
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_audience_override_success(audience_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_audience workflow using llm_override for all LLMChain calls."""
    initial_llm = mock_llm # From the fixture, to assert it's not used
    override_llm = MagicMock(spec=BaseChatModel) # The LLM to be used by chains
    # No need to mock override_llm.arun directly

    # Mock responses for each stage if override_llm was used by the chain
    segments_resp_data_override = [{"name": "Override Segment", "search_terms": ["override search"]}]
    needs_resp_data_override = {"information_needs": ["Override need"]}
    knowledge_resp_data_override = {"likely_knowledge_gaps": ["Override gap"]}
    strategies_resp_data_override = [{"title": "Override Strategy"}]
    
    mock_chain_arun_side_effects_override = [
        create_mock_json_response("audience_segments", segments_resp_data_override),
        json.dumps(needs_resp_data_override),
        json.dumps(knowledge_resp_data_override),
        json.dumps(strategies_resp_data_override)
    ]

    mock_source_override = [{"url": "http://override.com/source", "title": "Override Source"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source_override, [])

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_arun_side_effects_override
        # Call analyze_audience with llm_override. This llm_override should be used by all internal LLMChain instances.
        result = await audience_analyzer.analyze_audience(test_topic, llm_override=override_llm)

    assert len(result["audience_segments"]) == 1
    segment_result = result["audience_segments"][0]
    assert segment_result["name"] == "Override Segment"
    assert segment_result["sources"] == mock_source_override
    assert segment_result["needs_analysis"]["information_needs"] == ["Override need"]
    assert segment_result["knowledge_evaluation"]["likely_knowledge_gaps"] == ["Override gap"]
    assert segment_result["content_strategies"][0]["title"] == "Override Strategy"

    # Verify LLMChain.arun was called 4 times (once for each step with the overridden LLM)
    assert mock_arun.call_count == 4
    initial_llm.arun.assert_not_called() # Crucially, the original LLM from the fixture was not used
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_audience_llm_error_in_needs_analysis(audience_analyzer, mock_source_validator):
    """Test analyze_audience when an LLMChain.arun call fails during needs analysis."""
    segments_resp_data = [mock_segment]
    simulated_error_message = "LLMChain.arun failed intentionally for needs analysis"

    # LLMChain.arun succeeds for identify_segments, then fails for analyze_segment_needs
    mock_chain_arun_side_effects_error = [
        create_mock_json_response("audience_segments", segments_resp_data), 
        Exception(simulated_error_message) 
    ]

    mock_source_validator.find_supporting_contradicting_sources.return_value = ([], [])

    with patch('agents.research.audience_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_arun_side_effects_error
        
        # The overall function should raise AudienceAnalysisError, originating from the analyze_segment_needs failure
        with pytest.raises(AudienceAnalysisError, match=f"Failed to analyze segment needs: {simulated_error_message}"):
            await audience_analyzer.analyze_audience(test_topic)

    # Verify LLMChain.arun calls: identify_segments (success, 1), analyze_segment_needs (failure, 1)
    assert mock_arun.call_count == 2 
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()


# Add more tests for source finding errors, JSON parse errors, multiple segments, etc. 