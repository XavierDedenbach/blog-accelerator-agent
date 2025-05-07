import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio
import os
import time
import httpx
from typing import Any

# Langchain imports needed for mocks
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import LLMResult, Generation # IMPORTED
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun # ADDED
from openai import RateLimitError as OpenAIRateLimitError, APIStatusError # For simulating errors
from langchain.chat_models import ChatOpenAI

# Import the class to be tested
from agents.research.paradigm_analysis import ParadigmAnalyzer, ParadigmAnalysisError
from agents.utilities.source_validator import SourceValidator


# --- Mock Fixtures ---

@pytest.fixture
def mock_llm():
    """Creates a mock LLM suitable for testing LLMChain usage.
    Mocks 'agenerate' to return an LLMResult.
    Tests should set 'llm.mock_response_text' (for single calls) 
    or 'llm._response_text_queue' (list of strings for sequential calls) 
    to the desired JSON string output.
    """
    llm = MagicMock(spec=BaseChatModel)
    
    llm._response_text_queue = [] 

    async def mock_agenerate_func(prompts: list, stop: list[str] | None = None, callbacks: AsyncCallbackManagerForChainRun | None = None, *, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, run_name: str | None = None, **kwargs: Any):
        if llm._response_text_queue:
            if not llm._response_text_queue:
                raise Exception("Mock LLM response queue is empty but was expected.")
            current_text = llm._response_text_queue.pop(0)
        elif hasattr(llm, 'mock_response_text'):
            current_text = llm.mock_response_text
        else:
            current_text = '{"error": "mock_response_text or _response_text_queue not set on mock_llm"}'
        
        # Ensure current_text is a string, as expected by Generation
        if not isinstance(current_text, str):
            # This case should ideally not be hit if tests correctly set response_text to a JSON string.
            # If it's hit, it implies a test setup issue. For robustness, dump to JSON string.
            current_text = json.dumps(current_text)

        generation = Generation(text=current_text)
        # LLMResult expects a list of lists of Generations.
        return LLMResult(generations=[[generation]], llm_output={}) # MODIFIED: Added llm_output={}

    llm.agenerate = AsyncMock(side_effect=mock_agenerate_func)
    
    # This arun mock is a fallback if tests directly call llm.arun. 
    # LLMChain calls llm.agenerate, so that's the primary mock needed.
    async def arun_side_effect_for_direct_calls(*args, **kwargs):
        # Simplified: assumes if arun is called, it should behave like a single agenerate call
        if hasattr(llm, 'mock_response_text'):
            return llm.mock_response_text
        if llm._response_text_queue: # Should ideally not be used for arun directly
             if not llm._response_text_queue: raise Exception("Mock arun: response queue empty.")
             return llm._response_text_queue.pop(0) # arun typically returns a string
        return '{"error": "arun mock default from direct call"}'
    llm.arun = AsyncMock(side_effect=arun_side_effect_for_direct_calls) 
    
    return llm

@pytest.fixture
def mock_llm_failing():
    """Creates a mock LLM that raises an error on 'agenerate' and 'arun'."""
    llm = MagicMock(spec=BaseChatModel)
    
    # MODIFIED: Provide mock response and body for OpenAIRateLimitError
    mock_response = httpx.Response(
        status_code=429, 
        request=MagicMock(spec=httpx.Request),
        # json={"error": {"message": "Mock rate limit exceeded"}} # Option 1: if body is parsed as JSON
        # content=b'{"error": {"message": "Mock rate limit exceeded"}}' # Option 2: if body is bytes
    )
    # The 'body' argument for APIStatusError is typically the parsed response body if available, or None.
    # For a rate limit error, the exact body structure might vary, but an empty dict or None is safe for mocking.
    error_to_raise = OpenAIRateLimitError(
        message="Mock rate limit error from test fixture", 
        response=mock_response, 
        body={"error": {"message": "Mock rate limit exceeded"}} # Or None, or an empty dict {} based on what the error expects
    )
    llm.agenerate = AsyncMock(side_effect=error_to_raise)
    llm.arun = AsyncMock(side_effect=error_to_raise)
    return llm

@pytest.fixture
def mock_source_validator():
    """Creates a mock SourceValidator."""
    validator = MagicMock(spec=SourceValidator)
    validator.find_supporting_contradicting_sources = AsyncMock(return_value=([], []))
    return validator

@pytest.fixture
def paradigm_analyzer(mock_llm, mock_source_validator):
    """Creates an instance of ParadigmAnalyzer with mock dependencies."""
    analyzer = ParadigmAnalyzer(llm=mock_llm, source_validator=mock_source_validator)
    # This was in the original test fixture. ParadigmAnalyzer.__init__ now sets its own default (3).
    # This fixture override means the check `if len(paradigms) < self.min_paradigms:` in analyze_paradigms
    # will use 1. For LLM calls within ParadigmAnalyzer that pass min_paradigms (like identify_historical_paradigms),
    # they use the value passed in arun() which is 3.
    analyzer.min_paradigms = 1 
    return analyzer


# --- Test Cases ---

# # Helper function to create mock JSON responses - NO LONGER NEEDED as prompts return direct arrays/objects
# def create_mock_json_response(key, data):
#     return json.dumps({key: data})

# --- Tests for identify_historical_paradigms ---

@pytest.mark.asyncio
async def test_identify_historical_paradigms_success(paradigm_analyzer):
    """Test successful identification of historical paradigms by patching LLMChain.arun."""
    topic = "Personal Computing"
    expected_paradigms_list = [{"name": "Mainframe Era", "time_period": "1950s-1970s"}]
    mock_chain_output_string = json.dumps(expected_paradigms_list)

    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string
        paradigms = await paradigm_analyzer.identify_historical_paradigms(topic)
        assert paradigms == expected_paradigms_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['topic'] == topic
        assert called_kwargs['min_paradigms'] == 3

@pytest.mark.asyncio
async def test_identify_historical_paradigms_override_success(paradigm_analyzer):
    """Test successful paradigm identification using llm_override, by patching LLMChain.arun."""
    topic = "Mobile Computing"
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    expected_paradigms_list = [{"name": "Feature Phone Era", "time_period": "1990s-2000s"}]
    mock_chain_output_string_override = json.dumps(expected_paradigms_list)

    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override
        paradigms = await paradigm_analyzer.identify_historical_paradigms(topic, llm_override=override_llm_dummy)
        assert paradigms == expected_paradigms_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['topic'] == topic
        assert called_kwargs['min_paradigms'] == 3

@pytest.mark.asyncio
async def test_identify_historical_paradigms_llm_error(paradigm_analyzer, mock_source_validator):
    """Test paradigm identification when the LLMChain.arun call itself fails."""
    topic = "Test Topic"
    simulated_error_message = "LLMChain.arun failed for identify_historical_paradigms"
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        with pytest.raises(ParadigmAnalysisError, match=f"Failed to identify historical paradigms: {simulated_error_message}"):
            await paradigm_analyzer.identify_historical_paradigms(topic)
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['topic'] == topic
        assert called_kwargs['min_paradigms'] == 3

# --- Tests for analyze_paradigm_transitions ---

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_success(paradigm_analyzer):
    """Test successful analysis of paradigm transitions by patching LLMChain.arun."""
    paradigms = [{"name": "Era 1"}, {"name": "Era 2"}]
    expected_transitions_list = [{"from_paradigm": "Era 1", "to_paradigm": "Era 2", "trigger_factors": "Innovation X"}]
    mock_chain_output_string = json.dumps(expected_transitions_list)
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string
        transitions = await paradigm_analyzer.analyze_paradigm_transitions(paradigms)
        assert transitions == expected_transitions_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert "Era 1" in called_kwargs['paradigms'] 
        assert "Era 2" in called_kwargs['paradigms']

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_override_success(paradigm_analyzer):
    """Test successful transition analysis using llm_override, by patching LLMChain.arun."""
    paradigms = [{"name": "Old Way"}, {"name": "New Way"}]
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    expected_transitions_list = [{"from_paradigm": "Old Way", "to_paradigm": "New Way"}]
    mock_chain_output_string_override = json.dumps(expected_transitions_list)
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override
        transitions = await paradigm_analyzer.analyze_paradigm_transitions(paradigms, llm_override=override_llm_dummy)
        assert transitions == expected_transitions_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert "Old Way" in called_kwargs['paradigms']

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_llm_error(paradigm_analyzer, mock_source_validator):
    """Test transition analysis when the LLMChain.arun call itself fails."""
    paradigms = [{"name": "Era 1"}, {"name": "Era 2"}]
    simulated_error_message = "LLMChain.arun failed for transitions"
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        with pytest.raises(ParadigmAnalysisError, match=f"Failed to analyze paradigm transitions: {simulated_error_message}"):
            await paradigm_analyzer.analyze_paradigm_transitions(paradigms)
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert "Era 1" in called_kwargs['paradigms']

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_insufficient_paradigms(paradigm_analyzer):
    """Test transition analysis with fewer than 2 paradigms. LLMChain.arun should not be called."""
    paradigms = [{"name": "Only Era"}]
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        transitions = await paradigm_analyzer.analyze_paradigm_transitions(paradigms)
        assert transitions == []
        mock_arun.assert_not_called()


# --- Tests for extract_historical_lessons ---

@pytest.mark.asyncio
async def test_extract_historical_lessons_success(paradigm_analyzer):
    """Test successful extraction of historical lessons by patching LLMChain.arun."""
    paradigms = [{"name": "Era 1"}]
    transitions = [{"from_paradigm": "Era 0", "to_paradigm": "Era 1"}]
    expected_lessons_list = [{"lesson": "Adapt quickly", "explanation": "Era 1 required adaptation."}]
    mock_chain_output_string = json.dumps(expected_lessons_list)
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string
        lessons = await paradigm_analyzer.extract_historical_lessons(paradigms, transitions)
        assert lessons == expected_lessons_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert "Era 1" in called_kwargs['paradigms']
        assert "Era 0" in called_kwargs['transitions']

@pytest.mark.asyncio
async def test_extract_historical_lessons_override_success(paradigm_analyzer):
    """Test successful lesson extraction using llm_override, by patching LLMChain.arun."""
    paradigms = [{"name": "P1"}]
    transitions = [{"from_paradigm": "P0", "to_paradigm": "P1"}]
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    expected_lessons_list = [{"lesson": "Be flexible"}]
    mock_chain_output_string_override = json.dumps(expected_lessons_list)
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override
        lessons = await paradigm_analyzer.extract_historical_lessons(paradigms, transitions, llm_override=override_llm_dummy)
        assert lessons == expected_lessons_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert "P1" in called_kwargs['paradigms']

@pytest.mark.asyncio
async def test_extract_historical_lessons_llm_error(paradigm_analyzer, mock_source_validator):
    """Test lesson extraction when the LLMChain.arun call itself fails."""
    paradigms = [{"name": "P1"}]
    transitions = []
    simulated_error_message = "LLMChain.arun failed for lessons"
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        with pytest.raises(ParadigmAnalysisError, match=f"Failed to extract historical lessons: {simulated_error_message}"):
            await paradigm_analyzer.extract_historical_lessons(paradigms, transitions)
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert "P1" in called_kwargs['paradigms']


# --- Tests for project_future_paradigms ---

@pytest.mark.asyncio
async def test_project_future_paradigms_success(paradigm_analyzer):
    """Test successful projection of future paradigms by patching LLMChain.arun."""
    topic = "AI Development"
    historical_paradigms = [{"name": "Rule-Based AI"}]
    historical_transitions_dummy = [] 
    lessons = [{"lesson": "Data is key"}]
    expected_future_list = [{"name": "AGI Era", "emergence_conditions": "Compute breakthrough"}]
    mock_chain_output_string = json.dumps(expected_future_list)
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string
        future = await paradigm_analyzer.project_future_paradigms(topic, historical_paradigms, historical_transitions_dummy, lessons)
        assert future == expected_future_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['topic'] == topic
        assert "Rule-Based AI" in called_kwargs['paradigms']
        assert "Data is key" in called_kwargs['lessons']

@pytest.mark.asyncio
async def test_project_future_paradigms_override_success(paradigm_analyzer):
    """Test successful future projection using llm_override, by patching LLMChain.arun."""
    topic = "Space Travel"
    historical_paradigms = [{"name": "Chemical Rockets"}]
    historical_transitions_dummy = [] 
    lessons = [{"lesson": "Cost is prohibitive"}]
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    expected_future_list = [{"name": "Fusion Rockets", "potential_implications": "Fast interplanetary travel"}]
    mock_chain_output_string_override = json.dumps(expected_future_list)
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override
        future = await paradigm_analyzer.project_future_paradigms(
            topic, historical_paradigms, historical_transitions_dummy, lessons, llm_override=override_llm_dummy
        )
        assert future == expected_future_list
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['topic'] == topic

@pytest.mark.asyncio
async def test_project_future_paradigms_llm_error(paradigm_analyzer, mock_source_validator):
    """Test future projection when the LLMChain.arun call itself fails."""
    topic = "Test Topic"
    paradigms = []
    historical_transitions_dummy = [] 
    lessons = []
    simulated_error_message = "LLMChain.arun failed for project_future_paradigms"
    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        with pytest.raises(ParadigmAnalysisError, match=f"Failed to project future paradigms: {simulated_error_message}"):
            await paradigm_analyzer.project_future_paradigms(topic, paradigms, historical_transitions_dummy, lessons)
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['topic'] == topic


# --- Tests for find_sources_for_paradigm ---

@pytest.mark.asyncio
async def test_find_sources_for_paradigm_success(paradigm_analyzer, mock_source_validator):
    """Test finding sources for a paradigm."""
    paradigm = {"name": "Industrial Revolution", "search_terms": ["industrial revolution history", "steam engine impact"]}
    mock_sources = [{"url": "http://history.com/ir", "title": "IR Overview"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_sources, [])

    sources = await paradigm_analyzer.find_sources_for_paradigm(paradigm, count=1)

    assert sources == mock_sources
    assert mock_source_validator.find_supporting_contradicting_sources.call_count > 0
    call_args, call_kwargs = mock_source_validator.find_supporting_contradicting_sources.call_args_list[0]
    assert "industrial revolution history Industrial Revolution history" in call_args[0] # Check query
    assert call_kwargs['count'] == 1


# --- Tests for analyze_paradigms (Full Workflow) ---

@pytest.mark.asyncio
async def test_analyze_paradigms_success(paradigm_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_paradigms workflow using the initial LLM."""
    topic = "Internet Evolution"

    # MODIFIED: Ensure at least 2 paradigms for analyze_paradigm_transitions to make an LLM call
    hist_paradigms_data = [
        {"name": "Web 1.0", "time_period": "1990s", "search_terms": ["web 1.0 history"]},
        {"name": "Web 2.0", "time_period": "2000s", "search_terms": ["web 2.0 history"]}
    ]
    transitions_data = [
        {"from_paradigm": "Web 1.0", "to_paradigm": "Web 2.0", "trigger_factors": "Social Media"}
    ]
    lessons_data = [{"lesson": "Connectivity and participation drives growth"}] 
    future_paradigms_data = [{"name": "Web 3.0", "early_signals": ["DeFi"]}]

    mock_chain_responses = [
        json.dumps(hist_paradigms_data),
        json.dumps(transitions_data),
        json.dumps(lessons_data),
        json.dumps(future_paradigms_data)
    ]
    
    mock_source_web1 = [{"url": "http://web.com/1", "title": "Web 1.0 History"}]
    mock_source_web2 = [{"url": "http://web.com/2", "title": "Web 2.0 History"}]
    # Source validator will be called for each paradigm in hist_paradigms_data
    mock_source_validator.find_supporting_contradicting_sources.side_effect = [
        (mock_source_web1, []),
        (mock_source_web2, [])
    ]

    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_responses
        result = await paradigm_analyzer.analyze_paradigms(topic)

    assert result["topic"] == topic
    assert len(result["historical_paradigms"]) == 2
    assert result["historical_paradigms"][0]["name"] == "Web 1.0"
    assert result["historical_paradigms"][0]["sources"] == mock_source_web1 
    assert result["historical_paradigms"][1]["name"] == "Web 2.0"
    assert result["historical_paradigms"][1]["sources"] == mock_source_web2
    assert len(result["transitions"]) == 1
    assert result["transitions"][0]["to_paradigm"] == "Web 2.0"
    assert len(result["lessons"]) == 1
    assert result["lessons"][0]["lesson"] == "Connectivity and participation drives growth"
    assert len(result["future_paradigms"]) == 1
    assert result["future_paradigms"][0]["name"] == "Web 3.0"
    assert result["stats"]["paradigms_count"] == 2
    assert result["stats"]["sources_count"] == (len(mock_source_web1) + len(mock_source_web2))

    assert mock_arun.call_count == 4
    # Check arguments for the first call (identify_historical_paradigms)
    first_call_kwargs = mock_arun.call_args_list[0].kwargs
    assert first_call_kwargs['topic'] == topic
    assert first_call_kwargs['min_paradigms'] == 3 
    # Check arguments for the second call (analyze_paradigm_transitions)
    second_call_kwargs = mock_arun.call_args_list[1].kwargs
    assert hist_paradigms_data[0]['name'] in second_call_kwargs['paradigms'] 
    assert hist_paradigms_data[1]['name'] in second_call_kwargs['paradigms'] 
    # Check arguments for the third call (extract_historical_lessons)
    third_call_kwargs = mock_arun.call_args_list[2].kwargs
    assert hist_paradigms_data[0]['name'] in third_call_kwargs['paradigms']
    assert hist_paradigms_data[1]['name'] in third_call_kwargs['paradigms'] 
    assert transitions_data[0]['from_paradigm'] in third_call_kwargs['transitions']
    # Check arguments for the fourth call (project_future_paradigms)
    fourth_call_kwargs = mock_arun.call_args_list[3].kwargs
    assert fourth_call_kwargs['topic'] == topic
    assert hist_paradigms_data[0]['name'] in fourth_call_kwargs['paradigms']
    assert hist_paradigms_data[1]['name'] in fourth_call_kwargs['paradigms'] 
    assert lessons_data[0]['lesson'] in fourth_call_kwargs['lessons']

    mock_source_validator.find_supporting_contradicting_sources.assert_called()
    assert mock_source_validator.find_supporting_contradicting_sources.call_count == len(hist_paradigms_data)

@pytest.mark.asyncio
async def test_analyze_paradigms_override_success(paradigm_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_paradigms workflow using llm_override."""
    topic = "Renewable Energy Sources"
    initial_llm_in_fixture = mock_llm 
    override_llm = MagicMock(spec=BaseChatModel) 

    hist_paradigms_data = [
        {"name": "Hydro Power", "search_terms": ["history of hydro power"]},
        {"name": "Solar Power", "search_terms": ["history of solar power"]}
    ]
    transitions_data = [
        {"from_paradigm": "Hydro Power", "to_paradigm": "Solar Power", "trigger_factors": "Cost reduction"}
    ]
    lessons_data = [{"lesson": "Policy and innovation are key"}] 
    future_paradigms_data = [{"name": "Fusion Power"}]

    mock_chain_responses_override = [
        json.dumps(hist_paradigms_data),
        json.dumps(transitions_data),
        json.dumps(lessons_data),
        json.dumps(future_paradigms_data)
    ]
    
    mock_source_hydro = [{"url": "http://hydro.com/hist", "title": "Hydro History"}]
    mock_source_solar = [{"url": "http://solar.com/hist", "title": "Solar History"}]
    mock_source_validator.find_supporting_contradicting_sources.side_effect = [
        (mock_source_hydro, []),
        (mock_source_solar, [])
    ]

    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_responses_override
        result = await paradigm_analyzer.analyze_paradigms(topic, llm_override=override_llm)

    assert len(result["historical_paradigms"]) == 2
    assert result["historical_paradigms"][0]["name"] == "Hydro Power"
    assert result["historical_paradigms"][0]["sources"] == mock_source_hydro
    assert result["historical_paradigms"][1]["name"] == "Solar Power"
    assert result["historical_paradigms"][1]["sources"] == mock_source_solar
    assert len(result["transitions"]) == 1
    assert result["transitions"][0]["to_paradigm"] == "Solar Power"
    assert len(result["lessons"]) == 1
    assert result["lessons"][0]["lesson"] == "Policy and innovation are key"
    assert len(result["future_paradigms"]) == 1
    assert result["stats"]["paradigms_count"] == 2
    assert result["stats"]["sources_count"] == (len(mock_source_hydro) + len(mock_source_solar))

    assert mock_arun.call_count == 4
    first_call_kwargs = mock_arun.call_args_list[0].kwargs
    assert first_call_kwargs['topic'] == topic
    second_call_kwargs = mock_arun.call_args_list[1].kwargs
    assert hist_paradigms_data[0]['name'] in second_call_kwargs['paradigms']
    assert hist_paradigms_data[1]['name'] in second_call_kwargs['paradigms']
    third_call_kwargs = mock_arun.call_args_list[2].kwargs
    assert hist_paradigms_data[0]['name'] in third_call_kwargs['paradigms']
    assert hist_paradigms_data[1]['name'] in third_call_kwargs['paradigms']
    assert transitions_data[0]['from_paradigm'] in third_call_kwargs['transitions']
    fourth_call_kwargs = mock_arun.call_args_list[3].kwargs
    assert fourth_call_kwargs['topic'] == topic
    assert hist_paradigms_data[0]['name'] in fourth_call_kwargs['paradigms']
    assert hist_paradigms_data[1]['name'] in fourth_call_kwargs['paradigms']
    assert lessons_data[0]['lesson'] in fourth_call_kwargs['lessons']

    mock_source_validator.find_supporting_contradicting_sources.assert_called()
    assert mock_source_validator.find_supporting_contradicting_sources.call_count == len(hist_paradigms_data)
    initial_llm_in_fixture.agenerate.assert_not_called() 


@pytest.mark.asyncio
async def test_analyze_paradigms_llm_error_at_transitions(paradigm_analyzer, mock_llm, mock_source_validator):
    """Test analyze_paradigms when LLMChain.arun fails during transition analysis."""
    topic = "Transportation History"
    hist_paradigms_data = [{"name": "Horse & Buggy", "search_terms": ["horse buggy history"]}, {"name": "Automobile", "search_terms": ["automobile history"]}]
    
    # MODIFIED: First element is JSON string, second is Exception
    successful_identify_json_string = json.dumps(hist_paradigms_data)
    error_for_transitions = Exception("LLM transition analysis failed")

    mock_source_validator.find_supporting_contradicting_sources.return_value = ([], []) # Assume sources found for initial paradigms

    with patch('agents.research.paradigm_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = [
            successful_identify_json_string, 
            error_for_transitions
        ]
        
        # The overall function should raise ParadigmAnalysisError wrapping the specific failure
        expected_error_match = "Failed to complete paradigm analysis: Failed to analyze paradigm transitions: LLM transition analysis failed"
        with pytest.raises(ParadigmAnalysisError, match=expected_error_match):
            await paradigm_analyzer.analyze_paradigms(topic)

    assert mock_arun.call_count == 2 # Identify (success) + Transitions (fail)
    
    # Check call to identify_historical_paradigms
    first_call_kwargs = mock_arun.call_args_list[0].kwargs
    assert first_call_kwargs['topic'] == topic
    assert first_call_kwargs['min_paradigms'] == 3

    # Check call to analyze_paradigm_transitions
    second_call_kwargs = mock_arun.call_args_list[1].kwargs
    # Ensure paradigms from the first successful call are passed to the second (as a formatted string)
    # MODIFIED: Check for presence of names in the string, not length equality or exact structure match.
    assert hist_paradigms_data[0]['name'] in second_call_kwargs['paradigms'] 
    assert hist_paradigms_data[1]['name'] in second_call_kwargs['paradigms']

    # Source validator should be called for paradigms identified before the error
    assert mock_source_validator.find_supporting_contradicting_sources.call_count == len(hist_paradigms_data)
    # Example check for one of the source validator calls
    first_sv_call_args, first_sv_call_kwargs = mock_source_validator.find_supporting_contradicting_sources.call_args_list[0]
    assert "horse buggy history Horse & Buggy history" in first_sv_call_args[0] # Based on the first paradigm
    second_sv_call_args, second_sv_call_kwargs = mock_source_validator.find_supporting_contradicting_sources.call_args_list[1]
    assert "automobile history Automobile history" in second_sv_call_args[0] # Based on the second paradigm

# Add more tests for source finding errors, JSON parse errors, etc. 


# --- Integration Test with Real LLM (Optional) ---
# This test uses real API keys loaded from .env by the load_env_vars_and_debug fixture in conftest.py.
# It does NOT use the mock_env_variables fixture.

@pytest.mark.asyncio
async def test_analyze_paradigms_integration_real_llm(): 
    """Performs an integration test of the full analyze_paradigms workflow
    using real API calls to OpenAI and Brave Search. Requires .env file with valid keys.
    """
    # print("\n--- Starting test_analyze_paradigms_integration_real_llm ---")
    
    openai_api_key_env = os.getenv("OPENAI_API_KEY")
    brave_api_key_env = os.getenv("BRAVE_API_KEY")

    if not openai_api_key_env:
        pytest.skip("OpenAI API key not found in environment. Skipping integration test.")
    if not brave_api_key_env: # Brave key is used by SourceValidator
        pytest.skip("Brave API key not found in environment. Skipping integration test.")

    real_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key_env)
    real_source_validator = SourceValidator(brave_api_key=brave_api_key_env) 
    
    # Instantiate ParadigmAnalyzer with real components
    # Ensure ParadigmAnalyzer can be initialized this way or adjust if it expects specific config only
    analyzer = ParadigmAnalyzer(llm=real_llm, source_validator=real_source_validator)
    # If ParadigmAnalyzer has a settable min_paradigms, consider if it needs adjustment for integration testing time vs thoroughness.
    # analyzer.min_paradigms = 2 # Example: reduce for faster integration test, default is likely 3 from prompt
    
    topic = "The evolution of social media platforms" # A suitable topic for paradigm analysis
    
    try:
        # print(f"Attempting real paradigm analysis for topic: {topic}")
        start_time = time.time() # Requires import time
        result = await analyzer.analyze_paradigms(topic)
        end_time = time.time() # Requires import time
        # print(f"Real paradigm analysis took {end_time - start_time:.2f} seconds.")

        # Print a sample of the first identified historical paradigm
        if result and result.get('historical_paradigms'):
            print("\n\n--- Integration Test: First Historical Paradigm Identified (Sample) ---")
            if result['historical_paradigms']:
                print(json.dumps(result['historical_paradigms'][0], indent=2))
            else:
                print("No historical paradigms were identified by the first LLM call.")
            print("---------------------------------------------------------------------\n")
        elif result:
            print("\n\n--- Integration Test: LLM call did not produce 'historical_paradigms' list ---")
            print(f"Full result: {json.dumps(result, indent=2)}")
            print("----------------------------------------------------------------------------------\n")
        else:
            print("\n\n--- Integration Test: analyze_paradigms returned None or empty result ---")

        assert result is not None, "Result should not be None"
        assert result['topic'] == topic, "Result topic should match input topic"
        assert 'historical_paradigms' in result
        assert 'transitions' in result
        assert 'lessons' in result
        assert 'future_paradigms' in result
        assert 'stats' in result
        
        # Example of a more specific assertion - check if at least one historical paradigm was found
        assert len(result.get('historical_paradigms', [])) >= 1, "Should identify at least one historical paradigm"
            
    except OpenAIRateLimitError:
        pytest.skip("OpenAI rate limit hit during integration test. Skipping.")
    except httpx.HTTPStatusError as e: # Requires import httpx
        if e.response.status_code == 429:
            pytest.skip(f"API rate limit hit (likely Brave Search): {e}")
        elif e.response.status_code == 401:
             pytest.fail(f"API key unauthorized (likely Brave Search - 401): {e}. Ensure BRAVE_API_KEY is correct.")
        else:
            pytest.fail(f"Integration test failed with HTTPStatusError: {e}")
    except Exception as e:
        pytest.fail(f"Real LLM/Search integration test failed with an unexpected exception: {e}")
    # finally:
        # print("--- Finished test_analyze_paradigms_integration_real_llm ---\\n") 