import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio

# Langchain imports needed for mocks
from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError as OpenAIRateLimitError # For simulating errors

# Import the class to be tested
from agents.research.paradigm_analysis import ParadigmAnalyzer, ParadigmAnalysisError
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
    # Mock the source finding method used by ParadigmAnalyzer
    validator.find_supporting_contradicting_sources = AsyncMock(return_value=([], []))
    return validator

@pytest.fixture
def paradigm_analyzer(mock_llm, mock_source_validator):
    """Creates an instance of ParadigmAnalyzer with mock dependencies."""
    # Set a default min_paradigms if needed for testing
    analyzer = ParadigmAnalyzer(llm=mock_llm, source_validator=mock_source_validator)
    analyzer.min_paradigms = 1 # Lower for easier testing, adjust if necessary
    return analyzer


# --- Test Cases ---

# Helper function to create mock JSON responses
def create_mock_json_response(key, data):
    return json.dumps({key: data})

# --- Tests for identify_historical_paradigms ---

@pytest.mark.asyncio
async def test_identify_historical_paradigms_success(paradigm_analyzer, mock_llm):
    """Test successful identification of historical paradigms."""
    topic = "Personal Computing"
    expected_paradigms = [{"name": "Mainframe Era", "time_period": "1950s-1970s"}]
    mock_response = create_mock_json_response("paradigms", expected_paradigms)
    mock_llm.arun.return_value = mock_response

    paradigms = await paradigm_analyzer.identify_historical_paradigms(topic)

    assert paradigms == expected_paradigms
    mock_llm.arun.assert_called_once_with(topic=topic, min_paradigms=3)

@pytest.mark.asyncio
async def test_identify_historical_paradigms_override_success(paradigm_analyzer, mock_llm):
    """Test successful paradigm identification using llm_override."""
    topic = "Mobile Computing"
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_paradigms = [{"name": "Feature Phone Era", "time_period": "1990s-2000s"}]
    mock_response = create_mock_json_response("paradigms", expected_paradigms)
    override_llm.arun.return_value = mock_response

    paradigms = await paradigm_analyzer.identify_historical_paradigms(topic, llm_override=override_llm)

    assert paradigms == expected_paradigms
    override_llm.arun.assert_called_once_with(topic=topic, min_paradigms=3)
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_identify_historical_paradigms_llm_error(paradigm_analyzer, mock_llm_failing):
    """Test paradigm identification when the LLM fails."""
    topic = "Test Topic"
    analyzer = ParadigmAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(ParadigmAnalysisError, match="Failed to identify historical paradigms"):
        await analyzer.identify_historical_paradigms(topic)
    mock_llm_failing.arun.assert_called_once()

# --- Tests for analyze_paradigm_transitions ---

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_success(paradigm_analyzer, mock_llm):
    """Test successful analysis of paradigm transitions."""
    paradigms = [{"name": "Era 1"}, {"name": "Era 2"}]
    expected_transitions = [{"from_paradigm": "Era 1", "to_paradigm": "Era 2", "trigger_factors": "Innovation X"}]
    mock_response = create_mock_json_response("transitions", expected_transitions)
    mock_llm.arun.return_value = mock_response

    transitions = await paradigm_analyzer.analyze_paradigm_transitions(paradigms)

    assert transitions == expected_transitions
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert "Era 1" in call_kwargs['paradigms'] and "Era 2" in call_kwargs['paradigms']

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_override_success(paradigm_analyzer, mock_llm):
    """Test successful transition analysis using llm_override."""
    paradigms = [{"name": "Old Way"}, {"name": "New Way"}]
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_transitions = [{"from_paradigm": "Old Way", "to_paradigm": "New Way"}]
    mock_response = create_mock_json_response("transitions", expected_transitions)
    override_llm.arun.return_value = mock_response

    transitions = await paradigm_analyzer.analyze_paradigm_transitions(paradigms, llm_override=override_llm)

    assert transitions == expected_transitions
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_llm_error(paradigm_analyzer, mock_llm_failing):
    """Test transition analysis when the LLM fails."""
    paradigms = [{"name": "Era 1"}, {"name": "Era 2"}]
    analyzer = ParadigmAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(ParadigmAnalysisError, match="Failed to analyze paradigm transitions"):
        await analyzer.analyze_paradigm_transitions(paradigms)
    mock_llm_failing.arun.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_paradigm_transitions_insufficient_paradigms(paradigm_analyzer):
    """Test transition analysis with fewer than 2 paradigms."""
    paradigms = [{"name": "Only Era"}]
    transitions = await paradigm_analyzer.analyze_paradigm_transitions(paradigms)
    assert transitions == []
    paradigm_analyzer.initial_llm.arun.assert_not_called() # LLM shouldn't be called


# --- Tests for extract_historical_lessons ---

@pytest.mark.asyncio
async def test_extract_historical_lessons_success(paradigm_analyzer, mock_llm):
    """Test successful extraction of historical lessons."""
    paradigms = [{"name": "Era 1"}]
    transitions = [{"from_paradigm": "Era 0", "to_paradigm": "Era 1"}]
    expected_lessons = [{"lesson": "Adapt quickly", "explanation": "Era 1 required adaptation."}]
    mock_response = create_mock_json_response("lessons", expected_lessons)
    mock_llm.arun.return_value = mock_response

    lessons = await paradigm_analyzer.extract_historical_lessons(paradigms, transitions)

    assert lessons == expected_lessons
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert "Era 1" in call_kwargs['paradigms']
    assert "Era 0" in call_kwargs['transitions']
    assert call_kwargs['min_lessons'] == 3

@pytest.mark.asyncio
async def test_extract_historical_lessons_override_success(paradigm_analyzer, mock_llm):
    """Test successful lesson extraction using llm_override."""
    paradigms = [{"name": "P1"}]
    transitions = [{"from_paradigm": "P0", "to_paradigm": "P1"}]
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_lessons = [{"lesson": "Be flexible"}]
    mock_response = create_mock_json_response("lessons", expected_lessons)
    override_llm.arun.return_value = mock_response

    lessons = await paradigm_analyzer.extract_historical_lessons(paradigms, transitions, llm_override=override_llm)

    assert lessons == expected_lessons
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_extract_historical_lessons_llm_error(paradigm_analyzer, mock_llm_failing):
    """Test lesson extraction when the LLM fails."""
    paradigms = [{"name": "P1"}]
    transitions = []
    analyzer = ParadigmAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(ParadigmAnalysisError, match="Failed to extract historical lessons"):
        await analyzer.extract_historical_lessons(paradigms, transitions)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for project_future_paradigms ---

@pytest.mark.asyncio
async def test_project_future_paradigms_success(paradigm_analyzer, mock_llm):
    """Test successful projection of future paradigms."""
    topic = "AI Development"
    historical_paradigms = [{"name": "Rule-Based AI"}]
    lessons = [{"lesson": "Data is key"}]
    expected_future = [{"name": "AGI Era", "emergence_conditions": "Compute breakthrough"}]
    mock_response = create_mock_json_response("future_paradigms", expected_future)
    mock_llm.arun.return_value = mock_response

    future = await paradigm_analyzer.project_future_paradigms(topic, historical_paradigms, lessons)

    assert future == expected_future
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert call_kwargs['topic'] == topic
    assert "Rule-Based AI" in call_kwargs['historical_paradigms']
    assert "Data is key" in call_kwargs['lessons_learned']
    assert call_kwargs['min_future_paradigms'] == 3

@pytest.mark.asyncio
async def test_project_future_paradigms_override_success(paradigm_analyzer, mock_llm):
    """Test successful future projection using llm_override."""
    topic = "Space Travel"
    historical_paradigms = [{"name": "Chemical Rockets"}]
    lessons = [{"lesson": "Cost is prohibitive"}]
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_future = [{"name": "Fusion Rockets", "potential_implications": "Fast interplanetary travel"}]
    mock_response = create_mock_json_response("future_paradigms", expected_future)
    override_llm.arun.return_value = mock_response

    future = await paradigm_analyzer.project_future_paradigms(topic, historical_paradigms, lessons, llm_override=override_llm)

    assert future == expected_future
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_project_future_paradigms_llm_error(paradigm_analyzer, mock_llm_failing):
    """Test future projection when the LLM fails."""
    topic = "Test Topic"
    paradigms = []
    lessons = []
    analyzer = ParadigmAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(ParadigmAnalysisError, match="Failed to project future paradigms"):
        await analyzer.project_future_paradigms(topic, paradigms, lessons)
    mock_llm_failing.arun.assert_called_once()


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

    # Mock responses for each stage
    hist_paradigms = [{"name": "Web 1.0", "time_period": "1990s", "search_terms": ["web 1.0 history"]}]
    transitions = [{"from_paradigm": "Dial-up", "to_paradigm": "Web 1.0"}]
    lessons = [{"lesson": "Connectivity drives growth"}]
    future_paradigms = [{"name": "Web 3.0", "early_signals": ["DeFi"]}]

    # Configure side_effect for mock LLM
    mock_llm_responses = [
        create_mock_json_response("paradigms", hist_paradigms),
        create_mock_json_response("transitions", transitions),
        create_mock_json_response("lessons", lessons),
        create_mock_json_response("future_paradigms", future_paradigms)
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    # Mock source validator
    mock_source = [{"url": "http://web.com/1", "title": "Web History"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source, [])

    result = await paradigm_analyzer.analyze_paradigms(topic)

    assert result["topic"] == topic
    assert len(result["historical_paradigms"]) == 1
    assert result["historical_paradigms"][0]["name"] == "Web 1.0"
    assert result["historical_paradigms"][0]["sources"] == mock_source
    assert len(result["transitions"]) == 1
    assert result["transitions"][0]["to_paradigm"] == "Web 1.0"
    assert len(result["lessons"]) == 1
    assert result["lessons"][0]["lesson"] == "Connectivity drives growth"
    assert len(result["future_paradigms"]) == 1
    assert result["future_paradigms"][0]["name"] == "Web 3.0"
    assert result["stats"]["paradigms_count"] == 1
    assert result["stats"]["sources_count"] == 1

    # Verify LLM calls (identify, transitions, lessons, future)
    assert mock_llm.arun.call_count == 4
    # Verify source validator called once (for the one historical paradigm)
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_paradigms_override_success(paradigm_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_paradigms workflow using llm_override."""
    topic = "Renewable Energy Sources"
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    # Mock responses for override LLM
    hist_paradigms = [{"name": "Hydro Power", "search_terms": ["history of hydro power"]}]
    transitions = [{"from_paradigm": "Fossil Fuels", "to_paradigm": "Hydro Power"}]
    lessons = [{"lesson": "Policy matters"}]
    future_paradigms = [{"name": "Fusion Power"}]

    override_responses = [
        create_mock_json_response("paradigms", hist_paradigms),
        create_mock_json_response("transitions", transitions),
        create_mock_json_response("lessons", lessons),
        create_mock_json_response("future_paradigms", future_paradigms)
    ]
    override_llm.arun.side_effect = override_responses

    # Mock source validator
    mock_source = [{"url": "http://hydro.com/hist", "title": "Hydro History"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source, [])

    result = await paradigm_analyzer.analyze_paradigms(topic, llm_override=override_llm)

    assert len(result["historical_paradigms"]) == 1
    assert result["historical_paradigms"][0]["name"] == "Hydro Power"
    assert len(result["transitions"]) == 1
    assert len(result["lessons"]) == 1
    assert len(result["future_paradigms"]) == 1

    # Verify override LLM was called, initial was not
    assert override_llm.arun.call_count == 4
    initial_llm.arun.assert_not_called()
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_paradigms_llm_error_at_transitions(paradigm_analyzer, mock_llm, mock_source_validator):
    """Test analyze_paradigms when LLM fails during transition analysis."""
    topic = "Transportation History"
    hist_paradigms = [{"name": "Horse & Buggy"}, {"name": "Automobile"}]

    # Mock responses: identify succeeds, transitions fails
    mock_llm_responses = [
        create_mock_json_response("paradigms", hist_paradigms),
        AsyncMock(side_effect=Exception("LLM transition analysis failed"))
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    # Mock source validator
    mock_source_validator.find_supporting_contradicting_sources.return_value = ([], [])

    # The overall function should raise the error from the failed stage
    with pytest.raises(ParadigmAnalysisError, match="Failed to analyze paradigm transitions"):
        await paradigm_analyzer.analyze_paradigms(topic)

    # Verify LLM calls: identify (1), transitions (1 failed)
    assert mock_llm.arun.call_count == 2
    # Verify source validator called twice (for the two paradigms)
    assert mock_source_validator.find_supporting_contradicting_sources.call_count == 2

# Add more tests for source finding errors, JSON parse errors, etc. 