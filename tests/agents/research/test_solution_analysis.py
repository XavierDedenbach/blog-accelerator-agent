import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio

# Langchain imports needed for mocks
from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError as OpenAIRateLimitError # For simulating errors

# Import the class to be tested
from agents.research.solution_analysis import SolutionAnalyzer, SolutionAnalysisError
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
    # Mock the source finding method used by SolutionAnalyzer
    validator.find_supporting_contradicting_sources = AsyncMock(return_value=([], []))
    return validator

@pytest.fixture
def solution_analyzer(mock_llm, mock_source_validator):
    """Creates an instance of SolutionAnalyzer with mock dependencies."""
    return SolutionAnalyzer(llm=mock_llm, source_validator=mock_source_validator)


# --- Test Cases ---

# Helper function to create mock JSON responses
def create_mock_json_response(key, data):
    return json.dumps({key: data})

# --- Tests for generate_pro_arguments ---

@pytest.mark.asyncio
async def test_generate_pro_arguments_success(solution_analyzer, mock_llm):
    """Test successful generation of pro arguments using the initial LLM."""
    solution_title = "Decentralized Identity"
    expected_args = [{"name": "User Control", "description": "Gives users control over their data."}]
    mock_response = create_mock_json_response("arguments", expected_args)
    mock_llm.arun.return_value = mock_response

    arguments = await solution_analyzer.generate_pro_arguments(solution_title)

    assert arguments == expected_args
    mock_llm.arun.assert_called_once_with(solution=solution_title, min_arguments=5)

@pytest.mark.asyncio
async def test_generate_pro_arguments_override_success(solution_analyzer, mock_llm):
    """Test successful pro argument generation using llm_override."""
    solution_title = "Quantum Encryption"
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_args = [{"name": "Unbreakable Security", "description": "Leverages quantum mechanics."}]
    mock_response = create_mock_json_response("arguments", expected_args)
    override_llm.arun.return_value = mock_response

    arguments = await solution_analyzer.generate_pro_arguments(solution_title, llm_override=override_llm)

    assert arguments == expected_args
    override_llm.arun.assert_called_once_with(solution=solution_title, min_arguments=5)
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_generate_pro_arguments_llm_error(solution_analyzer, mock_llm_failing):
    """Test pro argument generation when the LLM fails."""
    solution_title = "Test Solution"
    analyzer = SolutionAnalyzer(llm=mock_llm_failing, source_validator=MagicMock()) # Use failing LLM

    with pytest.raises(SolutionAnalysisError, match="Failed to generate pro arguments"):
        await analyzer.generate_pro_arguments(solution_title)
    mock_llm_failing.arun.assert_called_once()

# --- Tests for generate_counter_arguments ---

@pytest.mark.asyncio
async def test_generate_counter_arguments_success(solution_analyzer, mock_llm):
    """Test successful generation of counter arguments using the initial LLM."""
    solution_title = "AI-Powered Diagnosis"
    expected_args = [{"name": "Bias Risk", "description": "Risk of biased training data."}]
    mock_response = create_mock_json_response("arguments", expected_args)
    mock_llm.arun.return_value = mock_response

    arguments = await solution_analyzer.generate_counter_arguments(solution_title)

    assert arguments == expected_args
    mock_llm.arun.assert_called_once_with(solution=solution_title, min_arguments=5)

@pytest.mark.asyncio
async def test_generate_counter_arguments_override_success(solution_analyzer, mock_llm):
    """Test successful counter argument generation using llm_override."""
    solution_title = "Carbon Capture Tech"
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_args = [{"name": "High Cost", "description": "Requires significant investment."}]
    mock_response = create_mock_json_response("arguments", expected_args)
    override_llm.arun.return_value = mock_response

    arguments = await solution_analyzer.generate_counter_arguments(solution_title, llm_override=override_llm)

    assert arguments == expected_args
    override_llm.arun.assert_called_once_with(solution=solution_title, min_arguments=5)
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_generate_counter_arguments_llm_error(solution_analyzer, mock_llm_failing):
    """Test counter argument generation when the LLM fails."""
    solution_title = "Test Solution"
    analyzer = SolutionAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(SolutionAnalysisError, match="Failed to generate counter arguments"):
        await analyzer.generate_counter_arguments(solution_title)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for identify_metrics ---

@pytest.mark.asyncio
async def test_identify_metrics_success(solution_analyzer, mock_llm):
    """Test successful identification of metrics using the initial LLM."""
    solution_title = "Predictive Maintenance System"
    expected_metrics = [{"name": "Downtime Reduction", "importance_context": "Reduces operational costs."}]
    mock_response = create_mock_json_response("metrics", expected_metrics)
    mock_llm.arun.return_value = mock_response

    metrics = await solution_analyzer.identify_metrics(solution_title)

    assert metrics == expected_metrics
    # Check call args - adjust based on actual prompt variables used in identify_metrics_prompt
    # Assuming 'solution' and 'min_metrics' are key variables
    mock_llm.arun.assert_called_once_with(solution=solution_title, min_metrics=5)

@pytest.mark.asyncio
async def test_identify_metrics_override_success(solution_analyzer, mock_llm):
    """Test successful metric identification using llm_override."""
    solution_title = "Personalized Learning Platform"
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_metrics = [{"name": "Student Engagement Score", "importance_context": "Measures platform effectiveness."}]
    mock_response = create_mock_json_response("metrics", expected_metrics)
    override_llm.arun.return_value = mock_response

    metrics = await solution_analyzer.identify_metrics(solution_title, llm_override=override_llm)

    assert metrics == expected_metrics
    override_llm.arun.assert_called_once_with(solution=solution_title, min_metrics=5)
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_identify_metrics_llm_error(solution_analyzer, mock_llm_failing):
    """Test metric identification when the LLM fails."""
    solution_title = "Test Solution"
    analyzer = SolutionAnalyzer(llm=mock_llm_failing, source_validator=MagicMock())

    with pytest.raises(SolutionAnalysisError, match="Failed to identify metrics"):
        await analyzer.identify_metrics(solution_title)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for find_argument_sources ---

@pytest.mark.asyncio
async def test_find_argument_sources_success(solution_analyzer, mock_source_validator):
    """Test finding sources for an argument."""
    argument = {"name": "Improved Efficiency", "search_terms": ["process optimization case studies"]}
    solution_title = "Automation Tool"
    mock_sources = [{"url": "http://example.com/eff1", "title": "Efficiency Study"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_sources, [])

    sources = await solution_analyzer.find_argument_sources(argument, solution_title, is_pro_argument=True)

    assert sources == mock_sources
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()
    # Check query construction
    call_args, call_kwargs = mock_source_validator.find_supporting_contradicting_sources.call_args
    assert "process optimization case studies Improved Efficiency" in call_args[0]


# --- Tests for analyze_solution (Full Workflow) ---

@pytest.mark.asyncio
async def test_analyze_solution_success(solution_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_solution workflow using the initial LLM."""
    solution_title = "Smart Grid Technology"

    # Mock responses for each LLM call stage
    pro_args = [{"name": "Reduced Blackouts", "search_terms": ["smart grid reliability"]}]
    con_args = [{"name": "Security Vulnerabilities", "search_terms": ["smart grid cyber security risks"]}]
    metrics = [{"name": "Energy Loss Reduction (%)"}]

    # Configure side_effect for the mock LLM to handle multiple calls
    mock_llm_responses = [
        create_mock_json_response("arguments", pro_args),  # For pro arguments
        create_mock_json_response("arguments", con_args),  # For counter arguments
        create_mock_json_response("metrics", metrics)      # For metrics
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    # Mock source validator response (same for pro and con args in this test)
    mock_source = [{"url": "http://grid.com/study", "title": "Grid Study"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source, [])

    result = await solution_analyzer.analyze_solution(solution_title)

    assert result["solution_title"] == solution_title
    assert len(result["pro_arguments"]) == 1
    assert result["pro_arguments"][0]["argument"]["name"] == "Reduced Blackouts"
    assert result["pro_arguments"][0]["sources"] == mock_source
    assert len(result["counter_arguments"]) == 1
    assert result["counter_arguments"][0]["argument"]["name"] == "Security Vulnerabilities"
    assert result["counter_arguments"][0]["sources"] == mock_source
    assert len(result["metrics"]) == 1
    assert result["metrics"][0]["name"] == "Energy Loss Reduction (%)"
    assert result["stats"]["pro_args_count"] == 1
    assert result["stats"]["counter_args_count"] == 1
    assert result["stats"]["metrics_count"] == 1
    assert result["stats"]["total_pro_sources"] == 1
    assert result["stats"]["total_con_sources"] == 1

    # Verify LLM was called 3 times
    assert mock_llm.arun.call_count == 3
    # Verify source validator called twice (once for pro, once for con)
    assert mock_source_validator.find_supporting_contradicting_sources.call_count == 2

@pytest.mark.asyncio
async def test_analyze_solution_override_success(solution_analyzer, mock_llm, mock_source_validator):
    """Test the full analyze_solution workflow using llm_override."""
    solution_title = "Bioprinting Organs"
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    # Mock responses for override LLM
    pro_args = [{"name": "Reduced Rejection", "search_terms": ["bioprinting immune rejection"]}]
    con_args = [{"name": "Ethical Concerns", "search_terms": ["bioprinting ethics"]}]
    metrics = [{"name": "Time to Viable Organ"}]
    override_responses = [
        create_mock_json_response("arguments", pro_args),
        create_mock_json_response("arguments", con_args),
        create_mock_json_response("metrics", metrics)
    ]
    override_llm.arun.side_effect = override_responses

    # Mock source validator
    mock_source = [{"url": "http://bio.com/paper", "title": "Bio Paper"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_source, [])

    result = await solution_analyzer.analyze_solution(solution_title, llm_override=override_llm)

    assert len(result["pro_arguments"]) == 1
    assert result["pro_arguments"][0]["argument"]["name"] == "Reduced Rejection"
    assert len(result["counter_arguments"]) == 1
    assert len(result["metrics"]) == 1

    # Verify override LLM was called, initial was not
    assert override_llm.arun.call_count == 3
    initial_llm.arun.assert_not_called()
    assert mock_source_validator.find_supporting_contradicting_sources.call_count == 2

@pytest.mark.asyncio
async def test_analyze_solution_llm_error_in_one_stage(solution_analyzer, mock_llm, mock_source_validator):
    """Test analyze_solution when one LLM call fails but others succeed."""
    solution_title = "Fusion Power"

    # Mock responses - pro args succeeds, counter args fails, metrics succeeds
    pro_args = [{"name": "Clean Energy"}]
    metrics = [{"name": "Q value"}]

    mock_llm_responses = [
        create_mock_json_response("arguments", pro_args),             # Success for pro arguments
        AsyncMock(side_effect=Exception("LLM counter args failed")), # Failure for counter arguments
        create_mock_json_response("metrics", metrics)                 # Success for metrics
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    # Mock source validator (only needed for pro args before the error)
    mock_source_validator.find_supporting_contradicting_sources.return_value = ([], [])

    # The overall function should raise the error from the failed stage
    with pytest.raises(SolutionAnalysisError, match="Failed to generate counter arguments"):
        await solution_analyzer.analyze_solution(solution_title)

    # Verify LLM calls: pro args (1), counter args (1 failed)
    assert mock_llm.arun.call_count == 2
    # Verify source validator called once (for pro args)
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()

# Add more tests for source finding errors, JSON parse errors, etc. 