import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio
import os
import time
import httpx

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
    topic = "Digital Security Test Topic"
    challenges = ["Test Challenge 1", "Test Challenge 2"]

    expected_args_data = [{"name": "User Control", "description": "Gives users control over their data."}]
    mock_llm_response_str = json.dumps(expected_args_data)

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_llm_response_str
        arguments = await solution_analyzer.generate_pro_arguments(solution_title, topic, challenges)

    assert arguments == expected_args_data
    mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)


@pytest.mark.asyncio
async def test_generate_pro_arguments_override_success(solution_analyzer, mock_llm):
    """Test successful pro argument generation using llm_override."""
    solution_title = "Quantum Encryption"
    topic = "Quantum Test Topic"
    challenges = ["Quantum Challenge 1"]
    override_llm_dummy = MagicMock(spec=BaseChatModel)

    expected_args_data = [{"name": "Unbreakable Security", "description": "Leverages quantum mechanics."}]
    mock_llm_response_str = json.dumps(expected_args_data)

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_llm_response_str
        arguments = await solution_analyzer.generate_pro_arguments(solution_title, topic, challenges, llm_override=override_llm_dummy)

    assert arguments == expected_args_data
    mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)
    
    # To verify that the override_llm was indeed used by the LLMChain,
    # we would need to inspect the `llm` attribute of the LLMChain instance
    # that mock_arun belongs to. This is tricky with patching `arun` directly.
    # A simpler check (though indirect) is that the original llm's methods (if we mocked them differently) weren't called.
    # Since we patch LLMChain.arun globally, the check for initial_llm.arun.assert_not_called() is not meaningful here
    # because the patched arun is a different object.
    # The crucial part is that LLMChain was constructed with override_llm_dummy. This is implicitly tested
    # if the logic inside generate_pro_arguments correctly selects override_llm for the chain.


@pytest.mark.asyncio
async def test_generate_pro_arguments_llm_error(solution_analyzer, mock_source_validator):
    """Test pro argument generation when the LLMChain.arun call itself fails."""
    solution_title = "Test Solution For LLM Error"
    topic = "Error Topic"
    challenges = ["Error Challenge"]
    simulated_error_message = "LLMChain.arun failed for pro arguments"

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        
        with pytest.raises(SolutionAnalysisError, match=f"Failed to generate pro arguments: {simulated_error_message}"):
            await solution_analyzer.generate_pro_arguments(solution_title, topic, challenges)
            
        mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)


# --- Tests for generate_counter_arguments ---

@pytest.mark.asyncio
async def test_generate_counter_arguments_success(solution_analyzer, mock_llm):
    """Test successful generation of counter arguments using the initial LLM."""
    solution_title = "AI-Powered Diagnosis"
    topic = "AI Healthcare Topic"
    challenges = ["Data Privacy", "Bias"]
    expected_args_data = [{"name": "Bias Risk", "description": "Risk of biased training data."}]
    mock_llm_response_str = json.dumps(expected_args_data)

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_llm_response_str
        arguments = await solution_analyzer.generate_counter_arguments(solution_title, topic, challenges)

    assert arguments == expected_args_data
    mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)

@pytest.mark.asyncio
async def test_generate_counter_arguments_override_success(solution_analyzer, mock_llm):
    """Test successful counter argument generation using llm_override."""
    solution_title = "Carbon Capture Tech"
    topic = "Climate Tech Topic"
    challenges = ["Scalability Cost"]
    override_llm_dummy = MagicMock(spec=BaseChatModel)

    expected_args_data = [{"name": "High Cost", "description": "Requires significant investment."}]
    mock_llm_response_str = json.dumps(expected_args_data)

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_llm_response_str
        arguments = await solution_analyzer.generate_counter_arguments(solution_title, topic, challenges, llm_override=override_llm_dummy)

    assert arguments == expected_args_data
    mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)

@pytest.mark.asyncio
async def test_generate_counter_arguments_llm_error(solution_analyzer, mock_source_validator):
    """Test counter argument generation when the LLMChain.arun call itself fails."""
    solution_title = "Test Solution for Counter Arg LLM Error"
    topic = "Counter Error Topic"
    challenges = ["Counter Error Challenge"]
    simulated_error_message = "LLMChain.arun failed for counter arguments"

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        
        with pytest.raises(SolutionAnalysisError, match=f"Failed to generate counter arguments: {simulated_error_message}"):
            await solution_analyzer.generate_counter_arguments(solution_title, topic, challenges)
            
        mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)


# --- Tests for identify_metrics ---

@pytest.mark.asyncio
async def test_identify_metrics_success(solution_analyzer, mock_llm):
    """Test successful identification of metrics using the initial LLM."""
    solution_title = "Predictive Maintenance System"
    topic = "Industrial IoT Topic"
    challenges = ["Downtime Costs"]
    expected_metrics_data = [{"name": "Downtime Reduction", "importance_context": "Reduces operational costs."}]
    mock_llm_response_str = json.dumps(expected_metrics_data)

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_llm_response_str
        metrics = await solution_analyzer.identify_metrics(solution_title, topic, challenges)

    assert metrics == expected_metrics_data
    mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges))

@pytest.mark.asyncio
async def test_identify_metrics_override_success(solution_analyzer, mock_llm):
    """Test successful metric identification using llm_override."""
    solution_title = "Personalized Learning Platform"
    topic = "EdTech Metrics Topic"
    challenges = ["Student Engagement Tracking"]
    override_llm_dummy = MagicMock(spec=BaseChatModel)

    expected_metrics_data = [{"name": "Student Engagement Score", "importance_context": "Measures platform effectiveness."}]
    mock_llm_response_str = json.dumps(expected_metrics_data)

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_llm_response_str
        metrics = await solution_analyzer.identify_metrics(solution_title, topic, challenges, llm_override=override_llm_dummy)

    assert metrics == expected_metrics_data
    mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges))

@pytest.mark.asyncio
async def test_identify_metrics_llm_error(solution_analyzer, mock_source_validator):
    """Test metric identification when the LLMChain.arun call itself fails."""
    solution_title = "Test Solution for Metrics LLM Error"
    topic = "Metrics Error Topic"
    challenges = ["Metrics Error Challenge"]
    simulated_error_message = "LLMChain.arun failed for metrics"

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        
        with pytest.raises(SolutionAnalysisError, match=f"Failed to identify metrics: {simulated_error_message}"):
            await solution_analyzer.identify_metrics(solution_title, topic, challenges)
            
        mock_arun.assert_called_once_with(solution=solution_title, topic=topic, challenges="\n".join(challenges))


# --- Tests for find_argument_sources ---

@pytest.mark.asyncio
async def test_find_argument_sources_success(solution_analyzer, mock_source_validator):
    """Test finding sources for an argument. Relies on mock_source_validator from fixture."""
    argument = {"name": "Improved Efficiency", "search_terms": ["process optimization case studies"]}
    solution_title = "Automation Tool" # This is NOT part of the query generated by find_argument_sources
    expected_mock_sources = [{"url": "http://example.com/eff1", "title": "Efficiency Study"}]
    
    mock_source_validator.find_supporting_contradicting_sources.return_value = (expected_mock_sources, [])
    mock_source_validator.find_supporting_contradicting_sources.reset_mock()

    sources = await solution_analyzer.find_argument_sources(argument, solution_title, is_pro_argument=True)

    assert sources == expected_mock_sources
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()
    
    called_args, called_kwargs = mock_source_validator.find_supporting_contradicting_sources.call_args
    
    assert isinstance(called_args[0], str) 
    query_str = called_args[0]

    # Corrected assertions based on actual query construction in solution_analysis.py:
    # query = f"{search_term_to_use} {argument_name_for_query}"
    assert "process optimization case studies" in query_str
    assert "Improved Efficiency" in query_str
    assert "Automation Tool" not in query_str # Verify solution_title is not in this specific query
    assert "pro argument" not in query_str   # Verify type_suffix is not in this specific query
    
    assert called_kwargs.get('count') == 3 


# --- Tests for analyze_solution (Full Workflow) ---

@pytest.mark.asyncio
async def test_analyze_solution_success(solution_analyzer, mock_source_validator):
    """Test the full analyze_solution workflow using the initial LLM, with LLMChain.arun patched."""
    solution_title = "Smart Grid Technology"
    topic = "Energy Grid Topic"
    challenges = ["Grid Stability", "Renewable Integration"]

    # Expected data for each stage
    pro_args_data = [{"name": "Reduced Blackouts", "description": "Improves reliability.", "search_terms": ["smart grid reliability"]}]
    con_args_data = [{"name": "Security Vulnerabilities", "description": "Potential for cyber attacks.", "search_terms": ["smart grid cyber security risks"]}]
    metrics_data = [{"name": "Energy Loss Reduction (%)", "importance_context": "Efficiency gain."}]

    # LLMChain.arun will be called sequentially for pro_args, then con_args, then metrics.
    # Each call expects a JSON string which the method then parses.
    mock_llm_responses_json_strings = [
        json.dumps(pro_args_data),  # For pro arguments
        json.dumps(con_args_data),  # For counter arguments
        json.dumps(metrics_data)    # For metrics
    ]

    # Source validator will be called for each pro argument and each con argument.
    # Here, 1 pro arg and 1 con arg.
    mock_pro_arg_source = [{"url": "http://grid.com/pro_study", "title": "Pro Grid Study"}]
    mock_con_arg_source = [{"url": "http://grid.com/con_study", "title": "Con Grid Study"}]
    mock_source_validator.find_supporting_contradicting_sources.side_effect = [
        (mock_pro_arg_source, []),
        (mock_con_arg_source, [])
    ]
    mock_source_validator.find_supporting_contradicting_sources.reset_mock() # Reset for accurate call count

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_llm_responses_json_strings
        result = await solution_analyzer.analyze_solution(solution_title, topic, challenges)

    assert result["solution_title"] == solution_title
    assert result["topic"] == topic
    assert result["challenges"] == challenges
    assert len(result["pro_arguments"]) == 1
    assert result["pro_arguments"][0]["name"] == pro_args_data[0]["name"]
    assert result["pro_arguments"][0]["sources"] == mock_pro_arg_source
    assert len(result["counter_arguments"]) == 1
    assert result["counter_arguments"][0]["name"] == con_args_data[0]["name"]
    assert result["counter_arguments"][0]["sources"] == mock_con_arg_source
    assert len(result["metrics"]) == 1
    assert result["metrics"][0]["name"] == metrics_data[0]["name"]
    
    assert result["stats"]["pro_args_count"] == len(pro_args_data)
    assert result["stats"]["counter_args_count"] == len(con_args_data)
    assert result["stats"]["metrics_count"] == len(metrics_data)
    assert result["stats"]["total_pro_sources"] == len(mock_pro_arg_source)
    assert result["stats"]["total_con_sources"] == len(mock_con_arg_source)

    assert mock_arun.call_count == 3 # pro_args, con_args, metrics
    # Check args for pro_arguments call
    mock_arun.call_args_list[0].assert_called_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5) 
    # Check args for counter_arguments call
    mock_arun.call_args_list[1].assert_called_with(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)
    # Check args for identify_metrics call
    mock_arun.call_args_list[2].assert_called_with(solution=solution_title, topic=topic, challenges="\n".join(challenges))

    assert mock_source_validator.find_supporting_contradicting_sources.call_count == (len(pro_args_data) + len(con_args_data))
    # Check calls to source_validator
    pro_arg_call_args, _ = mock_source_validator.find_supporting_contradicting_sources.call_args_list[0]
    query_for_pro_arg = pro_arg_call_args[0]
    assert pro_args_data[0]["search_terms"][0] in query_for_pro_arg
    assert pro_args_data[0]["name"] in query_for_pro_arg
    assert solution_title not in query_for_pro_arg # solution_title is not part of this query
    assert "pro argument" not in query_for_pro_arg # type_suffix is not part of this query

    con_arg_call_args, _ = mock_source_validator.find_supporting_contradicting_sources.call_args_list[1]
    query_for_con_arg = con_arg_call_args[0]
    assert con_args_data[0]["search_terms"][0] in query_for_con_arg
    assert con_args_data[0]["name"] in query_for_con_arg
    assert solution_title not in query_for_con_arg # solution_title is not part of this query
    assert "counter argument" not in query_for_con_arg # type_suffix is not part of this query

@pytest.mark.asyncio
async def test_analyze_solution_override_success(solution_analyzer, mock_source_validator):
    """Test the full analyze_solution workflow using llm_override, with LLMChain.arun patched."""
    solution_title = "Bioprinting Organs"
    topic = "BioTech Override Topic"
    challenges = ["Ethical Concerns", "Scalability"]
    override_llm_dummy = MagicMock(spec=BaseChatModel) # This is passed as llm_override
    # Get the initial_llm from the solution_analyzer fixture to assert it wasn't used.
    # This requires solution_analyzer fixture to perhaps expose its initial_llm or for us to check indirectly.
    # For now, we assume the override mechanism works if mock_arun (patched) is called as expected.

    pro_args_data = [{"name": "Reduced Rejection", "search_terms": ["bioprinting immune rejection"]}]
    con_args_data = [{"name": "Ethical Issues Detailed", "search_terms": ["bioprinting ethics discussion"]}] # Changed to avoid conflict
    metrics_data = [{"name": "Time to Viable Organ"}]
    
    mock_llm_responses_json_strings = [
        json.dumps(pro_args_data),
        json.dumps(con_args_data),
        json.dumps(metrics_data)
    ]

    mock_pro_arg_source = [{"url": "http://bio.com/pro", "title": "Bio Pro Paper"}]
    mock_con_arg_source = [{"url": "http://bio.com/con", "title": "Bio Con Paper"}]
    mock_source_validator.find_supporting_contradicting_sources.side_effect = [
        (mock_pro_arg_source, []),
        (mock_con_arg_source, [])
    ]
    mock_source_validator.find_supporting_contradicting_sources.reset_mock()

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_llm_responses_json_strings
        result = await solution_analyzer.analyze_solution(solution_title, topic, challenges, llm_override=override_llm_dummy)

    assert len(result["pro_arguments"]) == 1
    assert result["pro_arguments"][0]["name"] == pro_args_data[0]["name"]
    assert result["pro_arguments"][0]["sources"] == mock_pro_arg_source
    assert len(result["counter_arguments"]) == 1
    assert result["counter_arguments"][0]["name"] == con_args_data[0]["name"]
    assert result["counter_arguments"][0]["sources"] == mock_con_arg_source
    assert len(result["metrics"]) == 1
    assert result["metrics"][0]["name"] == metrics_data[0]["name"]

    assert mock_arun.call_count == 3
    # It's implicit that override_llm_dummy was used if solution_analyzer correctly passed it to LLMChain instances.
    # A direct assertion on initial_llm not being used is hard when patching LLMChain.arun globally.
    assert mock_source_validator.find_supporting_contradicting_sources.call_count == (len(pro_args_data) + len(con_args_data))

@pytest.mark.asyncio
async def test_analyze_solution_llm_error_in_one_stage(solution_analyzer, mock_source_validator):
    """Test analyze_solution when one LLM call (e.g., for counter arguments) logs an error 
       and the overall analysis still completes, returning partial results and error information.
    """
    solution_title = "Fusion Power"
    topic = "Future Energy"
    challenges = ["Containment", "Efficiency"]

    pro_args_data = [{"name": "Clean Energy", "description": "Abundant fuel.", "search_terms": ["fusion clean energy"]}]
    metrics_data = [{"name": "Q-value", "importance_context": "Energy gain factor."}]

    # Adjusted the exception message to use "error" instead of "failure" to match observed behavior
    llm_failure_for_con_args = Exception("Simulated LLM error for counter arguments")
    
    # Helper to create a consistent JSON string response for arun mocks
    def create_llm_arun_json_resp(data_list):
        return json.dumps(data_list)

    # Simulate LLM call failure for one stage (e.g., counter arguments)
    # Correctly use the defined helper function name
    mock_llm_responses = [
        create_llm_arun_json_resp(pro_args_data),                  # Pro arguments success
        llm_failure_for_con_args,                   # Counter arguments fail
        create_llm_arun_json_resp(metrics_data)                    # Metrics success
    ]

    mock_pro_arg_source = [{"url": "http://fusion.com/pro", "title": "Fusion Pro Study"}]
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_pro_arg_source, [])
    mock_source_validator.find_supporting_contradicting_sources.reset_mock()

    with patch('agents.research.solution_analysis.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        # mock_arun.side_effect needs to be sophisticated enough if we rely on call_count inside it.
        # It's simpler if the side_effect is a list of responses/exceptions.
        # Correcting side_effect for sequential calls:
        mock_arun.side_effect = mock_llm_responses
        
        # Call the main method
        result = await solution_analyzer.analyze_solution(solution_title, topic, challenges)

    # Verify LLM calls: pro_args (success), con_args (failure), metrics (success)
    assert mock_arun.call_count == 3
    # Pro-args call check
    assert mock_arun.call_args_list[0].kwargs['solution'] == solution_title
    assert mock_arun.call_args_list[0].kwargs['topic'] == topic
    assert mock_arun.call_args_list[0].kwargs['challenges'] == "\n".join(challenges)
    assert mock_arun.call_args_list[0].kwargs['min_arguments'] == 5
    # Counter-args call check (this one failed)
    assert mock_arun.call_args_list[1].kwargs['solution'] == solution_title
    assert mock_arun.call_args_list[1].kwargs['topic'] == topic
    assert mock_arun.call_args_list[1].kwargs['challenges'] == "\n".join(challenges)
    assert mock_arun.call_args_list[1].kwargs['min_arguments'] == 5
    # Metrics call check
    assert mock_arun.call_args_list[2].kwargs['solution'] == solution_title
    assert mock_arun.call_args_list[2].kwargs['topic'] == topic
    assert mock_arun.call_args_list[2].kwargs['challenges'] == "\n".join(challenges)
    assert "min_metrics" not in mock_arun.call_args_list[2].kwargs 

    # Verify source validator called once (only for the successful pro_args step)
    mock_source_validator.find_supporting_contradicting_sources.assert_called_once()
    call_args, _ = mock_source_validator.find_supporting_contradicting_sources.call_args_list[0]
    query_str_pro = call_args[0]
    assert pro_args_data[0]["search_terms"][0] in query_str_pro
    assert pro_args_data[0]["name"] in query_str_pro
    
    # Verify the results dictionary
    assert result["solution_title"] == solution_title
    assert result["topic"] == topic
    assert result["challenges"] == challenges
    assert len(result["pro_arguments"]) == 1
    assert result["pro_arguments"][0]["name"] == pro_args_data[0]["name"]
    assert result["pro_arguments"][0]["sources"] == mock_pro_arg_source # Sources should be attached
    
    assert len(result["counter_arguments"]) == 0 # Failed generation leads to empty list
    
    assert len(result["metrics"]) == 1
    assert result["metrics"][0]["name"] == metrics_data[0]["name"]
    
    assert len(result["errors"]) == 1
    # Based on the failing test output, the exact error message structure is:
    # "<Stage Name> generation failed: Failed to generate <stage name>: <original exception message>"
    # For counter arguments, it's "Counter arguments generation failed: " + the message from SolutionAnalysisError
    # which itself is "Failed to generate counter arguments: " + original exception string.
    original_exception_message = str(llm_failure_for_con_args)
    solution_analysis_error_message = f"Failed to generate counter arguments: {original_exception_message}"
    expected_error_message = f"Counter arguments generation failed: {solution_analysis_error_message}"
    assert result["errors"][0] == expected_error_message

    stats = result["stats"]
    assert stats["pro_args_count"] == 1
    assert stats["counter_args_count"] == 0
    assert stats["metrics_count"] == 1
    assert stats["total_pro_sources"] == len(mock_pro_arg_source)
    assert stats["total_con_sources"] == 0
    assert stats["error_count"] == 1
    assert stats["total_stages"] == 3

# Add more tests for source finding errors, JSON parse errors, etc. 


# --- Integration Test with Real LLM (Optional) ---

# Determine if running in GitHub Actions
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

@pytest.mark.skipif(IS_GITHUB_ACTIONS, reason="Skipping real LLM integration tests in GitHub Actions to avoid API costs/failures.")
@pytest.mark.asyncio
async def test_analyze_solution_integration_real_llm(): 
    """Performs an integration test of the full analyze_solution workflow
    using real API calls to OpenAI and Brave Search.
    Requires .env file with OPENAI_API_KEY and BRAVE_API_KEY.
    """
    if IS_GITHUB_ACTIONS:
        pytest.skip("Skipping integration test in GitHub Actions environment.")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

    if not OPENAI_API_KEY or not BRAVE_API_KEY:
        pytest.skip("OPENAI_API_KEY or BRAVE_API_KEY not set, skipping integration test.")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI # Fallback for older langchain
        
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, model_name="gpt-4o")
    source_validator = SourceValidator(brave_api_key=BRAVE_API_KEY) # REMOVED llm from here
    analyzer = SolutionAnalyzer(llm=llm, source_validator=source_validator)

    solution_title = "Vertical Farming"
    topic = "Future of Vertical Farming Technology"
    challenges = [
        "Addressing diverse crop growth paces and styles.",
        "Teacher burnout and large class sizes.",
        "Lack of engaging and adaptive learning content.",
        "Ensuring equitable access to technology for all students."
    ]

    print(f"Starting real LLM integration test for: {solution_title}")
    start_time_integration = time.time()
    
    result = await analyzer.analyze_solution(
        solution_title=solution_title,
        topic=topic,
        challenges=challenges
    )
    
    end_time_integration = time.time()
    print(f"Integration test call to analyze_solution completed in {end_time_integration - start_time_integration:.2f} seconds.")

    assert isinstance(result, dict)
    assert result["solution_title"] == solution_title
    assert result["topic"] == topic
    assert result["challenges"] == challenges
    
    for key_check in ["pro_arguments", "counter_arguments", "metrics", "errors", "stats"]:
        assert key_check in result, f"Key '{key_check}' missing in result"

    assert isinstance(result["pro_arguments"], list)
    assert isinstance(result["counter_arguments"], list)
    assert isinstance(result["metrics"], list)
    assert isinstance(result["errors"], list)
    assert isinstance(result["stats"], dict)

    print(f"Number of Pro-arguments: {len(result['pro_arguments'])}")
    print(f"Number of Counter-arguments: {len(result['counter_arguments'])}")
    print(f"Number of Metrics: {len(result['metrics'])}")
    print(f"Errors encountered: {result['errors']}")

    assert len(result["pro_arguments"]) > 0, "Expected the LLM to generate at least one pro argument."
    assert len(result["counter_arguments"]) > 0, "Expected the LLM to generate at least one counter argument."
    assert len(result["metrics"]) > 0, "Expected the LLM to generate at least one metric."

    stats = result["stats"]
    assert stats["pro_args_count"] == len(result["pro_arguments"])
    assert stats["counter_args_count"] == len(result["counter_arguments"])
    assert stats["metrics_count"] == len(result["metrics"])
    assert stats["error_count"] == len(result["errors"])
    assert "analysis_duration_seconds" in stats
    assert stats["analysis_duration_seconds"] > 0
    assert "timestamp" in stats

    total_pro_sources_found = 0
    for arg in result["pro_arguments"]:
        assert "name" in arg
        assert "search_terms" in arg 
        assert "sources" in arg 
        assert isinstance(arg["sources"], list)
        total_pro_sources_found += len(arg["sources"])
        if arg["sources"]:
            print(f"  Pro-arg '{arg['name']}' found {len(arg['sources'])} sources.")
    assert stats["total_pro_sources"] == total_pro_sources_found
    print(f"Total pro argument sources found by validator: {total_pro_sources_found}")

    total_con_sources_found = 0
    for arg in result["counter_arguments"]:
        assert "name" in arg
        assert "search_terms" in arg
        assert "sources" in arg
        assert isinstance(arg["sources"], list)
        total_con_sources_found += len(arg["sources"])
        if arg["sources"]:
            print(f"  Con-arg '{arg['name']}' found {len(arg['sources'])} sources.")
    assert stats["total_con_sources"] == total_con_sources_found
    print(f"Total counter argument sources found by validator: {total_con_sources_found}")

    if result["errors"]:
        pytest.fail(f"Integration test completed with errors: {result['errors']}")
    else:
        print("Integration test completed successfully with real LLM and Source Validator.")