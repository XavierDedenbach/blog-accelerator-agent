import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio
import os # Keep this os import as it's used later for API key checks and skipif
import time
import pytest_asyncio
from typing import Any, List as PyList # Import Any and rename List to PyList
import httpx # ADDED for handling HTTPStatusError

# Langchain imports needed for mocks and integration test
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI # Added for integration test
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage # Corrected import
from langchain_core.prompt_values import PromptValue # Import PromptValue
from openai import RateLimitError as OpenAIRateLimitError # For simulating errors
# JsonOutputParser is imported by the module under test, we patch its method.
from langchain_core.outputs import Generation # For type hinting in parse_result
# RunnableLambda is not directly used in this new approach for the parser mock.

# Import the class to be tested
from agents.research.industry_analysis import IndustryAnalyzer, IndustryAnalysisError
from agents.utilities.source_validator import SourceValidator

# This will be our new mock side_effect function for JsonOutputParser.parse_result
def mock_json_parser_parse_result_side_effect(generations: PyList[Generation], **kwargs: Any) -> Any:
    # print(f"mock_json_parser_parse_result_side_effect CALLED WITH: {type(generations)}, data: {str(generations)[:200]}...")
    
    if not generations or not isinstance(generations, list) or not generations[0]:
        raise ValueError(f"mock_json_parser_parse_result_side_effect received unexpected type or empty generations: {type(generations)}")

    generation_item = generations[0]
    if not isinstance(generation_item, Generation):
        # This check ensures the input to the parser is what's expected from an LLM call (a Generation object).
        raise ValueError(f"mock_json_parser_parse_result_side_effect expected Generation, got: {type(generation_item)}")

    content_to_parse = generation_item.text
    if content_to_parse is None: 
        # Fallback: if Generation.text is None, try to get content from Generation.message (if it's an AIMessage)
        if hasattr(generation_item, 'message') and isinstance(generation_item.message, AIMessage):
            content_to_parse = generation_item.message.content
        else:
            raise ValueError("mock_json_parser_parse_result_side_effect: content_to_parse is None from Generation object and no AIMessage fallback, cannot parse.")
    
    if content_to_parse is None: 
        # Final check if content is still None after fallback.
        raise ValueError("mock_json_parser_parse_result_side_effect: content_to_parse is ultimately None, cannot parse.")

    try:
        return json.loads(content_to_parse) # Attempt to parse the string content as JSON
    except json.JSONDecodeError as e:
        # If parsing fails, re-raise the error so tests for error handling can catch it.
        raise

@pytest.fixture
def mock_llm():
    """Provides a generic mock LLM (ChatOpenAI) for most unit tests.
    Its side_effect simulates different JSON responses based on keywords in the input prompt.
    """
    async_llm_mock = AsyncMock(spec=ChatOpenAI)
    async def side_effect_func(prompt_value_input: PromptValue, *args, **kwargs): 
        actual_string = prompt_value_input.to_string()
        # Default content is an empty JSON object.
        content = "{}" 
        if "identify at least" in actual_string and "critical challenges" in actual_string and "Step 2: Identify Critical Challenges" in actual_string:
            # Simulate response for identifying challenges
            content = json.dumps([{'name': 'Test Challenge', 'description': 'Test Description', 'criticality': 'High', 'search_terms': ['test search']}]) 
        elif "Challenge Name:" in actual_string and "Description:" in actual_string and "Below are sources" in actual_string and "Step 2: Analyze Components" in actual_string:
            # Simulate response for analyzing challenge components
            challenge_name_marker = "Challenge Name: "
            challenge_name_start = actual_string.find(challenge_name_marker)
            challenge_name = "Unknown Challenge" 
            if challenge_name_start != -1:
                challenge_name_end = actual_string.find("\\n", challenge_name_start) 
                if challenge_name_end != -1:
                    challenge_name = actual_string[challenge_name_start + len(challenge_name_marker):challenge_name_end].strip()
            content = json.dumps({
                'risk_factors': [f'Generic Risk for {challenge_name}'],
                'slowdown_factors': [f'Generic Slowdown for {challenge_name}'],
                'cost_factors': [f'Generic Cost for {challenge_name}'],
                'inefficiency_factors': [f'Generic Inefficiency for {challenge_name}']
            })
        return AIMessage(content=content)
    async_llm_mock.side_effect = side_effect_func
    # __signature__ = None is often needed for AsyncMocks used with Langchain's callable system
    # to avoid signature mismatch issues.
    async_llm_mock.__signature__ = None 
    return async_llm_mock

@pytest.fixture
def mock_llm_failing():
    """Provides a mock LLM that always raises an exception, for testing error handling."""
    async_llm_failing_mock = AsyncMock(spec=ChatOpenAI)
    async def side_effect_raising_exception(prompt_value_input: PromptValue, *args, **kwargs):
        raise Exception('LLM call failed') # Simulate a generic LLM failure
    async_llm_failing_mock.side_effect = side_effect_raising_exception
    async_llm_failing_mock.__signature__ = None 
    return async_llm_failing_mock

@pytest.fixture
def mock_source_validator():
    """Provides a mock SourceValidator with a default successful (empty) search result."""
    mock = MagicMock(spec=SourceValidator) 
    # find_supporting_contradicting_sources is an async method, so its mock needs to be an AsyncMock.
    mock.find_supporting_contradicting_sources = AsyncMock(return_value=([], [])) # Returns empty lists for supporting/contradicting sources
    return mock

@pytest_asyncio.fixture
async def industry_analyzer(mock_llm, mock_source_validator):
    """Provides an IndustryAnalyzer instance initialized with mock LLM and SourceValidator.
    It also patches JsonOutputParser.parse_result globally for tests using this fixture,
    directing its calls to our custom mock_json_parser_parse_result_side_effect.
    This is crucial for controlling the output of the JSON parsing step in the LLM chain.
    
    Note: Tests needing real API keys (like integration tests) should not use this fixture,
    or should override its components (llm, source_validator) with real instances.
    The mock_env_variables fixture is NOT automatically applied here; if specific mock
    environment variables are needed for the construction of IndustryAnalyzer (e.g. if it
    read them directly in __init__), those tests would need to explicitly request mock_env_variables.
    However, IndustryAnalyzer takes llm and source_validator as arguments, so direct env var
    dependency at construction is less likely for these components.
    """
    with patch('langchain_core.output_parsers.JsonOutputParser.parse_result', 
               side_effect=mock_json_parser_parse_result_side_effect) as mock_parse_result_method:
        analyzer = IndustryAnalyzer(llm=mock_llm, source_validator=mock_source_validator)
        yield analyzer

# --- Test Cases ---
# Most tests below are unit/component tests that verify specific methods of IndustryAnalyzer
# using the mocked LLM and SourceValidator provided by the 'industry_analyzer' fixture.

@pytest.mark.asyncio
async def test_identify_challenges_success(industry_analyzer, mock_llm):
    """Test successful identification of challenges."""
    topic = "Renewable Energy Storage"
    # print("\\n--- Starting test_identify_challenges_success ---")
    result = await industry_analyzer.identify_challenges(topic)
    # print(f"test_identify_challenges_success result: {result}")
    assert len(result) > 0
    assert "name" in result[0] # Basic check for expected structure
    # mock_llm.assert_called() # .side_effect is called, not the mock itself directly in the new LCEL structure.
    # print("--- Finished test_identify_challenges_success ---\\n")

@pytest.mark.asyncio
async def test_identify_challenges_override_success(industry_analyzer): 
    """Test successful identification of challenges using an overridden LLM."""
    topic = "Fusion Power Commercialization"
    # print("\\n--- Starting test_identify_challenges_override_success ---")
    
    # Create a specific LLM mock for this test case
    override_llm = AsyncMock(spec=ChatOpenAI)
    async def override_side_effect(prompt_value_input: PromptValue, *args, **kwargs):
        actual_string = prompt_value_input.to_string()
        if "Fusion Power Commercialization" in actual_string and "identify at least" in actual_string and "Step 2: Identify Critical Challenges" in actual_string:
            content = json.dumps([{'name': 'Plasma Instability', 'description': 'Difficult to sustain.', 'criticality': 'Very High', 'search_terms': ['tokamak plasma instability']}])
            return AIMessage(content=content)
        return AIMessage(content=json.dumps({})) # Default empty response
    override_llm.side_effect = override_side_effect
    override_llm.__signature__ = None 
    
    result = await industry_analyzer.identify_challenges(topic, llm_override=override_llm)
    # print(f"test_identify_challenges_override_success result: {result}")
    assert len(result) > 0
    assert result[0]['name'] == 'Plasma Instability' # Check specific overridden content
    # print("--- Finished test_identify_challenges_override_success ---\\n")

@pytest.mark.asyncio
async def test_find_sources_for_challenge_success(industry_analyzer, mock_source_validator):
    """Test successful finding of sources for a challenge."""
    # print("\\n--- Starting test_find_sources_for_challenge_success ---")
    challenge = {"name": "Scalability Issues", "search_terms": ["blockchain scalability solutions", "layer 2 scaling"]}
    mock_sources = [{"url": "http://example.com/source1", "title": "Source 1"}, {"url": "http://example.com/source2", "title": "Source 2"}]
    # Configure the mock_source_validator (already injected into industry_analyzer) for this test
    mock_source_validator.find_supporting_contradicting_sources.return_value = (mock_sources, [])
    
    sources = await industry_analyzer.find_sources_for_challenge(challenge, count=2)
    # print(f"test_find_sources_for_challenge_success sources: {sources}")
    assert len(sources) == 2
    mock_source_validator.find_supporting_contradicting_sources.assert_called() # Verify the mock was called
    # print("--- Finished test_find_sources_for_challenge_success ---\\n")

@pytest.mark.asyncio
async def test_analyze_challenge_components_success(industry_analyzer, mock_llm):
    """Test successful analysis of challenge components."""
    # print("\\n--- Starting test_analyze_challenge_components_success ---")
    challenge = {"name": "Grid Integration", "description": "Matching supply/demand.", "criticality": "High", "search_terms": ["grid scale battery integration"]}
    sources = [{"url": "a.com", "title": "s1"}] # Sample sources
    
    result = await industry_analyzer.analyze_challenge_components(challenge, sources)
    # print(f"test_analyze_challenge_components_success result: {result}")
    assert 'risk_factors' in result # Basic check for expected structure based on mock_llm's response
    # print("--- Finished test_analyze_challenge_components_success ---\\n")

@pytest.mark.asyncio
async def test_analyze_challenge_components_override_success(industry_analyzer): 
    """Test successful analysis of challenge components with an overridden LLM."""
    # print("\\n--- Starting test_analyze_challenge_components_override_success ---")
    challenge = {"name": "Grid Integration", "description": "Matching supply/demand.", "criticality": "High", "search_terms": ["grid scale battery integration"]}
    sources = [{"url": "a.com", "title": "s1"}]
    
    # Create a specific LLM mock for this test case
    override_llm_components = AsyncMock(spec=ChatOpenAI)
    async def override_components_side_effect(prompt_value_input: PromptValue, *args, **kwargs):
        actual_string = prompt_value_input.to_string()
        if "Challenge Name: Grid Integration" in actual_string and \
           "Description:" in actual_string and \
           "Below are sources" in actual_string and \
           "Step 2: Analyze Components" in actual_string:
            content = json.dumps({'risk_factors': ['Containment failure'], 'slowdown_factors': ['Long experiment cycles']})
            return AIMessage(content=content)
        return AIMessage(content=json.dumps({})) # Default empty response
    override_llm_components.side_effect = override_components_side_effect
    override_llm_components.__signature__ = None 

    result = await industry_analyzer.analyze_challenge_components(challenge, sources, llm_override=override_llm_components)
    # print(f"test_analyze_challenge_components_override_success result: {result}")
    assert result['risk_factors'] == ['Containment failure'] # Check specific overridden content
    # print("--- Finished test_analyze_challenge_components_override_success ---\\n")

@pytest.mark.asyncio
async def test_analyze_industry_success(industry_analyzer, mock_llm, mock_source_validator):
    """Test the overall analyze_industry workflow with mocked components succeeding."""
    topic = "Renewable Energy Storage"
    # print("\\n--- Starting test_analyze_industry_success ---")
    
    # Reset mocks to ensure clean state for call count assertions if they were re-enabled.
    mock_llm.reset_mock() 
    mock_source_validator.find_supporting_contradicting_sources.reset_mock()

    # Simulate SourceValidator finding one source
    industry_analyzer.source_validator.find_supporting_contradicting_sources.return_value = ([{"url":"test.com", "title":"Test Source"}], [])
    
    result_with_sources = await industry_analyzer.analyze_industry(topic)
    # print(f"test_analyze_industry_success with sources result: {str(result_with_sources)[:500]}")
    assert result_with_sources['topic'] == topic
    assert len(result_with_sources['challenges']) > 0
    assert 'total_sources' in result_with_sources 
    # Expect components to be analyzed because sources were found.
    assert result_with_sources['challenges'][0]['components'] != {} 
    # print("--- Finished test_analyze_industry_success ---\\n")

@pytest.mark.asyncio
async def test_analyze_industry_override_success(industry_analyzer, mock_source_validator): 
    """Test the overall analyze_industry workflow with a specifically overridden LLM for multiple steps."""
    topic = "Fusion Power Commercialization"
    # print("\\n--- Starting test_analyze_industry_override_success ---")
    
    # This LLM mock needs to handle prompts for both identifying challenges AND analyzing components.
    override_llm_industry = AsyncMock(spec=ChatOpenAI)
    async def override_industry_side_effect(prompt_value_input: PromptValue, *args, **kwargs):
        actual_string = prompt_value_input.to_string()
        content_str = "{}" # Default empty JSON
        if "Fusion Power Commercialization" in actual_string and "identify at least" in actual_string and "Step 2: Identify Critical Challenges" in actual_string:
            # Response for identifying challenges
            content_str = json.dumps([{'name': 'Plasma Instability', 'description': 'Difficult to sustain.', 'criticality': 'Very High', 'search_terms': ['tokamak plasma instability']}])
        elif "Challenge Name: Plasma Instability" in actual_string and \
             "Description:" in actual_string and \
             "Below are sources" in actual_string and \
             "Step 2: Analyze Components" in actual_string: 
            # Response for analyzing components of the "Plasma Instability" challenge
            content_str = json.dumps({'risk_factors': ['Override Risk'], 'slowdown_factors': ['Override Slowdown']})
        return AIMessage(content=content_str)
    override_llm_industry.side_effect = override_industry_side_effect
    override_llm_industry.__signature__ = None 

    # Simulate SourceValidator finding one source
    mock_source_validator.find_supporting_contradicting_sources.return_value = ([{"url":"override.com", "title":"Override Source"}], [])
    override_llm_industry.reset_mock() 

    result = await industry_analyzer.analyze_industry(topic, llm_override=override_llm_industry)
    # print(f"test_analyze_industry_override_success result: {str(result)[:500]}")
    assert result["topic"] == topic
    assert len(result["challenges"]) > 0
    assert result["challenges"][0]["name"] == "Plasma Instability" 
    assert result["challenges"][0]["components"]['risk_factors'] == ['Override Risk'] # Check specific component analysis
    # print("--- Finished test_analyze_industry_override_success ---\\n")

@pytest.mark.asyncio
async def test_analyze_industry_no_sources_found(industry_analyzer, mock_llm, mock_source_validator):
    """Test analyze_industry workflow when no sources are found for challenges.
    Component analysis should be skipped for challenges without sources.
    """
    topic = "Underwater Basket Weaving Automation"
    # print("\\n--- Starting test_analyze_industry_no_sources_found ---")
    
    mock_llm.reset_mock()
    # Simulate SourceValidator finding no sources
    mock_source_validator.find_supporting_contradicting_sources.return_value = ([], []) 
    
    result = await industry_analyzer.analyze_industry(topic)
    # print(f"test_analyze_industry_no_sources_found result: {str(result)[:500]}")
    assert result['total_sources'] == 0
    assert len(result['challenges']) > 0 # Challenges should still be identified
    if result['challenges']:
        for ch_data in result['challenges']:
            # Expect empty components and sources for each challenge
            assert ch_data["components"] == {} 
            assert ch_data["sources"] == []
    # print("--- Finished test_analyze_industry_no_sources_found ---\\n")

@pytest.mark.asyncio
async def test_analyze_industry_llm_error_identify(mock_llm_failing, mock_source_validator): 
    """Test error handling when the LLM fails during the 'identify_challenges' step."""
    topic = "Space Elevators"
    # print("\\n--- Starting test_analyze_industry_llm_error_identify ---")
    # Initialize analyzer directly with the failing LLM for this specific test.
    analyzer = IndustryAnalyzer(llm=mock_llm_failing, source_validator=mock_source_validator)
    with pytest.raises(IndustryAnalysisError, match="Failed to identify challenges: LLM call failed"):
        await analyzer.identify_challenges(topic)
    # print("--- Finished test_analyze_industry_llm_error_identify ---\\n")

@pytest.mark.asyncio
async def test_analyze_industry_llm_error_analyze_components(industry_analyzer, mock_source_validator): # Removed mock_llm from params as we define a custom one
    """Test error handling within analyze_industry when LLM fails during 'analyze_challenge_components' for one challenge.
    The overall process should still complete but include error information for the failed challenge.
    """
    topic = "Cybernetic Implants"
    # print("\\n--- Starting test_analyze_industry_llm_error_analyze_components ---")
    
    # Content for a successful initial challenge identification
    challenges_response_content = json.dumps([{'name': 'Biocompatibility', 'description': 'Immune rejection.', 'criticality': 'High', 'search_terms': ['biocompatibility cybernetics']}])
    
    # Custom LLM mock: succeeds for identify_challenges, fails for analyze_challenge_components
    llm_for_test = AsyncMock(spec=ChatOpenAI)
    async def side_effect_for_component_failure(prompt_value_input: PromptValue, *args, **kwargs):
        actual_string = prompt_value_input.to_string()
        if "Cybernetic Implants" in actual_string and "identify at least" in actual_string and "Step 2: Identify Critical Challenges" in actual_string:
            # Succeed for challenge identification
            return AIMessage(content=challenges_response_content)
        elif "Challenge Name: Biocompatibility" in actual_string and \
             "Description:" in actual_string and \
             "Below are sources" in actual_string and \
             "Step 2: Analyze Components" in actual_string: 
            # Fail for component analysis of the "Biocompatibility" challenge
            raise Exception('LLM component analysis failed')
        return AIMessage(content=json.dumps({})) # Default
    llm_for_test.side_effect = side_effect_for_component_failure
    llm_for_test.__signature__ = None 

    # Temporarily replace the LLM in the fixture-provided industry_analyzer instance
    original_analyzer_llm = industry_analyzer.initial_llm 
    industry_analyzer.initial_llm = llm_for_test # Use our custom mock
    
    # Ensure SourceValidator finds sources, so component analysis is attempted
    original_source_return = mock_source_validator.find_supporting_contradicting_sources.return_value
    mock_source_validator.find_supporting_contradicting_sources.return_value = ([{'url': 'c.com', 'title': 's3'}], [])
    mock_source_validator.find_supporting_contradicting_sources.reset_mock()
    llm_for_test.reset_mock()
    
    result = await industry_analyzer.analyze_industry(topic) # Call the main method
    # print(f"test_analyze_industry_llm_error_analyze_components result: {str(result)[:500]}")
    
    assert len(result['challenges']) > 0
    # Check that the first challenge (Biocompatibility) recorded an error during component analysis
    assert 'error' in result['challenges'][0] 
    assert "Failed to analyze challenge components: LLM component analysis failed" in result['challenges'][0]['error']
    
    # print("--- Finished test_analyze_industry_llm_error_analyze_components ---\\n")

    # Restore original LLM to the fixture instance to avoid affecting other tests
    industry_analyzer.initial_llm = original_analyzer_llm 
    mock_source_validator.find_supporting_contradicting_sources.return_value = original_source_return

@pytest.mark.asyncio
async def test_identify_challenges_json_decode_error(industry_analyzer, mock_llm):
    """Test error handling when the LLM returns malformed JSON during 'identify_challenges'."""
    topic = "Faulty JSON Test"
    # print("\\n--- Starting test_identify_challenges_json_decode_error ---")
    
    # Temporarily modify the fixture-provided mock_llm's side_effect
    original_llm_side_effect = mock_llm.side_effect 
    async def faulty_json_side_effect(prompt_value_input: PromptValue, *args, **kwargs):
        actual_string = prompt_value_input.to_string()
        if "Faulty JSON Test" in actual_string and "identify at least" in actual_string and "Step 2: Identify Critical Challenges" in actual_string:
            malformed_content = "{'name': 'Test Challenge', 'description': 'This is not valid JSON because of single quotes and no closing brace'" # Invalid JSON
            return AIMessage(content=malformed_content)
        return AIMessage(content=json.dumps({})) # Default
    mock_llm.side_effect = faulty_json_side_effect
    mock_llm.reset_mock()

    with pytest.raises(IndustryAnalysisError) as excinfo:
        await industry_analyzer.identify_challenges(topic)
    
    # print(f"test_identify_challenges_json_decode_error caught error: {excinfo.value}")
    # Check that the underlying cause was a JSONDecodeError
    assert isinstance(excinfo.value.__context__, json.JSONDecodeError)
    # Check that the error message from IndustryAnalysisError includes details from JSONDecodeError
    assert "Expecting property name enclosed in double quotes" in str(excinfo.value) 
    
    # print("--- Finished test_identify_challenges_json_decode_error ---\\n") 
    mock_llm.side_effect = original_llm_side_effect # Restore original side_effect


# --- Integration Test with Real LLM (Optional) ---
# This test uses real API keys loaded from .env by the load_env_vars_and_debug fixture in conftest.py.
# It does NOT use the mock_env_variables fixture.

@pytest.mark.asyncio
async def test_analyze_industry_integration_real_llm(): 
    """Performs an integration test of the full analyze_industry workflow
    using real API calls to OpenAI and Brave Search. Requires .env file with valid keys.
    """
    # print("\\n--- Starting test_analyze_industry_integration_real_llm ---")
    
    # Get API keys from environment (should be loaded by conftest.py's load_env_vars_and_debug)
    openai_api_key_env = os.getenv("OPENAI_API_KEY")
    brave_api_key_env = os.getenv("BRAVE_API_KEY")
    
    # Commented out debug prints for API keys, as they've served their purpose.
    # print(f"DEBUG: Fetched OPENAI_API_KEY for test: '{openai_api_key_env}'")
    # print(f"DEBUG: Fetched BRAVE_API_KEY for test: '{brave_api_key_env}'")

    # Skip test if keys are not found (e.g. .env file missing or keys not set)
    if not openai_api_key_env:
        pytest.skip("OpenAI API key not found in environment. Skipping integration test.")
    if not brave_api_key_env:
        pytest.skip("Brave API key not found in environment. Skipping integration test.")

    # Initialize with real LLM and SourceValidator using the fetched API keys
    real_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key_env)
    real_source_validator = SourceValidator(brave_api_key=brave_api_key_env) 
    
    analyzer = IndustryAnalyzer(llm=real_llm, source_validator=real_source_validator)
    
    topic = "The future of decentralized finance (DeFi)" # A suitably complex topic for integration testing
    
    try:
        # print(f"Attempting real industry analysis for topic: {topic}")
        start_time = time.time()
        result = await analyzer.analyze_industry(topic)
        end_time = time.time()
        # print(f"Real industry analysis took {end_time - start_time:.2f} seconds.")
        # print(f"test_analyze_industry_integration_real_llm result: {json.dumps(result, indent=2)}")

        # ADDED: Print the first identified challenge from the first LLM call's parsed response
        if result and result.get('challenges'):
            print("\n\n--- Integration Test: First LLM Call (Identify Challenges) - Parsed Response Sample ---")
            if result['challenges']:
                # The 'challenges' list is the result of the first LLM call (identify_challenges)
                # after being processed by the JsonOutputParser.
                # We print the first identified challenge as a sample of the LLM's structured output.
                print(json.dumps(result['challenges'][0], indent=2))
            else:
                print("No challenges were identified by the first LLM call (empty list).")
            print("----------------------------------------------------------------------------------\n")
        elif result:
            print("\n\n--- Integration Test: Identify Challenges call did not produce a 'challenges' list ---")
            print(f"Full result: {json.dumps(result, indent=2)}")
            print("----------------------------------------------------------------------------------\n")
        else:
            print("\n\n--- Integration Test: analyze_industry returned None or empty result ---")

        
        # Basic assertions for the structure and content of the real result
        assert result is not None, "Result should not be None"
        assert result['topic'] == topic, "Result topic should match input topic"
        assert 'challenges' in result, "Result should contain 'challenges'"
        assert len(result['challenges']) >= 1, "Should identify at least one challenge" 
        
        first_challenge = result['challenges'][0]
        assert 'name' in first_challenge, "First challenge should have a name"
        assert 'description' in first_challenge, "First challenge should have a description"
        assert 'criticality' in first_challenge, "First challenge should have criticality"
        assert 'search_terms' in first_challenge, "First challenge should have search_terms"
        assert 'sources' in first_challenge, "First challenge should have sources" # Populated by Brave Search
        assert 'components' in first_challenge, "First challenge should have components"
        
        if first_challenge.get('sources'):
            assert first_challenge.get('components'), "Components should exist if sources were found"
        else:
            # If no sources are found by Brave (which is possible for some queries/challenges),
            # components analysis might be skipped or result in an empty components dict.
            # The current implementation of analyze_industry results in empty components if no sources.
            assert first_challenge.get('components') == {} or not first_challenge.get('components'), \
                "Components should be empty or non-existent if no sources found by Brave"
            # print(f"Note: No sources found by Brave Search for challenge: {first_challenge.get('name')}")

        assert 'total_sources' in result, "Result should contain 'total_sources'"
        # print(f"Total sources found by Brave Search: {result['total_sources']}")
            
    except OpenAIRateLimitError:
        pytest.skip("OpenAI rate limit hit during integration test. Skipping.")
    except httpx.HTTPStatusError as e:
        # Handle potential HTTP errors from Brave Search or other network calls
        if e.response.status_code == 429: # Rate limit
            pytest.skip(f"API rate limit hit (likely Brave Search): {e}")
        elif e.response.status_code == 401: # Unauthorized
             pytest.fail(f"API key unauthorized (likely Brave Search - 401): {e}. Ensure BRAVE_API_KEY is correct.")
        else: # Other HTTP errors
            pytest.fail(f"Integration test failed with HTTPStatusError: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the integration test
        pytest.fail(f"Real LLM/Search integration test failed with an unexpected exception: {e}")
    # finally:
        # print("--- Finished test_analyze_industry_integration_real_llm ---\\n") 