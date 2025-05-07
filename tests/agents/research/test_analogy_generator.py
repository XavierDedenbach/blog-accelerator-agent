import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio

# Langchain imports needed for mocks
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import Generation, LLMResult
from openai import RateLimitError as OpenAIRateLimitError # For simulating errors

# Import the class to be tested
from agents.research.analogy_generator import AnalogyGenerator, AnalogyGenerationError
from agents.utilities.source_validator import SourceValidator
from agents.utilities.firecrawl_client import FirecrawlClient


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
    # Mock methods used by AnalogyGenerator
    validator.search_web = AsyncMock(return_value=[]) # For search_existing_analogies & visual fallback
    validator.find_supporting_contradicting_sources = AsyncMock(return_value=([], [])) # If needed elsewhere
    return validator

@pytest.fixture
def mock_firecrawl_client():
    """Creates a mock FirecrawlClient."""
    client = MagicMock(spec=FirecrawlClient)
    client.search_images = AsyncMock(return_value=[]) # Default to no results
    return client

@pytest.fixture
def analogy_generator(mock_llm, mock_source_validator, mock_firecrawl_client):
    """Creates an instance of AnalogyGenerator with mock dependencies."""
    return AnalogyGenerator(
        llm=mock_llm,
        source_validator=mock_source_validator,
        firecrawl_client=mock_firecrawl_client,
        min_analogies=1 # Lower for easier testing
    )


# --- Test Cases ---

# Helper function to create mock JSON responses
def create_mock_json_response(data):
    return json.dumps(data)

# Sample data for tests
test_concept = "Blockchain Technology"
mock_analogy = {
    "title": "Digital Ledger",
    "domain": "Technology & Computing",
    "description": "Like a shared, immutable spreadsheet.",
    "explanation": "Shows distributed trust.",
    "mapping": "Block -> Record, Chain -> Linked List",
    "limitations": "Doesn't capture consensus well.",
    "visual_description": "Diagram of linked blocks."
}
mock_evaluation = {
    "clarity": {"score": 8, "explanation": "Clear"},
    "accuracy": {"score": 7, "explanation": "Mostly accurate"},
    "overall_score": 7.5
}
mock_refined_analogy = {**mock_analogy, "title": "Improved Digital Ledger"}
mock_existing_analogy = {
    "source": "http://example.com/blog",
    "analogy": "Like a Google Doc",
    "mapping": "Shared state",
    "credited_to": "Some Blogger"
}
mock_visual_asset = {"url": "http://images.com/block.png", "title": "Block Diagram"}


# --- Tests for generate_domain_analogies ---

@pytest.mark.asyncio
async def test_generate_domain_analogies_success(analogy_generator):
    """Test successful generation of domain analogies by patching LLMChain.arun."""
    topic = test_concept
    expected_analogies_list = [mock_analogy]
    mock_chain_output_string = json.dumps(expected_analogies_list)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string
        
        analogies = await analogy_generator.generate_domain_analogies(topic)
        
        assert analogies == expected_analogies_list
        mock_arun.assert_called_once()
        
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['concept'] == topic
        assert called_kwargs['min_analogies'] == analogy_generator.min_analogies
        assert "Nature & Biology" in called_kwargs['domains']

@pytest.mark.asyncio
async def test_generate_domain_analogies_override_success(analogy_generator):
    """Test successful domain analogy generation using llm_override, by patching LLMChain.arun."""
    topic = test_concept
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    mock_chain_output_string_override = json.dumps(mock_evaluation)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override
        
        evaluation = await analogy_generator.evaluate_analogy(test_concept, mock_analogy, llm_override=override_llm_dummy)

    assert evaluation == mock_evaluation
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert mock_analogy['title'] in called_kwargs['analogy']

@pytest.mark.asyncio
async def test_generate_domain_analogies_llm_error(analogy_generator):
    """Test domain analogy generation when LLMChain.arun call itself fails."""
    topic = test_concept
    simulated_error_message = "LLMChain.arun failed for domain analogies"

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        
        with pytest.raises(AnalogyGenerationError, match=f"Failed to generate analogies: {simulated_error_message}"):
            await analogy_generator.generate_domain_analogies(topic)
            
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['concept'] == topic
        assert called_kwargs['min_analogies'] == analogy_generator.min_analogies
        assert "Nature & Biology" in called_kwargs['domains']

@pytest.mark.asyncio
async def test_generate_domain_analogies_json_error(analogy_generator, mock_llm):
    """Test domain analogy generation with invalid JSON response.
    This test ensures that if the LLM returns a string that is not valid JSON,
    the _parse_llm_response_to_json method catches the JSONDecodeError
    and the overall function raises an AnalogyGenerationError with the correct message.
    """
    invalid_json_string = "This is not valid JSON {{{{:} ``"

    # Patch LLMChain.arun directly for this test to ensure it returns the invalid string
    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_chain_arun:
        mock_chain_arun.return_value = invalid_json_string

        # The _parse_llm_response_to_json will raise an AnalogyGenerationError with a message like:
        # "Failed to parse domain analogies for <concept> JSON: Expecting value: line 1 column 1 (char 0)"
        # This is then caught by generate_domain_analogies and re-raised as:
        # "Failed to generate analogies: Failed to parse domain analogies for <concept> JSON: Expecting value: line 1 column 1 (char 0)"
        expected_error_regex = (
            r"Failed to generate analogies: Failed to parse domain analogies for .*? JSON: "
            r"Expecting value: line 1 column 1 \(char 0\)"
        )
        with pytest.raises(AnalogyGenerationError, match=expected_error_regex):
            await analogy_generator.generate_domain_analogies(test_concept)
        
        mock_chain_arun.assert_called_once()


# --- Tests for evaluate_analogy ---

@pytest.mark.asyncio
async def test_evaluate_analogy_success(analogy_generator):
    """Test successful evaluation of an analogy by patching LLMChain.arun."""
    mock_chain_output_string = json.dumps(mock_evaluation)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string
        
        evaluation = await analogy_generator.evaluate_analogy(test_concept, mock_analogy)

    assert evaluation == mock_evaluation
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert mock_analogy['title'] in called_kwargs['analogy']

@pytest.mark.asyncio
async def test_evaluate_analogy_override_success(analogy_generator):
    """Test successful analogy evaluation using llm_override, by patching LLMChain.arun."""
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    mock_chain_output_string_override = json.dumps(mock_evaluation)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override
        
        evaluation = await analogy_generator.evaluate_analogy(test_concept, mock_analogy, llm_override=override_llm_dummy)

    assert evaluation == mock_evaluation
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert mock_analogy['title'] in called_kwargs['analogy']

@pytest.mark.asyncio
async def test_evaluate_analogy_llm_error(analogy_generator):
    """Test analogy evaluation when LLMChain.arun call itself fails."""
    simulated_error_message = "LLMChain.arun failed for analogy evaluation"

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        
        with pytest.raises(AnalogyGenerationError, match=f"Failed to evaluate analogy: {simulated_error_message}"):
            await analogy_generator.evaluate_analogy(test_concept, mock_analogy)
            
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['concept'] == test_concept
        assert mock_analogy['title'] in called_kwargs['analogy']


# --- Tests for refine_analogy ---

@pytest.mark.asyncio
async def test_refine_analogy_success(analogy_generator):
    """Test successful refinement of an analogy by patching LLMChain.arun."""
    mock_chain_output_string = json.dumps(mock_refined_analogy)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string
        
        refined = await analogy_generator.refine_analogy(test_concept, mock_analogy, mock_evaluation)

    assert refined == mock_refined_analogy
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert mock_analogy['title'] in called_kwargs['analogy']
    assert "Clarity: 8/10 - Clear" in called_kwargs['evaluation']

@pytest.mark.asyncio
async def test_refine_analogy_override_success(analogy_generator):
    """Test successful analogy refinement using llm_override, by patching LLMChain.arun."""
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    mock_chain_output_string_override = json.dumps(mock_refined_analogy)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override
        
        refined = await analogy_generator.refine_analogy(test_concept, mock_analogy, mock_evaluation, llm_override=override_llm_dummy)

    assert refined == mock_refined_analogy
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert mock_analogy['title'] in called_kwargs['analogy']
    assert "Clarity: 8/10 - Clear" in called_kwargs['evaluation']

@pytest.mark.asyncio
async def test_refine_analogy_llm_error(analogy_generator):
    """Test analogy refinement when LLMChain.arun call itself fails."""
    simulated_error_message = "LLMChain.arun failed for analogy refinement"

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)
        
        with pytest.raises(AnalogyGenerationError, match=f"Failed to refine analogy: {simulated_error_message}"):
            await analogy_generator.refine_analogy(test_concept, mock_analogy, mock_evaluation)
            
        mock_arun.assert_called_once()
        called_kwargs = mock_arun.call_args.kwargs
        assert called_kwargs['concept'] == test_concept
        assert mock_analogy['title'] in called_kwargs['analogy']
        assert "Clarity: 8/10 - Clear" in called_kwargs['evaluation']


# --- Tests for search_existing_analogies ---

@pytest.mark.asyncio
async def test_search_existing_analogies_success(analogy_generator, mock_source_validator):
    """Test successful search for existing analogies by patching LLMChain.arun."""
    mock_search_results = [{"title": "Blog Post", "url": "http://example.com/blog", "description": "Explains blockchain..."}]
    mock_source_validator.search_web.return_value = mock_search_results

    expected_existing_list = [mock_existing_analogy]
    mock_chain_output_string = json.dumps(expected_existing_list)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string

        existing = await analogy_generator.search_existing_analogies(test_concept)

    assert existing == expected_existing_list
    mock_source_validator.search_web.assert_called_once_with(f"{test_concept} analogy metaphor explanation", count=5)
    
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert "Blog Post" in called_kwargs['results']

@pytest.mark.asyncio
async def test_search_existing_analogies_override_success(analogy_generator, mock_source_validator):
    """Test successful existing analogy search using llm_override, by patching LLMChain.arun."""
    override_llm_dummy = MagicMock(spec=BaseChatModel)
    
    mock_search_results = [{"title": "Override Blog", "url": "http://override.com/blog", "description": "Desc..."}]
    mock_source_validator.search_web.return_value = mock_search_results

    expected_existing_list = [mock_existing_analogy]
    mock_chain_output_string_override = json.dumps(expected_existing_list)

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = mock_chain_output_string_override

        existing = await analogy_generator.search_existing_analogies(test_concept, llm_override=override_llm_dummy)

    assert existing == expected_existing_list
    mock_source_validator.search_web.assert_called_once()
    # Ensure correct query for source_validator if needed, though covered in non-override test
    
    mock_arun.assert_called_once()
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert "Override Blog" in called_kwargs['results'] # Check results from override search were passed to LLM

@pytest.mark.asyncio
async def test_search_existing_analogies_llm_error(analogy_generator, mock_source_validator):
    """Test existing analogy search when LLMChain.arun fails."""
    mock_search_results = [{"title": "Blog Post", "url": "http://example.com/blog", "description": "Desc..."}]
    mock_source_validator.search_web.return_value = mock_search_results

    simulated_error_message = "LLMChain.arun failed for searching existing analogies"
    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = Exception(simulated_error_message)

        with pytest.raises(AnalogyGenerationError, match=f"Failed to search existing analogies: {simulated_error_message}"):
            await analogy_generator.search_existing_analogies(test_concept)
    
    mock_source_validator.search_web.assert_called_once()
    mock_arun.assert_called_once() # LLMChain.arun was called
    called_kwargs = mock_arun.call_args.kwargs
    assert called_kwargs['concept'] == test_concept
    assert "Blog Post" in called_kwargs['results']

@pytest.mark.asyncio
async def test_search_existing_analogies_no_search_results(analogy_generator, mock_llm, mock_source_validator):
    """Test existing analogy search when web search yields no results."""
    mock_source_validator.search_web.return_value = [] # No results

    existing = await analogy_generator.search_existing_analogies(test_concept)

    assert existing == [] # Should return empty list
    mock_source_validator.search_web.assert_called_once()
    mock_llm.arun.assert_not_called() # LLM shouldn't be called if no search results


# --- Tests for generate_visual_representation ---

@pytest.mark.asyncio
async def test_generate_visual_representation_firecrawl_success(analogy_generator, mock_firecrawl_client):
    """Test successful visual generation using Firecrawl."""
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    result = await analogy_generator.generate_visual_representation(mock_analogy)

    assert result["success"] is True
    assert len(result["assets"]) == 1
    assert result["assets"][0] == mock_visual_asset
    assert result["query"] == f"{mock_analogy['domain']} {mock_analogy['title']} visual diagram illustration"
    mock_firecrawl_client.search_images.assert_called_once()

@pytest.mark.asyncio
async def test_generate_visual_representation_brave_fallback_success(analogy_generator, mock_firecrawl_client, mock_source_validator):
    """Test successful visual generation using Brave fallback."""
    mock_firecrawl_client.search_images.return_value = [] # Firecrawl finds nothing
    mock_brave_result = [{"url": "http://brave.com/img.jpg", "title": "Brave Image", "description": "From Brave", "source": "Brave Search"}]
    mock_source_validator.search_web.return_value = mock_brave_result

    result = await analogy_generator.generate_visual_representation(mock_analogy)

    assert result["success"] is True
    assert len(result["assets"]) == 1
    assert result["assets"][0]["url"] == mock_brave_result[0]["url"]
    assert result["assets"][0]["source"] == "Brave Search" # Check source is Brave
    mock_firecrawl_client.search_images.assert_called_once()
    mock_source_validator.search_web.assert_called_once()
    call_args, call_kwargs = mock_source_validator.search_web.call_args
    assert "image" in call_args[0] # Check query includes 'image'

@pytest.mark.asyncio
async def test_generate_visual_representation_no_results(analogy_generator, mock_firecrawl_client, mock_source_validator):
    """Test visual generation when neither Firecrawl nor Brave finds results."""
    mock_firecrawl_client.search_images.return_value = []
    mock_source_validator.search_web.return_value = [] # Brave also finds nothing

    result = await analogy_generator.generate_visual_representation(mock_analogy)

    assert result["success"] is False
    assert len(result["assets"]) == 0
    mock_firecrawl_client.search_images.assert_called_once()
    mock_source_validator.search_web.assert_called_once()

@pytest.mark.asyncio
async def test_generate_visual_representation_firecrawl_error_fallback_success(analogy_generator, mock_firecrawl_client, mock_source_validator):
    """Test visual generation fallback when Firecrawl raises an error."""
    mock_firecrawl_client.search_images.side_effect = Exception("Firecrawl API down")
    mock_brave_result = [{"url": "http://brave.com/img.jpg", "title": "Brave Image", "source": "Brave Search"}]
    mock_source_validator.search_web.return_value = mock_brave_result

    result = await analogy_generator.generate_visual_representation(mock_analogy)

    assert result["success"] is True
    assert len(result["assets"]) == 1
    assert result["assets"][0]["url"] == mock_brave_result[0]["url"]
    mock_firecrawl_client.search_images.assert_called_once()
    mock_source_validator.search_web.assert_called_once()


# --- Tests for generate_analogies (Full Workflow) ---

@pytest.mark.asyncio
async def test_generate_analogies_success_no_refinement(analogy_generator, mock_source_validator, mock_firecrawl_client):
    """Test full workflow: analogy generated, score >= threshold, existing found, visual found, using patched LLMChain.arun."""
    high_score_evaluation = {"overall_score": 8.0}
    # These are the JSON strings that LLMChain.arun will return sequentially
    # Order of LLM calls in generate_analogies: 
    # 1. generate_domain_analogies
    # 2. search_existing_analogies (LLM part, after source_validator.search_web)
    # 3. evaluate_analogy (for each generated analogy)
    # (refine_analogy is not called in this scenario)
    mock_chain_arun_side_effects = [
        json.dumps([mock_analogy]),                # For generate_domain_analogies
        json.dumps([mock_existing_analogy]),       # For search_existing_analogies (LLM part)
        json.dumps(high_score_evaluation),         # For evaluate_analogy (mock_analogy)
    ]

    # Mock search results for existing analogies (SourceValidator part)
    mock_source_validator.search_web.return_value = [{"title": "Blog", "url": "http://example.com/blog", "description": "Desc..."}]
    # Mock visual search (FirecrawlClient part)
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_arun_side_effects
        result = await analogy_generator.generate_analogies(test_concept) # Default refinement_threshold is 7.0

    assert result["concept"] == test_concept
    assert len(result["generated_analogies"]) == 1
    gen_analogy = result["generated_analogies"][0]
    assert gen_analogy["title"] == mock_analogy["title"]
    assert "evaluation" in gen_analogy
    assert gen_analogy["evaluation"]["overall_score"] == 8.0
    assert "visual" in gen_analogy
    assert gen_analogy["visual"]["success"] is True
    assert len(gen_analogy["visual"]["assets"]) == 1
    assert len(result["existing_analogies"]) == 1
    assert result["existing_analogies"][0]["analogy"] == mock_existing_analogy["analogy"]
    assert result["stats"]["generated_count"] == 1
    assert result["stats"]["existing_count"] == 1
    assert result["stats"]["visual_assets_count"] == 1
    assert result["stats"]["average_score"] == 8.0

    # Expected LLM calls: generate_domain_analogies, search_existing_analogies (LLM), evaluate_analogy
    assert mock_arun.call_count == 3
    # Detailed checks for each call (optional but good for robustness)
    # Call 1: generate_domain_analogies
    assert mock_arun.call_args_list[0].kwargs['concept'] == test_concept
    # Call 2: search_existing_analogies
    assert mock_arun.call_args_list[1].kwargs['concept'] == test_concept
    assert "Blog" in mock_arun.call_args_list[1].kwargs['results']
    # Call 3: evaluate_analogy
    assert mock_arun.call_args_list[2].kwargs['concept'] == test_concept
    assert mock_analogy['title'] in mock_arun.call_args_list[2].kwargs['analogy']
    
    mock_source_validator.search_web.assert_called_once() # Called for existing analogies search
    mock_firecrawl_client.search_images.assert_called_once() # Called for visual

@pytest.mark.asyncio
async def test_generate_analogies_success_with_refinement(analogy_generator, mock_source_validator, mock_firecrawl_client):
    """Test full workflow: analogy generated, score < threshold, refined, existing found, visual found."""
    low_score_evaluation = {"overall_score": 6.0, "improvement_suggestions": ["Make it simpler"]}
    # Order of LLM calls in generate_analogies for this scenario:
    # 1. generate_domain_analogies
    # 2. search_existing_analogies (LLM part, after source_validator.search_web)
    # 3. evaluate_analogy (for the generated analogy)
    # 4. refine_analogy (because score is low)
    mock_chain_arun_side_effects = [
        json.dumps([mock_analogy]),           # For generate_domain_analogies
        json.dumps([mock_existing_analogy]),  # For search_existing_analogies (LLM part)
        json.dumps(low_score_evaluation),     # For evaluate_analogy (for mock_analogy)
        json.dumps(mock_refined_analogy)      # For refine_analogy
    ]

    # Mock search results for existing analogies (SourceValidator part)
    mock_source_validator.search_web.return_value = [{"title": "Blog", "url": "http://example.com/blog", "description": "Desc..."}]
    # Mock visual search (FirecrawlClient part)
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_arun_side_effects
        result = await analogy_generator.generate_analogies(test_concept, refinement_threshold=7.0)

    assert result["concept"] == test_concept
    assert len(result["generated_analogies"]) == 1
    gen_analogy = result["generated_analogies"][0]
    
    # The final list should contain the *refined* analogy
    assert gen_analogy["title"] == mock_refined_analogy["title"]
    
    # Refined analogy (from mock_refined_analogy) doesn't have 'evaluation' key from its prompt.
    # 'visual' key is added by the workflow.
    assert "evaluation" not in gen_analogy 
    assert "visual" in gen_analogy
    assert gen_analogy["visual"]["success"] is True
    assert len(gen_analogy["visual"]["assets"]) == 1
    
    assert len(result["existing_analogies"]) == 1
    assert result["existing_analogies"][0]["analogy"] == mock_existing_analogy["analogy"]
    
    assert result["stats"]["generated_count"] == 1
    assert result["stats"]["existing_count"] == 1
    assert result["stats"]["visual_assets_count"] == 1
    # Average score is 0.0 because the single refined analogy in final_analogies 
    # doesn't have an 'evaluation' field with an 'overall_score'.
    assert result["stats"]["average_score"] == 0.0 

    # Expected LLM calls: generate_domain_analogies, search_existing_analogies (LLM), evaluate_analogy, refine_analogy
    assert mock_arun.call_count == 4
    
    # Detailed checks for each call's arguments
    # Call 1: generate_domain_analogies
    assert mock_arun.call_args_list[0].kwargs['concept'] == test_concept
    assert mock_arun.call_args_list[0].kwargs['min_analogies'] == analogy_generator.min_analogies
    assert "Nature & Biology" in mock_arun.call_args_list[0].kwargs['domains']
    # Call 2: search_existing_analogies (LLM part)
    assert mock_arun.call_args_list[1].kwargs['concept'] == test_concept
    assert "Blog" in mock_arun.call_args_list[1].kwargs['results'] # Check that search_web result was passed
    # Call 3: evaluate_analogy
    assert mock_arun.call_args_list[2].kwargs['concept'] == test_concept
    assert mock_analogy['title'] in mock_arun.call_args_list[2].kwargs['analogy']
    # Call 4: refine_analogy
    assert mock_arun.call_args_list[3].kwargs['concept'] == test_concept
    assert mock_analogy['title'] in mock_arun.call_args_list[3].kwargs['analogy'] # Original analogy passed
    assert "Make it simpler" in mock_arun.call_args_list[3].kwargs['evaluation'] # Evaluation feedback passed

    mock_source_validator.search_web.assert_called_once() 
    mock_firecrawl_client.search_images.assert_called_once()

@pytest.mark.asyncio
async def test_generate_analogies_override_success(analogy_generator, mock_llm, mock_source_validator, mock_firecrawl_client):
    """Test full workflow using llm_override, ensuring LLMChain.arun is called as expected."""
    # mock_llm is analogy_generator.initial_llm via fixture setup. We'll assert it's not used.
    initial_llm_from_fixture = mock_llm 
    
    override_llm = MagicMock(spec=BaseChatModel)
    # No need to mock override_llm.arun if we are patching LLMChain.arun,
    # as AnalogyGenerator will use override_llm to construct the LLMChain internally.

    high_score_evaluation = {"overall_score": 9.0}
    # These are the JSON strings that the patched LLMChain.arun will return sequentially.
    # Order of LLM calls: generate_domain_analogies, search_existing_analogies (LLM), evaluate_analogy
    mock_chain_arun_side_effects = [
        json.dumps([mock_analogy]),          # For generate_domain_analogies
        json.dumps([mock_existing_analogy]), # For search_existing_analogies (LLM part)
        json.dumps(high_score_evaluation),   # For evaluate_analogy
    ]

    mock_source_validator.search_web.return_value = [{"title": "Blog Override", "url": "http://example.com/override", "description": "Desc..."}]
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_chain_arun:
        mock_chain_arun.side_effect = mock_chain_arun_side_effects
        
        # Call the main method with llm_override
        result = await analogy_generator.generate_analogies(test_concept, llm_override=override_llm)

    assert result["concept"] == test_concept
    assert len(result["generated_analogies"]) == 1
    gen_analogy = result["generated_analogies"][0]
    assert gen_analogy["title"] == mock_analogy["title"]
    assert "evaluation" in gen_analogy
    assert gen_analogy["evaluation"]["overall_score"] == 9.0
    assert "visual" in gen_analogy
    assert gen_analogy["visual"]["success"] is True
    
    assert len(result["existing_analogies"]) == 1
    assert result["existing_analogies"][0]["analogy"] == mock_existing_analogy["analogy"]
    
    assert result["stats"]["generated_count"] == 1
    assert result["stats"]["average_score"] == 9.0

    # Verify the patched LLMChain.arun was called
    assert mock_chain_arun.call_count == 3
    
    # Detailed checks for each call's arguments to the patched LLMChain.arun
    # Call 1: generate_domain_analogies
    assert mock_chain_arun.call_args_list[0].kwargs['concept'] == test_concept
    # Call 2: search_existing_analogies (LLM part)
    assert mock_chain_arun.call_args_list[1].kwargs['concept'] == test_concept
    assert "Blog Override" in mock_chain_arun.call_args_list[1].kwargs['results']
    # Call 3: evaluate_analogy
    assert mock_chain_arun.call_args_list[2].kwargs['concept'] == test_concept
    assert mock_analogy['title'] in mock_chain_arun.call_args_list[2].kwargs['analogy']

    # Verify the initial LLM (from fixture) was NOT used
    initial_llm_from_fixture.arun.assert_not_called() 
    # If initial_llm.agenerate was the underlying call for LLMChain, check that too.
    # However, AnalogyGenerator uses chain.arun, so initial_llm.arun.assert_not_called() is most direct
    # if we assume direct arun usage. Given we mock LLMChain.arun, this check ensures the
    # override path didn't somehow fall back to chains made with initial_llm and then try to call initial_llm.arun.
    # A more robust check would be to patch LLMChain.__init__ to see which LLM it was constructed with,
    # but that is more involved. For now, this covers the main point.

    mock_source_validator.search_web.assert_called_once()
    mock_firecrawl_client.search_images.assert_called_once()


@pytest.mark.asyncio
async def test_generate_analogies_llm_error_in_evaluate(analogy_generator, mock_source_validator, mock_firecrawl_client):
    """Test full workflow when LLMChain.arun fails during evaluation, ensuring graceful handling."""
    # Define the side effects for the patched LLMChain.arun
    # 1. generate_domain_analogies (Success)
    # 2. search_existing_analogies (LLM part - Success)
    # 3. evaluate_analogy (Failure)
    mock_chain_arun_side_effects = [
        json.dumps([mock_analogy]),                       # For generate_domain_analogies
        json.dumps([mock_existing_analogy]),          # For search_existing_analogies (LLM part)
        Exception("LLM evaluation chain failed")        # For evaluate_analogy (causes the error)
    ]

    mock_source_validator.search_web.return_value = [{"title": "Blog Error Case", "url": "http://example.com/error", "description": "Desc..."}]
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    with patch('agents.research.analogy_generator.LLMChain.arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.side_effect = mock_chain_arun_side_effects
        
        # The workflow should still complete, but the generated analogy might lack evaluation/refinement
        result = await analogy_generator.generate_analogies(test_concept)

    assert result["concept"] == test_concept
    assert len(result["generated_analogies"]) == 1 # Analogy was generated initially
    gen_analogy = result["generated_analogies"][0]
    assert gen_analogy["title"] == mock_analogy["title"] # The original analogy is kept
    
    # Evaluation failed, so 'evaluation' key should not be present or should indicate failure
    # Depending on implementation, it might be missing, or have an error field.
    # The current code in AnalogyGenerator means it will be missing if evaluate_analogy fails and error is caught.
    assert "evaluation" not in gen_analogy 
    
    # Visual generation still runs on the initially generated analogy
    assert "visual" in gen_analogy
    assert gen_analogy["visual"]["success"] is True 
    assert len(gen_analogy["visual"]["assets"]) == 1
    
    assert len(result["existing_analogies"]) == 1
    assert result["existing_analogies"][0]["analogy"] == mock_existing_analogy["analogy"]
    
    assert result["stats"]["generated_count"] == 1
    # No valid scores because evaluation failed for the one generated analogy
    assert result["stats"]["average_score"] == 0.0 

    # Expected LLMChain.arun calls: generate_domain_analogies, search_existing_analogies (LLM), evaluate_analogy (failed)
    assert mock_arun.call_count == 3
    
    # Detailed checks for arguments of successful calls
    # Call 1: generate_domain_analogies
    assert mock_arun.call_args_list[0].kwargs['concept'] == test_concept
    # Call 2: search_existing_analogies (LLM part)
    assert mock_arun.call_args_list[1].kwargs['concept'] == test_concept
    assert "Blog Error Case" in mock_arun.call_args_list[1].kwargs['results']
    # Call 3: evaluate_analogy (this one failed but was called)
    assert mock_arun.call_args_list[2].kwargs['concept'] == test_concept
    assert mock_analogy['title'] in mock_arun.call_args_list[2].kwargs['analogy']

    mock_source_validator.search_web.assert_called_once()
    mock_firecrawl_client.search_images.assert_called_once()


# Add more tests for variations like visual search failing, JSON errors in workflow, etc. 