import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
import json
import asyncio

# Langchain imports needed for mocks
from langchain_core.language_models.chat_models import BaseChatModel
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
async def test_generate_domain_analogies_success(analogy_generator, mock_llm):
    """Test successful generation of domain analogies."""
    expected_analogies = [mock_analogy]
    mock_response = create_mock_json_response(expected_analogies)
    mock_llm.arun.return_value = mock_response

    analogies = await analogy_generator.generate_domain_analogies(test_concept)

    assert analogies == expected_analogies
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert call_kwargs['concept'] == test_concept
    assert call_kwargs['min_analogies'] == 1
    assert "Nature & Biology" in call_kwargs['domains']

@pytest.mark.asyncio
async def test_generate_domain_analogies_override_success(analogy_generator, mock_llm):
    """Test successful domain analogy generation using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    expected_analogies = [mock_analogy]
    mock_response = create_mock_json_response(expected_analogies)
    override_llm.arun.return_value = mock_response

    analogies = await analogy_generator.generate_domain_analogies(test_concept, llm_override=override_llm)

    assert analogies == expected_analogies
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_generate_domain_analogies_llm_error(analogy_generator, mock_llm_failing):
    """Test domain analogy generation when the LLM fails."""
    generator = AnalogyGenerator(llm=mock_llm_failing, source_validator=MagicMock(), firecrawl_client=MagicMock())

    with pytest.raises(AnalogyGenerationError, match="Failed to generate analogies"):
        await generator.generate_domain_analogies(test_concept)
    mock_llm_failing.arun.assert_called_once()

@pytest.mark.asyncio
async def test_generate_domain_analogies_json_error(analogy_generator, mock_llm):
    """Test domain analogy generation with invalid JSON response."""
    mock_llm.arun.return_value = "This is not JSON"

    with pytest.raises(AnalogyGenerationError, match="Failed to parse analogies JSON"):
        await analogy_generator.generate_domain_analogies(test_concept)


# --- Tests for evaluate_analogy ---

@pytest.mark.asyncio
async def test_evaluate_analogy_success(analogy_generator, mock_llm):
    """Test successful evaluation of an analogy."""
    mock_response = create_mock_json_response(mock_evaluation)
    mock_llm.arun.return_value = mock_response

    evaluation = await analogy_generator.evaluate_analogy(test_concept, mock_analogy)

    assert evaluation == mock_evaluation
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert call_kwargs['concept'] == test_concept
    assert mock_analogy['title'] in call_kwargs['analogy']

@pytest.mark.asyncio
async def test_evaluate_analogy_override_success(analogy_generator, mock_llm):
    """Test successful analogy evaluation using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    mock_response = create_mock_json_response(mock_evaluation)
    override_llm.arun.return_value = mock_response

    evaluation = await analogy_generator.evaluate_analogy(test_concept, mock_analogy, llm_override=override_llm)

    assert evaluation == mock_evaluation
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_evaluate_analogy_llm_error(analogy_generator, mock_llm_failing):
    """Test analogy evaluation when the LLM fails."""
    generator = AnalogyGenerator(llm=mock_llm_failing, source_validator=MagicMock(), firecrawl_client=MagicMock())

    with pytest.raises(AnalogyGenerationError, match="Failed to evaluate analogy"):
        await generator.evaluate_analogy(test_concept, mock_analogy)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for refine_analogy ---

@pytest.mark.asyncio
async def test_refine_analogy_success(analogy_generator, mock_llm):
    """Test successful refinement of an analogy."""
    mock_response = create_mock_json_response(mock_refined_analogy)
    mock_llm.arun.return_value = mock_response

    refined = await analogy_generator.refine_analogy(test_concept, mock_analogy, mock_evaluation)

    assert refined == mock_refined_analogy
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert call_kwargs['concept'] == test_concept
    assert mock_analogy['title'] in call_kwargs['analogy']
    assert "Clarity: 8/10 - Clear" in call_kwargs['evaluation']

@pytest.mark.asyncio
async def test_refine_analogy_override_success(analogy_generator, mock_llm):
    """Test successful analogy refinement using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    mock_response = create_mock_json_response(mock_refined_analogy)
    override_llm.arun.return_value = mock_response

    refined = await analogy_generator.refine_analogy(test_concept, mock_analogy, mock_evaluation, llm_override=override_llm)

    assert refined == mock_refined_analogy
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_refine_analogy_llm_error(analogy_generator, mock_llm_failing):
    """Test analogy refinement when the LLM fails."""
    generator = AnalogyGenerator(llm=mock_llm_failing, source_validator=MagicMock(), firecrawl_client=MagicMock())

    with pytest.raises(AnalogyGenerationError, match="Failed to refine analogy"):
        await generator.refine_analogy(test_concept, mock_analogy, mock_evaluation)
    mock_llm_failing.arun.assert_called_once()


# --- Tests for search_existing_analogies ---

@pytest.mark.asyncio
async def test_search_existing_analogies_success(analogy_generator, mock_llm, mock_source_validator):
    """Test successful search for existing analogies."""
    mock_search_results = [{"title": "Blog Post", "url": "http://example.com/blog", "description": "Explains blockchain..."}]
    mock_source_validator.search_web.return_value = mock_search_results

    expected_existing = [mock_existing_analogy]
    mock_response = create_mock_json_response(expected_existing)
    mock_llm.arun.return_value = mock_response

    existing = await analogy_generator.search_existing_analogies(test_concept)

    assert existing == expected_existing
    mock_source_validator.search_web.assert_called_once_with(f"{test_concept} analogy metaphor explanation", count=5)
    mock_llm.arun.assert_called_once()
    call_args, call_kwargs = mock_llm.arun.call_args
    assert call_kwargs['concept'] == test_concept
    assert "Blog Post" in call_kwargs['results']

@pytest.mark.asyncio
async def test_search_existing_analogies_override_success(analogy_generator, mock_llm, mock_source_validator):
    """Test successful existing analogy search using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    mock_search_results = [{"title": "Override Blog", "url": "http://override.com/blog"}]
    mock_source_validator.search_web.return_value = mock_search_results
    expected_existing = [mock_existing_analogy]
    mock_response = create_mock_json_response(expected_existing)
    override_llm.arun.return_value = mock_response

    existing = await analogy_generator.search_existing_analogies(test_concept, llm_override=override_llm)

    assert existing == expected_existing
    mock_source_validator.search_web.assert_called_once()
    override_llm.arun.assert_called_once()
    initial_llm.arun.assert_not_called()

@pytest.mark.asyncio
async def test_search_existing_analogies_llm_error(analogy_generator, mock_llm_failing, mock_source_validator):
    """Test existing analogy search when the LLM fails."""
    mock_search_results = [{"title": "Blog Post", "url": "http://example.com/blog"}]
    mock_source_validator.search_web.return_value = mock_search_results
    generator = AnalogyGenerator(llm=mock_llm_failing, source_validator=mock_source_validator, firecrawl_client=MagicMock())

    with pytest.raises(AnalogyGenerationError, match="Failed to parse existing analogies JSON"): # Error occurs during JSON parsing
        await generator.search_existing_analogies(test_concept)
    mock_source_validator.search_web.assert_called_once()
    mock_llm_failing.arun.assert_called_once()

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
async def test_generate_analogies_success_no_refinement(analogy_generator, mock_llm, mock_source_validator, mock_firecrawl_client):
    """Test full workflow: analogy generated, score >= threshold, existing found, visual found."""
    high_score_evaluation = {"overall_score": 8.0}
    mock_llm_responses = [
        create_mock_json_response([mock_analogy]),          # generate_domain_analogies
        create_mock_json_response([mock_existing_analogy]), # search_existing_analogies (LLM call)
        create_mock_json_response(high_score_evaluation),   # evaluate_analogy
        # No refine_analogy call expected
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    # Mock search results for existing analogies
    mock_source_validator.search_web.side_effect = [
        [{"title": "Blog", "url": "http://example.com/blog"}], # For existing analogies
        [{"url": "http://brave.com/img.jpg"}] # For visual fallback (if needed)
    ]
    # Mock visual search
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    result = await analogy_generator.generate_analogies(test_concept)

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

    # Expected LLM calls: generate, search existing, evaluate
    assert mock_llm.arun.call_count == 3
    mock_source_validator.search_web.assert_called_once() # Only called for existing analogies search
    mock_firecrawl_client.search_images.assert_called_once() # Called for visual

@pytest.mark.asyncio
async def test_generate_analogies_success_with_refinement(analogy_generator, mock_llm, mock_source_validator, mock_firecrawl_client):
    """Test full workflow: analogy generated, score < threshold, refined, existing found, visual found."""
    low_score_evaluation = {"overall_score": 6.0, "improvement_suggestions": ["Make it simpler"]}
    mock_llm_responses = [
        create_mock_json_response([mock_analogy]),           # generate_domain_analogies
        create_mock_json_response([mock_existing_analogy]),  # search_existing_analogies (LLM call)
        create_mock_json_response(low_score_evaluation),    # evaluate_analogy
        create_mock_json_response(mock_refined_analogy)     # refine_analogy
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    mock_source_validator.search_web.return_value = [{"title": "Blog", "url": "http://example.com/blog"}] # For existing
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset] # For visual

    result = await analogy_generator.generate_analogies(test_concept, refinement_threshold=7.0)

    assert len(result["generated_analogies"]) == 1
    gen_analogy = result["generated_analogies"][0]
    # The final list should contain the *refined* analogy
    assert gen_analogy["title"] == mock_refined_analogy["title"]
    # Evaluation might not be directly attached to the refined one unless refine adds it back
    # assert "evaluation" not in gen_analogy # Or check if refine merges evaluation
    assert "visual" in gen_analogy and gen_analogy["visual"]["success"] is True
    assert len(result["existing_analogies"]) == 1
    assert result["stats"]["average_score"] == 0.0 # Average score recalculated based on *final* list, which lacks score here

    # Expected LLM calls: generate, search existing, evaluate, refine
    assert mock_llm.arun.call_count == 4

@pytest.mark.asyncio
async def test_generate_analogies_override_success(analogy_generator, mock_llm, mock_source_validator, mock_firecrawl_client):
    """Test full workflow using llm_override."""
    initial_llm = mock_llm
    override_llm = MagicMock(spec=BaseChatModel)
    override_llm.arun = AsyncMock()

    high_score_evaluation = {"overall_score": 9.0}
    override_responses = [
        create_mock_json_response([mock_analogy]),          # generate_domain_analogies
        create_mock_json_response([mock_existing_analogy]), # search_existing_analogies (LLM call)
        create_mock_json_response(high_score_evaluation),   # evaluate_analogy
    ]
    override_llm.arun.side_effect = override_responses

    mock_source_validator.search_web.return_value = [{"title": "Blog", "url": "http://example.com/blog"}]
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    result = await analogy_generator.generate_analogies(test_concept, llm_override=override_llm)

    assert len(result["generated_analogies"]) == 1
    assert result["generated_analogies"][0]["title"] == mock_analogy["title"]
    assert len(result["existing_analogies"]) == 1
    assert result["stats"]["average_score"] == 9.0

    # Verify override LLM was called, initial was not
    assert override_llm.arun.call_count == 3
    initial_llm.arun.assert_not_called()
    mock_source_validator.search_web.assert_called_once()
    mock_firecrawl_client.search_images.assert_called_once()


@pytest.mark.asyncio
async def test_generate_analogies_llm_error_in_evaluate(analogy_generator, mock_llm, mock_source_validator, mock_firecrawl_client):
    """Test full workflow when LLM fails during evaluation."""
    mock_llm_responses = [
        create_mock_json_response([mock_analogy]),          # generate_domain_analogies (Success)
        create_mock_json_response([mock_existing_analogy]), # search_existing_analogies (Success)
        AsyncMock(side_effect=Exception("LLM evaluation failed")), # evaluate_analogy (Failure)
    ]
    mock_llm.arun.side_effect = mock_llm_responses

    mock_source_validator.search_web.return_value = [{"title": "Blog", "url": "http://example.com/blog"}]
    mock_firecrawl_client.search_images.return_value = [mock_visual_asset]

    # The workflow should still complete, but the generated analogy might lack evaluation/refinement
    result = await analogy_generator.generate_analogies(test_concept)

    assert len(result["generated_analogies"]) == 1 # Analogy was generated initially
    gen_analogy = result["generated_analogies"][0]
    assert gen_analogy["title"] == mock_analogy["title"]
    assert "evaluation" not in gen_analogy # Evaluation failed
    assert "visual" in gen_analogy and gen_analogy["visual"]["success"] is True # Visual generation still runs
    assert len(result["existing_analogies"]) == 1
    assert result["stats"]["average_score"] == 0.0 # No valid scores

    # Expected LLM calls: generate, search existing, evaluate (failed)
    assert mock_llm.arun.call_count == 3


# Add more tests for variations like visual search failing, JSON errors in workflow, etc. 