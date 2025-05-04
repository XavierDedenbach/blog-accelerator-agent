"""
Tests for the Visual Asset Collector component.
"""

import pytest
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
from agents.research.visual_asset_collector import VisualAssetCollector, VisualAssetCollectionError


@pytest.fixture
def mock_firecrawl_client():
    """Mock FirecrawlClient for testing."""
    mock_client = MagicMock()
    
    # Mock search_images method
    mock_client.search_images = MagicMock(return_value=[
        {
            "url": "https://example.com/image1.jpg",
            "title": "Test Image 1",
            "source_url": "https://example.com/page1",
            "width": 800,
            "height": 600,
            "format": "jpg"
        },
        {
            "url": "https://example.com/image2.png",
            "title": "Test Image 2",
            "source_url": "https://example.com/page2",
            "width": 1024,
            "height": 768,
            "format": "png"
        }
    ])
    
    # Mock download_image method
    mock_client.download_image = MagicMock(return_value={
        "url": "https://example.com/image1.jpg",
        "file_path": "/tmp/test/image1.jpg",
        "file_size": 12345,
        "width": 800,
        "height": 600,
        "format": "jpg"
    })
    
    return mock_client


@pytest.fixture
def mock_source_validator():
    """Mock SourceValidator for testing."""
    mock_validator = MagicMock()
    return mock_validator


@pytest.fixture
def visual_asset_collector(mock_firecrawl_client, mock_source_validator):
    """VisualAssetCollector instance with mocked dependencies for testing."""
    with patch('agents.research.visual_asset_collector.ChatOpenAI') as mock_openai, \
         patch('agents.research.visual_asset_collector.LLMChain') as mock_chain:
        
        # Create a more sophisticated mock for LLMChain
        mock_chain_instance = MagicMock()
        mock_chain_instance.arun = AsyncMock()
        mock_chain.return_value = mock_chain_instance
        
        collector = VisualAssetCollector(
            openai_api_key="test_key",
            firecrawl_client=mock_firecrawl_client,
            source_validator=mock_source_validator
        )
        
        return collector


@pytest.fixture
def mock_solution_data():
    """Sample solution data for testing."""
    return {
        "solution": "A machine learning approach to optimize supply chains",
        "pro_arguments": [
            {
                "name": "Improved Efficiency",
                "description": "ML algorithms can identify patterns and optimize routing"
            },
            {
                "name": "Cost Reduction",
                "description": "Optimized inventory management reduces carrying costs"
            }
        ],
        "counter_arguments": [
            {
                "name": "Implementation Complexity",
                "description": "Requires significant data infrastructure and expertise"
            }
        ],
        "metrics": [
            {
                "name": "Delivery Time Reduction",
                "description": "Measures percentage improvement in delivery times"
            }
        ]
    }


@pytest.fixture
def mock_paradigm_data():
    """Sample paradigm data for testing."""
    return {
        "historical_paradigms": [
            {
                "name": "Manual Logistics Management",
                "description": "Pre-digital era of paper-based logistics"
            },
            {
                "name": "Digital Transformation Era",
                "description": "Initial computerization of supply chains"
            }
        ],
        "transitions": [
            {
                "from_paradigm": "Manual Logistics Management",
                "to_paradigm": "Digital Transformation Era",
                "trigger_factors": "Rise of enterprise computing systems"
            }
        ],
        "future_paradigms": [
            {
                "name": "Autonomous Supply Chain Systems",
                "description": "Self-optimizing systems with minimal human intervention"
            }
        ]
    }


@pytest.mark.asyncio
async def test_collect_solution_visuals(visual_asset_collector, mock_solution_data):
    """Test collecting solution visuals."""
    # Replace the internal methods with mocks
    visual_asset_collector._collect_category_visuals = AsyncMock(return_value=[
        {
            "url": "https://example.com/diagram1.jpg",
            "title": "Supply Chain Optimization Diagram",
            "category": "diagrams",
            "relevance_score": 9,
            "quality_score": 8,
            "caption": "ML-driven supply chain optimization workflow"
        },
        {
            "url": "https://example.com/chart1.png",
            "title": "Efficiency Gains Chart",
            "category": "charts",
            "relevance_score": 8,
            "quality_score": 7,
            "caption": "Performance metrics after ML implementation"
        }
    ])
    
    # Call the method
    result = await visual_asset_collector.collect_solution_visuals(
        topic="Machine Learning in Supply Chain Management",
        solution_data=mock_solution_data,
        count=10
    )
    
    # Check the structure of the result
    assert "topic" in result
    assert "collection_type" in result
    assert "visuals" in result
    assert "stats" in result
    
    # Check the stats
    assert "total_visuals_collected" in result["stats"]
    assert "collection_duration_seconds" in result["stats"]
    assert "visuals_by_category" in result["stats"]
    assert "timestamp" in result["stats"]
    
    # Check if _collect_category_visuals was called for each category
    assert visual_asset_collector._collect_category_visuals.call_count == len(visual_asset_collector.categories)


@pytest.mark.asyncio
async def test_collect_paradigm_visuals(visual_asset_collector, mock_paradigm_data):
    """Test collecting paradigm visuals."""
    # Replace the internal methods with mocks
    visual_asset_collector._collect_category_visuals = AsyncMock(return_value=[
        {
            "url": "https://example.com/timeline.jpg",
            "title": "Supply Chain Evolution Timeline",
            "category": "diagrams",
            "relevance_score": 9,
            "quality_score": 8,
            "caption": "Historical evolution of supply chain management"
        },
        {
            "url": "https://example.com/future.png",
            "title": "Future Supply Chain Concept",
            "category": "illustrations",
            "relevance_score": 7,
            "quality_score": 9,
            "caption": "Vision of autonomous supply chain systems"
        }
    ])
    
    # Call the method
    result = await visual_asset_collector.collect_paradigm_visuals(
        topic="Evolution of Supply Chain Management",
        paradigm_data=mock_paradigm_data,
        count=6
    )
    
    # Check the structure of the result
    assert "topic" in result
    assert "collection_type" in result
    assert "visuals" in result
    assert "stats" in result
    
    # Check the stats
    assert "total_visuals_collected" in result["stats"]
    assert "collection_duration_seconds" in result["stats"]
    assert "visuals_by_category" in result["stats"]
    assert "timestamp" in result["stats"]
    
    # Check if _collect_category_visuals was called for each category
    assert visual_asset_collector._collect_category_visuals.call_count == len(visual_asset_collector.categories)


@pytest.mark.asyncio
async def test_generate_search_queries(visual_asset_collector):
    """Test generating search queries."""
    # Create a patch specifically for this test to override the existing chain
    with patch.object(visual_asset_collector, '_generate_search_queries', wraps=visual_asset_collector._generate_search_queries) as wrapped_method:
        # We need to patch the internal LLMChain creation and execution
        with patch('agents.research.visual_asset_collector.LLMChain') as mock_chain:
            mock_instance = MagicMock()
            mock_instance.arun = AsyncMock(return_value='["query 1", "query 2", "query 3"]')
            mock_chain.return_value = mock_instance
            
            # Call the method (this will use our patched LLMChain)
            queries = await visual_asset_collector._generate_search_queries(
                topic="Test Topic",
                collection_type="solutions",
                category="diagrams",
                research_data={"key": "value"}
            )
            
            # Check the results - the test_generate_search_queries method should have received the mocked response
            assert len(queries) == 3
            assert "query 1" in queries
            assert "query 2" in queries
            assert "query 3" in queries


@pytest.mark.asyncio
async def test_filter_and_categorize_visuals(visual_asset_collector):
    """Test filtering and categorizing visual assets."""
    # Sample visual assets
    visual_assets = [
        {
            "url": "https://example.com/image1.jpg",
            "title": "High Quality Relevant Image",
            "source_url": "https://example.com/page1",
            "width": 800,
            "height": 600
        },
        {
            "url": "https://example.com/image2.jpg",
            "title": "Low Quality Image",
            "source_url": "https://example.com/page2",
            "width": 400,
            "height": 300
        }
    ]
    
    # Create a patch specifically for this test
    with patch.object(visual_asset_collector, '_filter_and_categorize_visuals', wraps=visual_asset_collector._filter_and_categorize_visuals) as wrapped_method:
        # We need to patch the internal LLMChain creation and execution
        with patch('agents.research.visual_asset_collector.LLMChain') as mock_chain:
            mock_instance = MagicMock()
            mock_instance.arun = AsyncMock(return_value=json.dumps([
                {
                    "keep_asset": True,
                    "relevance_score": 9,
                    "quality_score": 8,
                    "subcategory": "workflow",
                    "caption": "ML-driven workflow optimization",
                    "key_points": ["Efficiency", "Automation"]
                },
                {
                    "keep_asset": False,
                    "relevance_score": 3,
                    "quality_score": 2,
                    "subcategory": "general",
                    "caption": "Low quality image",
                    "key_points": [],
                    "rejection_reason": "Poor quality and low relevance"
                }
            ]))
            mock_chain.return_value = mock_instance
            
            # Call the method
            result = await visual_asset_collector._filter_and_categorize_visuals(
                topic="Test Topic",
                collection_type="solutions",
                category="diagrams",
                visual_assets=visual_assets,
                research_data={"key": "value"}
            )
            
            # Check the results
            assert len(result) == 2
            assert result[0]["keep_asset"] is True
            assert result[0]["relevance_score"] == 9
            assert result[0]["quality_score"] == 8
            assert result[0]["url"] == "https://example.com/image1.jpg"
            
            assert result[1]["keep_asset"] is False
            assert result[1]["rejection_reason"] == "Poor quality and low relevance"
            assert result[1]["url"] == "https://example.com/image2.jpg"


@pytest.mark.asyncio
async def test_generate_visual_metadata(visual_asset_collector):
    """Test generating metadata for a visual asset."""
    # Sample visual asset
    visual_asset = {
        "url": "https://example.com/image1.jpg",
        "title": "Supply Chain Optimization Diagram",
        "source_url": "https://example.com/page1",
        "width": 800,
        "height": 600,
        "caption": "Initial caption",
        "subcategory": "workflow"
    }
    
    # Create a patch specifically for this test
    with patch.object(visual_asset_collector, '_generate_visual_metadata', wraps=visual_asset_collector._generate_visual_metadata) as wrapped_method:
        # We need to patch the internal LLMChain creation and execution
        with patch('agents.research.visual_asset_collector.LLMChain') as mock_chain:
            mock_instance = MagicMock()
            mock_instance.arun = AsyncMock(return_value=json.dumps({
                "caption": "ML-driven supply chain optimization workflow",
                "extended_description": "This diagram illustrates how machine learning algorithms can optimize various stages of the supply chain process, from demand forecasting to inventory management.",
                "key_concepts": ["Machine Learning", "Supply Chain Optimization", "Workflow Automation"],
                "technical_details": "Incorporates neural networks for demand prediction and reinforcement learning for routing optimization",
                "research_connections": "Directly supports the 'Improved Efficiency' pro argument by visualizing the optimization process",
                "suggested_placement": "After introducing the ML optimization concept, before discussing implementation details",
                "audience_benefit": "Helps technical decision makers understand the end-to-end process"
            }))
            mock_chain.return_value = mock_instance
            
            # Call the method
            result = await visual_asset_collector._generate_visual_metadata(
                topic="Machine Learning in Supply Chain Management",
                collection_type="solutions",
                category="diagrams",
                visual_asset=visual_asset,
                research_data={"key": "value"}
            )
            
            # Check the results
            assert result["caption"] == "ML-driven supply chain optimization workflow"
            assert "This diagram illustrates" in result["extended_description"]
            assert "Machine Learning" in result["key_concepts"]
            assert "neural networks" in result["technical_details"]
            assert "Improved Efficiency" in result["research_connections"]
            assert "After introducing" in result["suggested_placement"]
            assert "technical decision makers" in result["audience_benefit"]


@pytest.mark.asyncio
async def test_collect_category_visuals(visual_asset_collector):
    """Test collecting visuals for a specific category."""
    # Mock internal methods with manually created mocks
    visual_asset_collector._generate_search_queries = AsyncMock(return_value=["query 1", "query 2"])
    visual_asset_collector._filter_and_categorize_visuals = AsyncMock(return_value=[
        {
            "url": "https://example.com/image1.jpg",
            "title": "Test Image 1",
            "keep_asset": True,
            "relevance_score": 9,
            "quality_score": 8
        },
        {
            "url": "https://example.com/image2.jpg",
            "title": "Test Image 2",
            "keep_asset": False,
            "relevance_score": 3,
            "quality_score": 2
        }
    ])
    visual_asset_collector._generate_visual_metadata = AsyncMock(return_value={
        "caption": "Enhanced caption",
        "extended_description": "Detailed description",
        "key_concepts": ["Concept 1", "Concept 2"]
    })
    
    # Call the method
    result = await visual_asset_collector._collect_category_visuals(
        topic="Test Topic",
        collection_type="solutions",
        category="diagrams",
        research_data={"key": "value"},
        target_count=3
    )
    
    # Check that the method calls were made correctly
    visual_asset_collector._generate_search_queries.assert_called_once()
    visual_asset_collector._filter_and_categorize_visuals.assert_called_once()
    
    # Check that metadata was generated only for assets to keep
    assert visual_asset_collector._generate_visual_metadata.call_count == 1
    
    # Check the results
    assert len(result) == 1  # Only one asset was kept
    assert result[0]["url"] == "https://example.com/image1.jpg"
    assert result[0]["caption"] == "Enhanced caption"
    assert "extended_description" in result[0]
    assert "key_concepts" in result[0]


@pytest.mark.asyncio
async def test_analyze_visual_assets(visual_asset_collector, mock_solution_data, mock_paradigm_data):
    """Test the complete visual asset analysis process."""
    # Mock the collection methods
    solution_visuals = {
        "topic": "Test Topic",
        "collection_type": "solutions",
        "visuals": [
            {
                "url": "https://example.com/solution1.jpg",
                "category": "diagrams",
                "caption": "Solution diagram"
            },
            {
                "url": "https://example.com/solution2.jpg",
                "category": "charts",
                "caption": "Solution chart"
            }
        ],
        "stats": {
            "total_visuals_collected": 2,
            "collection_duration_seconds": 1.5
        }
    }
    
    paradigm_visuals = {
        "topic": "Test Topic",
        "collection_type": "paradigms",
        "visuals": [
            {
                "url": "https://example.com/paradigm1.jpg",
                "category": "diagrams",
                "caption": "Paradigm diagram"
            }
        ],
        "stats": {
            "total_visuals_collected": 1,
            "collection_duration_seconds": 0.8
        }
    }
    
    visual_asset_collector.collect_solution_visuals = AsyncMock(return_value=solution_visuals)
    visual_asset_collector.collect_paradigm_visuals = AsyncMock(return_value=paradigm_visuals)
    
    # Call the method
    result = await visual_asset_collector.analyze_visual_assets(
        topic="Test Topic",
        solution_data=mock_solution_data,
        paradigm_data=mock_paradigm_data
    )
    
    # Check the structure of the result
    assert "topic" in result
    assert "solution_visuals" in result
    assert "paradigm_visuals" in result
    assert "stats" in result
    
    # Check the stats
    assert result["stats"]["total_solution_visuals"] == 2
    assert result["stats"]["total_paradigm_visuals"] == 1
    assert result["stats"]["total_visuals"] == 3
    assert "visuals_by_category" in result["stats"]
    assert "analysis_duration_seconds" in result["stats"]
    assert "timestamp" in result["stats"]
    
    # Check if collection methods were called with correct parameters
    visual_asset_collector.collect_solution_visuals.assert_called_once_with(
        topic="Test Topic",
        solution_data=mock_solution_data
    )
    
    visual_asset_collector.collect_paradigm_visuals.assert_called_once_with(
        topic="Test Topic",
        paradigm_data=mock_paradigm_data
    )


@pytest.mark.asyncio
async def test_error_handling(visual_asset_collector, mock_solution_data, mock_paradigm_data):
    """Test error handling in visual asset collector."""
    # Mock collect_solution_visuals to raise an exception
    visual_asset_collector.collect_solution_visuals = AsyncMock(side_effect=Exception("Test error"))
    
    # Test analyze_visual_assets error handling
    with pytest.raises(VisualAssetCollectionError):
        await visual_asset_collector.analyze_visual_assets(
            topic="Test Topic",
            solution_data=mock_solution_data,
            paradigm_data=mock_paradigm_data
        ) 