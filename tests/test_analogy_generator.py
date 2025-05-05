"""
Tests for the AnalogyGenerator component.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock
import json
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.research.analogy_generator import AnalogyGenerator, AnalogyGenerationError

# Use IsolatedAsyncioTestCase for async tests
class TestAnalogyGenerator(unittest.IsolatedAsyncioTestCase):
    """Test cases for the AnalogyGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the SourceValidator and FirecrawlClient
        self.mock_source_validator = MagicMock()
        # Configure the mock search_web to be an async function returning a list
        self.mock_source_validator.search_web = AsyncMock(return_value=[])

        self.mock_firecrawl_client = MagicMock()
        # Configure the mock search_images to be an async function returning a list
        self.mock_firecrawl_client.search_images = AsyncMock(return_value=[])
        
        # Create the generator with mocked dependencies
        self.generator = AnalogyGenerator(
            openai_api_key="mock_key",
            source_validator=self.mock_source_validator,
            firecrawl_client=self.mock_firecrawl_client
        )
        
        # Mock the LLM chains (can use a simple MagicMock for basic tests)
        self.generator.llm = MagicMock()

    def test_initialization(self):
        """Test that the generator initializes properly."""
        self.assertEqual(self.generator.min_analogies, 3)
        self.assertEqual(len(self.generator.domains), 10)
        self.assertIsNotNone(self.generator.generate_analogies_prompt)
        self.assertIsNotNone(self.generator.evaluate_analogy_prompt)
        self.assertIsNotNone(self.generator.refine_analogy_prompt)
        self.assertIsNotNone(self.generator.search_existing_analogies_prompt)
        
    async def test_generate_analogies_flow(self):
        """Test the main generate_analogies method flow."""
        test_concept = "Quantum Computing"

        # Mock the internal async methods
        self.generator.generate_domain_analogies = AsyncMock(return_value=[
            {"title": "Analogy 1", "domain": "Physics", "description": "...", "mapping": "..."},
            {"title": "Analogy 2", "domain": "Nature", "description": "...", "mapping": "..."},
            {"title": "Analogy 3", "domain": "Computing", "description": "...", "mapping": "..."}
        ])
        self.generator.search_existing_analogies = AsyncMock(return_value=[
            {"source": "url", "analogy": "Existing 1", "mapping": "..."}
        ])
        # Mock evaluate_analogy to return valid scores (including one below threshold)
        self.generator.evaluate_analogy = AsyncMock(side_effect=[
            {"clarity": {"score": 8}, "accuracy": {"score": 7}, "memorability": {"score": 9}, "relatability": {"score": 8}, "educational_value": {"score": 8}, "overall_score": 8.0, "improvement_suggestions": []},
            {"clarity": {"score": 5}, "accuracy": {"score": 6}, "memorability": {"score": 5}, "relatability": {"score": 6}, "educational_value": {"score": 5}, "overall_score": 5.4, "improvement_suggestions": ["Refine X"]},
            {"clarity": {"score": 9}, "accuracy": {"score": 9}, "memorability": {"score": 9}, "relatability": {"score": 9}, "educational_value": {"score": 9}, "overall_score": 9.0, "improvement_suggestions": []},
        ])
        # Mock refine_analogy
        self.generator.refine_analogy = AsyncMock(return_value=
            {"title": "Refined Analogy 2", "domain": "Nature", "description": "Refined...", "mapping": "Refined..."}
        )
        # Mock generate_visual_representation
        self.generator.generate_visual_representation = AsyncMock(return_value={
            "success": True, "assets": [{"url": "visual_url"}], "query": "...", "visual_description": "..."
        })

        try:
            # Call the main method
            result = await self.generator.generate_analogies(test_concept, refinement_threshold=7.0)
            
            # Basic assertions on the result structure
            self.assertIsInstance(result, dict)
            self.assertIn("concept", result)
            self.assertEqual(result["concept"], test_concept)
            self.assertIn("generated_analogies", result)
            self.assertIsInstance(result["generated_analogies"], list)
            self.assertIn("existing_analogies", result)
            self.assertIsInstance(result["existing_analogies"], list)
            self.assertIn("stats", result)
            self.assertIsInstance(result["stats"], dict)
            
            # Check that the mocks were called
            self.generator.generate_domain_analogies.assert_called_once_with(test_concept)
            self.generator.search_existing_analogies.assert_called_once_with(test_concept)
            self.assertEqual(self.generator.evaluate_analogy.call_count, 3) # Called for each generated analogy
            self.generator.refine_analogy.assert_called_once() # Called for the one below threshold
            self.assertEqual(self.generator.generate_visual_representation.call_count, 3) # Called for each refined/kept analogy

        except TypeError as e:
            self.fail(f"generate_analogies raised unexpected TypeError: {e}")
        except Exception as e:
            self.fail(f"generate_analogies raised unexpected exception: {e}")


if __name__ == '__main__':
    unittest.main() 