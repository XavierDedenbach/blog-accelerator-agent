"""
Tests for the AnalogyGenerator component.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.research.analogy_generator import AnalogyGenerator


class TestAnalogyGenerator(unittest.TestCase):
    """Test cases for the AnalogyGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the SourceValidator and FirecrawlClient
        self.mock_source_validator = MagicMock()
        self.mock_firecrawl_client = MagicMock()
        
        # Create the generator with mocked dependencies
        self.generator = AnalogyGenerator(
            openai_api_key="mock_key",
            source_validator=self.mock_source_validator,
            firecrawl_client=self.mock_firecrawl_client
        )
        
        # Mock the LLM chains
        self.generator.llm = MagicMock()
    
    def test_initialization(self):
        """Test that the generator initializes properly."""
        self.assertEqual(self.generator.min_analogies, 3)
        self.assertEqual(len(self.generator.domains), 10)
        self.assertIsNotNone(self.generator.generate_analogies_prompt)
        self.assertIsNotNone(self.generator.evaluate_analogy_prompt)
        self.assertIsNotNone(self.generator.refine_analogy_prompt)
        self.assertIsNotNone(self.generator.search_existing_analogies_prompt)


if __name__ == '__main__':
    unittest.main() 