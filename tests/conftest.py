"""
Configuration for pytest fixtures shared across test modules.
"""

import os
import pytest
from unittest.mock import patch

@pytest.fixture(scope="session", autouse=True)
def mock_env_variables():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "MONGODB_URI": "mongodb://testmongo:27017",
        "OPENAI_API_KEY": "test-openai-key",
        "GROQ_API_KEY": "test-groq-key",
        "BRAVE_API_KEY": "test-brave-key",
        "FIRECRAWL_SERVER": "http://test-firecrawl:4000",
        "OPIK_SERVER": "http://test-opik:7000"
    }):
        yield

@pytest.fixture
def sample_blog_data():
    """Return sample blog data for testing."""
    return {
        "title": "why-microgrids-will-replace-utilities",
        "current_version": 3,
        "asset_folder": "uploads/why-microgrids-v3/",
        "versions": [
            {
                "version": 3,
                "file_path": "review/why-microgrids_v3.md",
                "timestamp": "2024-05-02T20:10:00Z",
                "review_status": {
                    "factual_review": {"complete": False, "result_file": None},
                    "style_review": {"complete": False, "result_file": None},
                    "grammar_review": {"complete": False, "result_file": None},
                    "final_release": {"complete": False}
                },
                "readiness_score": None
            }
        ]
    }

@pytest.fixture
def sample_yaml_data():
    """Return sample YAML data for testing."""
    return {
        "blog_title": "why-microgrids-will-replace-utilities",
        "current_version": 3,
        "review_pipeline": {
            "factual_review": {
                "complete": False,
                "completed_by": None,
                "result_file": None,
                "timestamp": None
            },
            "style_review": {
                "complete": False,
                "completed_by": None,
                "result_file": None,
                "timestamp": None
            },
            "grammar_review": {
                "complete": False,
                "completed_by": None,
                "result_file": None,
                "timestamp": None
            }
        },
        "final_release": {
            "complete": False,
            "released_by": None,
            "timestamp": None
        }
    } 