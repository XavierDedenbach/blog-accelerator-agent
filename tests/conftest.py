"""
Configuration for pytest fixtures shared across test modules.

This file defines fixtures that can be used by any test in the 'tests' directory
and its subdirectories. Fixtures defined here can set up and tear down resources,
provide mock objects, or load test data.

Key Fixtures:
- mock_env_variables: Mocks common environment variables with test/placeholder values.
  It is NOT autouse; tests needing these mocks must request it explicitly.
- load_env_vars_and_debug: (Autouse, Session-scoped) Attempts to load real environment
  variables from a .env file at the project root. This allows integration tests
  to use real credentials if present, while unit tests can rely on mock_env_variables.
- sample_blog_data: Provides sample blog data for testing.
- sample_yaml_data: Provides sample YAML data for testing.
"""

import os
import pytest
from unittest.mock import patch
from dotenv import load_dotenv, dotenv_values

@pytest.fixture(scope="session") # Not autouse; tests must request it if they need these mocks.
def mock_env_variables():
    """Mock common environment variables with placeholder/test values."""
    mock_values = {
        "MONGODB_URI": "mongodb://testmongo:27017",
        "OPENAI_API_KEY": "test-openai-key",
        "GROQ_API_KEY": "test-groq-key",
        "BRAVE_API_KEY": "test-brave-key",
        "FIRECRAWL_SERVER": "http://test-firecrawl:4000",
        "OPIK_SERVER": "http://test-opik:7000"
    }
    # print(f"\n--- conftest.py: Applying mock_env_variables: {mock_values} ---")
    with patch.dict(os.environ, mock_values, clear=True): # clear=True ensures a clean environment for the test.
        yield
    # print("--- conftest.py: mock_env_variables finished ---\n")

@pytest.fixture
def sample_blog_data():
    """Provides a sample blog data structure for testing purposes."""
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
    """Provides a sample YAML data structure for testing purposes."""
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

@pytest.fixture(scope="session", autouse=True)
def load_env_vars_and_debug():
    """
    (Autouse, Session-Scoped) Loads environment variables from .env file at project root.
    This runs once per session before any tests. It attempts to load real credentials
    for integration tests. `override=True` ensures .env values take precedence.
    """
    # print("\n--- conftest.py: Attempting to load .env ---")
    project_root_for_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # tests/ -> project_root
    dotenv_path_to_load = os.path.join(project_root_for_env, '.env')
    
    # print(f"DEBUG (conftest.py): Path for .env: {dotenv_path_to_load}")
    
    if not os.path.exists(dotenv_path_to_load):
        # print(f"DEBUG (conftest.py): .env file NOT FOUND at {dotenv_path_to_load}. Integration tests requiring real keys might be skipped or fail.")
        return

    # print(f"DEBUG (conftest.py): .env file found at {dotenv_path_to_load}")
    
    # For debugging: shows what python-dotenv reads before trying to load into os.environ
    # env_values_read = dotenv_values(dotenv_path_to_load)
    # print(f"DEBUG (conftest.py): Values read by dotenv_values(): {env_values_read}")
    
    # Load into os.environ. override=True ensures these values win over pre-existing shell env vars.
    found_dotenv = load_dotenv(dotenv_path=dotenv_path_to_load, override=True)
    # print(f"DEBUG (conftest.py): load_dotenv() successful: {found_dotenv}")
    
    # print("--- conftest.py: .env loading attempt finished ---\n")

# Add any other session-wide fixtures or hooks here if needed. 