"""
Tests for the API endpoints.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "version" in response.json()
    assert "service" in response.json()


@pytest.fixture
def mock_researcher_agent():
    """Mock the ResearcherAgent class."""
    with patch("api.endpoints.process.ResearcherAgent") as mock_agent_class:
        # Setup mock returns
        mock_agent = MagicMock()
        mock_agent.process_blog.return_value = {
            "status": "success",
            "blog_title": "test-blog",
            "version": 1,
            "readiness_score": 75,
            "yaml_path": "/path/to/yaml"
        }
        mock_agent.db_client.get_blog_status.return_value = {
            "title": "test-blog",
            "current_version": 1,
            "versions": [
                {
                    "version": 1,
                    "readiness_score": 75
                }
            ]
        }
        mock_agent_class.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def mock_reviewer_agent():
    """Mock the ReviewerAgent class."""
    with patch("api.endpoints.review.ReviewerAgent") as mock_agent_class:
        # Setup mock returns
        mock_agent = MagicMock()
        mock_agent.process_review_stage.return_value = {
            "status": "success",
            "blog_title": "test-blog",
            "version": 1,
            "stage": "factual_review",
            "report_filename": "test-blog_factual_review_v1.md",
            "yaml_updated": True,
            "current_stage": "style_review"
        }
        mock_agent.mark_blog_as_released.return_value = {
            "status": "success",
            "blog_title": "test-blog",
            "version": 1,
            "released": True,
            "yaml_path": "/path/to/yaml"
        }
        mock_agent_class.return_value = mock_agent
        yield mock_agent


@pytest.fixture
def mock_yaml_functions():
    """Mock YAML functions."""
    with patch("api.endpoints.review.load_yaml") as mock_load:
        with patch("api.endpoints.review.validate_yaml_structure") as mock_validate:
            with patch("api.endpoints.review.validate_stage_transition") as mock_transition:
                with patch("api.endpoints.review.get_review_status") as mock_status:
                    # Setup mock returns
                    mock_load.return_value = {
                        "blog_title": "test-blog",
                        "current_version": 1
                    }
                    mock_status.return_value = {
                        "blog_title": "test-blog",
                        "version": 1,
                        "current_stage": "style_review",
                        "all_stages_complete": False,
                        "released": False
                    }
                    yield (mock_load, mock_validate, mock_transition, mock_status)


@pytest.fixture
def mock_file_ops():
    """Mock file operations."""
    with patch("os.path.exists") as mock_exists:
        with patch("shutil.copyfileobj") as mock_copy:
            with patch("builtins.open", create=True) as mock_open:
                # Setup mock returns
                mock_exists.return_value = True
                mock_open.return_value = MagicMock()
                yield (mock_exists, mock_copy, mock_open)


def test_upload_blog_endpoint(client, mock_researcher_agent, mock_file_ops):
    """Test the blog upload endpoint."""
    # Create a mock file for upload
    with open("test_file.md", "w") as f:
        f.write("# Test Blog\n\nThis is a test blog post.")
    
    try:
        # Test with a markdown file
        with open("test_file.md", "rb") as f:
            response = client.post(
                "/process/upload",
                files={"file": ("test-blog.md", f, "text/markdown")}
            )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["blog_title"] == "test-blog"
        assert response.json()["version"] == 1
        assert response.json()["readiness_score"] == 75
        
        # Verify mock was called
        mock_researcher_agent.process_blog.assert_called_once()
    
    finally:
        # Clean up test file
        if os.path.exists("test_file.md"):
            os.remove("test_file.md")


def test_get_blog_status_endpoint(client, mock_researcher_agent):
    """Test the blog status endpoint."""
    response = client.get("/process/status/test-blog")
    
    assert response.status_code == 200
    assert response.json()["title"] == "test-blog"
    assert response.json()["current_version"] == 1
    
    # Verify mock was called
    mock_researcher_agent.db_client.get_blog_status.assert_called_once_with("test-blog")


def test_process_review_stage_endpoint(client, mock_reviewer_agent, mock_yaml_functions, mock_file_ops):
    """Test the review stage processing endpoint."""
    review_request = {
        "blog_title": "test-blog",
        "version": 1,
        "stage": "factual_review",
        "async_process": False
    }
    
    response = client.post("/review/stage", json=review_request)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["blog_title"] == "test-blog"
    assert response.json()["version"] == 1
    assert response.json()["stage"] == "factual_review"
    assert response.json()["next_stage"] == "style_review"
    
    # Verify mock was called
    mock_reviewer_agent.process_review_stage.assert_called_once()


def test_mark_blog_released_endpoint(client, mock_reviewer_agent, mock_yaml_functions, mock_file_ops):
    """Test the blog release endpoint."""
    # Change the mock status to indicate all stages are complete
    mock_yaml_functions[3].return_value = {
        "blog_title": "test-blog",
        "version": 1,
        "current_stage": None,
        "all_stages_complete": True,
        "released": False
    }
    
    release_request = {
        "blog_title": "test-blog",
        "version": 1
    }
    
    response = client.post("/review/release", json=release_request)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["blog_title"] == "test-blog"
    assert response.json()["version"] == 1
    
    # Verify mock was called
    mock_reviewer_agent.mark_blog_as_released.assert_called_once()


def test_get_review_status_endpoint(client, mock_yaml_functions, mock_file_ops):
    """Test the review status endpoint."""
    response = client.get("/review/status/test-blog")
    
    assert response.status_code == 200
    assert response.json()["blog_title"] == "test-blog"
    assert response.json()["version"] == 1
    assert response.json()["current_stage"] == "style_review"
    assert response.json()["all_stages_complete"] is False
    assert response.json()["released"] is False
    
    # Verify mock was called
    mock_yaml_functions[3].assert_called_once() 