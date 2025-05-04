"""
Tests for the YAML validation guards in yaml_guard.py.
"""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from agents.utilities.yaml_guard import (
    load_yaml, save_yaml, validate_yaml_structure, validate_stage_transition,
    mark_stage_complete, mark_blog_released, get_current_stage,
    create_tracker_yaml, get_review_status, YamlGuardError, ReviewStageError,
    REVIEW_STAGES
)


@pytest.fixture
def temp_yaml_file():
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        yield f.name
    # Clean up after test
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def valid_yaml_data(sample_yaml_data):
    """Get valid YAML data from the conftest fixture."""
    return sample_yaml_data


@pytest.fixture
def invalid_yaml_data():
    """Create invalid YAML data missing required fields."""
    return {
        "blog_title": "test-blog",
        "current_version": 1,
        # Missing review_pipeline
        "final_release": {
            "complete": False,
            "released_by": None,
            "timestamp": None
        }
    }


def test_load_yaml(temp_yaml_file, valid_yaml_data):
    """Test loading a YAML file."""
    # Write test data to file
    with open(temp_yaml_file, 'w') as f:
        import yaml
        yaml.dump(valid_yaml_data, f)
    
    # Test loading
    data = load_yaml(temp_yaml_file)
    assert data == valid_yaml_data


def test_load_yaml_file_not_found():
    """Test loading a non-existent YAML file."""
    with pytest.raises(YamlGuardError) as excinfo:
        load_yaml("/nonexistent/file.yaml")
    assert "YAML file not found" in str(excinfo.value)


def test_save_yaml(temp_yaml_file, valid_yaml_data):
    """Test saving data to a YAML file."""
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    # Verify file was written
    assert os.path.exists(temp_yaml_file)
    
    # Verify content
    loaded_data = load_yaml(temp_yaml_file)
    assert loaded_data == valid_yaml_data


def test_validate_yaml_structure_valid(valid_yaml_data):
    """Test validating a valid YAML structure."""
    # Should not raise exceptions
    validate_yaml_structure(valid_yaml_data)


def test_validate_yaml_structure_invalid(invalid_yaml_data):
    """Test validating an invalid YAML structure."""
    with pytest.raises(YamlGuardError) as excinfo:
        validate_yaml_structure(invalid_yaml_data)
    assert "Missing required key: review_pipeline" in str(excinfo.value)


def test_validate_yaml_structure_missing_stage():
    """Test validating YAML with missing review stage."""
    # Create data with missing stage
    data = {
        "blog_title": "test-blog",
        "current_version": 1,
        "review_pipeline": {
            "factual_review": {
                "complete": False,
                "completed_by": None,
                "result_file": None,
                "timestamp": None
            },
            # Missing style_review
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
    
    with pytest.raises(YamlGuardError) as excinfo:
        validate_yaml_structure(data)
    assert "Missing review stage: style_review" in str(excinfo.value)


def test_validate_stage_transition_valid(valid_yaml_data):
    """Test validating a valid stage transition."""
    # Should not raise exception
    validate_stage_transition(valid_yaml_data, "factual_review")


def test_validate_stage_transition_invalid_stage():
    """Test validating transition to an invalid stage."""
    data = {
        "review_pipeline": {
            "factual_review": {"complete": False},
            "style_review": {"complete": False},
            "grammar_review": {"complete": False}
        }
    }
    
    with pytest.raises(ReviewStageError) as excinfo:
        validate_stage_transition(data, "nonexistent_stage")
    assert "Invalid review stage" in str(excinfo.value)


def test_validate_stage_transition_already_complete(valid_yaml_data):
    """Test validating transition to an already completed stage."""
    # Mark stage as complete
    valid_yaml_data["review_pipeline"]["factual_review"]["complete"] = True
    
    with pytest.raises(ReviewStageError) as excinfo:
        validate_stage_transition(valid_yaml_data, "factual_review")
    assert "is already completed" in str(excinfo.value)


def test_validate_stage_transition_prerequisite_incomplete(valid_yaml_data):
    """Test validating transition when prerequisite stage is incomplete."""
    with pytest.raises(ReviewStageError) as excinfo:
        validate_stage_transition(valid_yaml_data, "style_review")
    assert "Cannot proceed to style_review until factual_review is complete" in str(excinfo.value)


def test_mark_stage_complete(temp_yaml_file, valid_yaml_data):
    """Test marking a stage as complete."""
    # Save valid data to file
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    # Mark stage complete
    updated_data = mark_stage_complete(
        temp_yaml_file,
        "factual_review",
        "test-agent",
        "test-result.md"
    )
    
    # Verify data was updated
    assert updated_data["review_pipeline"]["factual_review"]["complete"] is True
    assert updated_data["review_pipeline"]["factual_review"]["completed_by"] == "test-agent"
    assert updated_data["review_pipeline"]["factual_review"]["result_file"] == "test-result.md"
    assert updated_data["review_pipeline"]["factual_review"]["timestamp"] is not None


def test_mark_stage_complete_invalid_stage(temp_yaml_file, valid_yaml_data):
    """Test marking an invalid stage as complete."""
    # Save valid data to file
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    with pytest.raises(ReviewStageError) as excinfo:
        mark_stage_complete(temp_yaml_file, "invalid_stage", "test-agent")
    assert "Invalid review stage" in str(excinfo.value)


def test_mark_stage_complete_out_of_order(temp_yaml_file, valid_yaml_data):
    """Test marking stages complete out of order."""
    # Save valid data to file
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    with pytest.raises(ReviewStageError) as excinfo:
        mark_stage_complete(temp_yaml_file, "style_review", "test-agent")
    assert "Cannot proceed to style_review until factual_review is complete" in str(excinfo.value)


def test_mark_blog_released(temp_yaml_file, valid_yaml_data):
    """Test marking a blog as released."""
    # Mark all stages as complete
    for stage in REVIEW_STAGES:
        valid_yaml_data["review_pipeline"][stage]["complete"] = True
    
    # Save to file
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    # Mark blog released
    updated_data = mark_blog_released(temp_yaml_file, "test-agent")
    
    # Verify
    assert updated_data["final_release"]["complete"] is True
    assert updated_data["final_release"]["released_by"] == "test-agent"
    assert updated_data["final_release"]["timestamp"] is not None


def test_mark_blog_released_incomplete_stages(temp_yaml_file, valid_yaml_data):
    """Test marking a blog as released with incomplete stages."""
    # Save to file
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    with pytest.raises(ReviewStageError) as excinfo:
        mark_blog_released(temp_yaml_file, "test-agent")
    assert "Cannot release blog until factual_review is complete" in str(excinfo.value)


def test_get_current_stage(valid_yaml_data):
    """Test getting the current stage in the review pipeline."""
    # All stages incomplete
    assert get_current_stage(valid_yaml_data) == "factual_review"
    
    # First stage complete
    valid_yaml_data["review_pipeline"]["factual_review"]["complete"] = True
    assert get_current_stage(valid_yaml_data) == "style_review"
    
    # All stages complete
    for stage in REVIEW_STAGES:
        valid_yaml_data["review_pipeline"][stage]["complete"] = True
    assert get_current_stage(valid_yaml_data) is None


def test_create_tracker_yaml():
    """Test creating a new YAML tracker file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test without research data
        yaml_path = create_tracker_yaml("test-blog", 1, output_dir=temp_dir)
        
        # Verify file was created
        assert os.path.exists(yaml_path)
        
        # Verify content
        data = load_yaml(yaml_path)
        assert data["blog_title"] == "test-blog"
        assert data["current_version"] == 1
        assert not data["review_pipeline"]["factual_review"]["complete"]
        assert not data["final_release"]["complete"]
        assert "research_data" not in data
        
        # Test with research data
        mock_research_data = {
            "industry_analysis": {
                "challenges": ["challenge1", "challenge2"]
            },
            "proposed_solution": {
                "pro_arguments": ["pro1", "pro2"],
                "counter_arguments": ["con1"],
                "metrics": ["metric1"]
            },
            "current_paradigm": {
                "origin_year": 2020,
                "alternatives": ["alt1"]
            },
            "audience_analysis": {
                "knowledge_gaps": ["gap1"],
                "acronyms": ["acronym1", "acronym2"]
            },
            "analogies": {
                "generated_analogies": ["analogy1"]
            },
            "citations": ["citation1", "citation2"],
            "visual_assets": ["asset1", "asset2", "asset3"]
        }
        
        yaml_path = create_tracker_yaml("test-blog-2", 1, mock_research_data, temp_dir)
        
        # Verify file was created
        assert os.path.exists(yaml_path)
        
        # Verify content
        data = load_yaml(yaml_path)
        assert data["blog_title"] == "test-blog-2"
        assert data["current_version"] == 1
        assert not data["review_pipeline"]["factual_review"]["complete"]
        assert not data["final_release"]["complete"]
        
        # Verify research data stats
        assert "research_data" in data
        assert data["research_data"]["industry_analysis"]["challenges_count"] == 2
        assert data["research_data"]["industry_analysis"]["sources_count"] == 2
        assert data["research_data"]["proposed_solution"]["pro_arguments_count"] == 2
        assert data["research_data"]["proposed_solution"]["counter_arguments_count"] == 1
        assert data["research_data"]["proposed_solution"]["metrics_count"] == 1
        assert data["research_data"]["proposed_solution"]["visual_assets_count"] == 3
        assert data["research_data"]["current_paradigm"]["origin_year"] == 2020
        assert data["research_data"]["current_paradigm"]["alternatives_count"] == 1
        assert data["research_data"]["audience_analysis"]["knowledge_gaps_count"] == 1
        assert data["research_data"]["audience_analysis"]["acronyms_count"] == 2
        assert data["research_data"]["audience_analysis"]["analogies_count"] == 1


def test_get_review_status(temp_yaml_file, valid_yaml_data):
    """Test getting the review status from a YAML file."""
    # Save to file
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    # Get status
    status = get_review_status(temp_yaml_file)
    
    # Verify
    assert status["blog_title"] == valid_yaml_data["blog_title"]
    assert status["version"] == valid_yaml_data["current_version"]
    assert status["current_stage"] == "factual_review"
    assert not status["all_stages_complete"]
    assert not status["released"]
    
    # All stages complete
    for stage in REVIEW_STAGES:
        valid_yaml_data["review_pipeline"][stage]["complete"] = True
    save_yaml(temp_yaml_file, valid_yaml_data)
    
    status = get_review_status(temp_yaml_file)
    assert status["all_stages_complete"] is True
    assert status["current_stage"] is None 