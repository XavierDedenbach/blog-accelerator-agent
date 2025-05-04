"""
YAML validation guard for review step transitions.
Enforces order of operations and provides helpers for updating YAML state.
"""

import os
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class YamlGuardError(Exception):
    """Exception raised for YAML validation errors."""
    pass


class ReviewStageError(YamlGuardError):
    """Exception raised when review stage transition is invalid."""
    pass


# Define the required structure
REVIEW_STAGES = ["factual_review", "style_review", "grammar_review"]


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load a YAML file safely.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        Dict containing the YAML content
        
    Raises:
        YamlGuardError: If file does not exist or is malformed
    """
    try:
        yaml_path = os.path.abspath(yaml_path)
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise YamlGuardError(f"YAML file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise YamlGuardError(f"YAML parsing error: {e}")


def save_yaml(yaml_path: str, data: Dict[str, Any]) -> None:
    """
    Save data to a YAML file.
    
    Args:
        yaml_path: Path to the YAML file
        data: Dict containing the YAML content
        
    Raises:
        YamlGuardError: If file cannot be written
    """
    try:
        yaml_path = os.path.abspath(yaml_path)
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    except (IOError, OSError) as e:
        raise YamlGuardError(f"Cannot write YAML file: {e}")


def validate_yaml_structure(data: Dict[str, Any]) -> None:
    """
    Validate the YAML has the correct structure.
    
    Args:
        data: Dict containing the YAML content
        
    Raises:
        YamlGuardError: If YAML structure is invalid
    """
    required_keys = ["blog_title", "current_version", "review_pipeline", "final_release"]
    for key in required_keys:
        if key not in data:
            raise YamlGuardError(f"Missing required key: {key}")
    
    # Validate the review_pipeline structure
    if not isinstance(data["review_pipeline"], dict):
        raise YamlGuardError("review_pipeline must be a dictionary")
    
    # Check all stages are present
    for stage in REVIEW_STAGES:
        if stage not in data["review_pipeline"]:
            raise YamlGuardError(f"Missing review stage: {stage}")
        
        # Check stage structure
        stage_data = data["review_pipeline"][stage]
        if not isinstance(stage_data, dict):
            raise YamlGuardError(f"Review stage {stage} must be a dictionary")
        
        required_stage_keys = ["complete", "completed_by", "result_file", "timestamp"]
        for key in required_stage_keys:
            if key not in stage_data:
                raise YamlGuardError(f"Missing required key in {stage}: {key}")
    
    # Validate final_release structure
    if not isinstance(data["final_release"], dict):
        raise YamlGuardError("final_release must be a dictionary")
    
    required_release_keys = ["complete", "released_by", "timestamp"]
    for key in required_release_keys:
        if key not in data["final_release"]:
            raise YamlGuardError(f"Missing required key in final_release: {key}")


def validate_stage_transition(data: Dict[str, Any], stage: str) -> None:
    """
    Validate that the requested stage transition is valid, enforcing the order.
    
    Args:
        data: Dict containing the YAML content
        stage: The stage to transition to
        
    Raises:
        ReviewStageError: If the stage transition is invalid
    """
    if stage not in REVIEW_STAGES:
        raise ReviewStageError(f"Invalid review stage: {stage}. Must be one of {REVIEW_STAGES}")
    
    # Check if stage already completed
    if data["review_pipeline"][stage]["complete"]:
        raise ReviewStageError(f"Stage {stage} is already completed")
    
    # Check prerequisite stages
    stage_index = REVIEW_STAGES.index(stage)
    if stage_index > 0:  # Not the first stage
        for i in range(stage_index):
            prev_stage = REVIEW_STAGES[i]
            if not data["review_pipeline"][prev_stage]["complete"]:
                raise ReviewStageError(
                    f"Cannot proceed to {stage} until {prev_stage} is complete"
                )


def mark_stage_complete(
    yaml_path: str, 
    stage: str, 
    completed_by: str, 
    result_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Mark a stage as complete in the YAML file.
    
    Args:
        yaml_path: Path to the YAML file
        stage: The stage to mark as complete
        completed_by: Who/what completed the stage
        result_file: Optional path to the review result file
        
    Returns:
        Updated YAML data
        
    Raises:
        ReviewStageError: If the stage transition is invalid
        YamlGuardError: If YAML file operations fail
    """
    # Load and validate YAML
    data = load_yaml(yaml_path)
    validate_yaml_structure(data)
    
    # Check if the stage transition is valid
    validate_stage_transition(data, stage)
    
    # Update the stage
    now = datetime.now(timezone.utc).isoformat()
    data["review_pipeline"][stage].update({
        "complete": True,
        "completed_by": completed_by,
        "result_file": result_file,
        "timestamp": now
    })
    
    # Save the updated YAML
    save_yaml(yaml_path, data)
    
    return data


def mark_blog_released(yaml_path: str, released_by: str) -> Dict[str, Any]:
    """
    Mark a blog as released in the YAML file.
    
    Args:
        yaml_path: Path to the YAML file
        released_by: Who/what released the blog
        
    Returns:
        Updated YAML data
        
    Raises:
        ReviewStageError: If any review stage is incomplete
        YamlGuardError: If YAML file operations fail
    """
    # Load and validate YAML
    data = load_yaml(yaml_path)
    validate_yaml_structure(data)
    
    # Check if all stages are complete
    for stage in REVIEW_STAGES:
        if not data["review_pipeline"][stage]["complete"]:
            raise ReviewStageError(f"Cannot release blog until {stage} is complete")
    
    # Update the final_release status
    now = datetime.now(timezone.utc).isoformat()
    data["final_release"].update({
        "complete": True,
        "released_by": released_by,
        "timestamp": now
    })
    
    # Save the updated YAML
    save_yaml(yaml_path, data)
    
    return data


def get_current_stage(data: Dict[str, Any]) -> Optional[str]:
    """
    Determine the current stage in the review pipeline.
    
    Args:
        data: Dict containing the YAML content
        
    Returns:
        Current stage name or None if all stages are complete
    """
    for stage in REVIEW_STAGES:
        if not data["review_pipeline"][stage]["complete"]:
            return stage
    
    return None


def create_tracker_yaml(
    blog_title: str, 
    version: int,
    research_data: Dict[str, Any] = None,
    output_dir: str = "data/tracker_yaml"
) -> str:
    """
    Create a new YAML tracker file for a blog with enhanced research data statistics.
    
    Args:
        blog_title: Title of the blog
        version: Version number
        research_data: Optional research data to include in the YAML
        output_dir: Directory to save the YAML file
        
    Returns:
        Path to the created YAML file
        
    Raises:
        YamlGuardError: If YAML file operations fail
    """
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the YAML file path
    yaml_path = os.path.join(output_dir, f"{blog_title}_review_tracker.yaml")
    
    # Create the initial YAML structure
    data = {
        "blog_title": blog_title,
        "current_version": version,
        "review_pipeline": {
            stage: {
                "complete": False,
                "completed_by": None,
                "result_file": None,
                "timestamp": None
            } for stage in REVIEW_STAGES
        },
        "final_release": {
            "complete": False,
            "released_by": None,
            "timestamp": None
        }
    }
    
    # Add research data statistics if provided
    if research_data:
        data["research_data"] = {
            "industry_analysis": {
                "challenges_count": len(research_data.get('industry_analysis', {}).get('challenges', [])),
                "sources_count": len(research_data.get('citations', [])),
            },
            "proposed_solution": {
                "pro_arguments_count": len(research_data.get('proposed_solution', {}).get('pro_arguments', [])),
                "counter_arguments_count": len(research_data.get('proposed_solution', {}).get('counter_arguments', [])),
                "metrics_count": len(research_data.get('proposed_solution', {}).get('metrics', [])),
                "visual_assets_count": len(research_data.get('visual_assets', [])),
            },
            "current_paradigm": {
                "origin_year": research_data.get('current_paradigm', {}).get('origin_year'),
                "alternatives_count": len(research_data.get('current_paradigm', {}).get('alternatives', [])),
            },
            "audience_analysis": {
                "knowledge_gaps_count": len(research_data.get('audience_analysis', {}).get('knowledge_gaps', [])),
                "acronyms_count": len(research_data.get('audience_analysis', {}).get('acronyms', [])),
                "analogies_count": len(research_data.get('analogies', {}).get('generated_analogies', [])),
            },
        }
    
    # Save the YAML file
    save_yaml(yaml_path, data)
    
    return yaml_path


def get_review_status(yaml_path: str) -> Dict[str, Any]:
    """
    Get the current review status from a YAML file.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        Dict with review status information
    """
    data = load_yaml(yaml_path)
    validate_yaml_structure(data)
    
    current_stage = get_current_stage(data)
    released = data["final_release"]["complete"]
    
    return {
        "blog_title": data["blog_title"],
        "version": data["current_version"],
        "current_stage": current_stage,
        "all_stages_complete": current_stage is None,
        "released": released,
        "stages": {
            stage: data["review_pipeline"][stage] 
            for stage in REVIEW_STAGES
        }
    } 