"""
Review endpoints for handling blog review stages and running the reviewer agent pipeline.

This module:
1. Accepts review stage triggers (factual, style, grammar)
2. Validates YAML state transitions
3. Runs reviewer agent for the selected stage
4. Updates YAML state and MongoDB
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.reviewer_agent import ReviewerAgent, ReviewStageError
from agents.utilities.yaml_guard import (
    load_yaml, validate_yaml_structure, validate_stage_transition,
    mark_stage_complete, mark_blog_released, get_review_status
)

# Create router
router = APIRouter()

# Request models
class ReviewRequest(BaseModel):
    """Request model for review stage processing."""
    blog_title: str
    version: int
    stage: str = Field(..., description="Review stage to process (factual_review, style_review, grammar_review)")
    yaml_path: Optional[str] = None
    async_process: bool = False

    class Config:
        schema_extra = {
            "example": {
                "blog_title": "why-microgrids-will-replace-utilities",
                "version": 1,
                "stage": "factual_review",
                "async_process": False
            }
        }

class ReleaseRequest(BaseModel):
    """Request model for marking a blog as released."""
    blog_title: str
    version: int
    yaml_path: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "blog_title": "why-microgrids-will-replace-utilities",
                "version": 1
            }
        }

# Response models
class ReviewResponse(BaseModel):
    """Response model for review endpoints."""
    status: str
    message: str
    blog_title: Optional[str] = None
    version: Optional[int] = None
    stage: Optional[str] = None
    next_stage: Optional[str] = None
    report_filename: Optional[str] = None
    yaml_path: Optional[str] = None


async def process_review_stage_background(
    blog_title: str,
    version: int,
    stage: str,
    yaml_path: str
) -> Dict[str, Any]:
    """
    Process a review stage in the background.
    
    Args:
        blog_title: Title of the blog
        version: Version number
        stage: Review stage to process
        yaml_path: Path to the YAML file
        
    Returns:
        Dict with processing results
    """
    try:
        agent = ReviewerAgent()
        result = agent.process_review_stage(blog_title, version, stage, yaml_path)
        return result
    except Exception as e:
        print(f"Error in background review processing: {e}")
        raise


def resolve_yaml_path(blog_title: str, provided_path: Optional[str] = None) -> str:
    """
    Resolve the YAML path for a blog.
    
    Args:
        blog_title: Title of the blog
        provided_path: Optionally provided YAML path
        
    Returns:
        Resolved YAML path
    """
    if provided_path:
        if os.path.exists(provided_path):
            return provided_path
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Provided YAML path does not exist: {provided_path}"
            )
    
    # Default path in data/tracker_yaml directory
    default_path = os.path.join("data", "tracker_yaml", f"{blog_title}_review_tracker.yaml")
    if os.path.exists(default_path):
        return default_path
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"YAML tracker file not found for blog: {blog_title}"
        )


@router.post("/stage", response_model=ReviewResponse)
async def process_review_stage(
    background_tasks: BackgroundTasks,
    review_request: ReviewRequest = Body(...)
):
    """
    Process a review stage for a blog.
    
    Args:
        background_tasks: FastAPI background tasks
        review_request: Review request data
        
    Returns:
        ReviewResponse with status and info
    """
    try:
        # Validate stage
        if review_request.stage not in ["factual_review", "style_review", "grammar_review"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid review stage: {review_request.stage}. Must be one of: factual_review, style_review, grammar_review"
            )
        
        # Resolve YAML path
        yaml_path = resolve_yaml_path(review_request.blog_title, review_request.yaml_path)
        
        # Validate YAML and stage transition
        yaml_data = load_yaml(yaml_path)
        validate_yaml_structure(yaml_data)
        
        try:
            validate_stage_transition(yaml_data, review_request.stage)
        except ReviewStageError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if review_request.async_process:
            # Process in background
            background_tasks.add_task(
                process_review_stage_background,
                review_request.blog_title,
                review_request.version,
                review_request.stage,
                yaml_path
            )
            
            return ReviewResponse(
                status="accepted",
                message=f"Review stage {review_request.stage} for blog {review_request.blog_title} (v{review_request.version}) queued for processing.",
                blog_title=review_request.blog_title,
                version=review_request.version,
                stage=review_request.stage,
                yaml_path=yaml_path
            )
        else:
            # Process synchronously
            agent = ReviewerAgent()
            result = agent.process_review_stage(
                review_request.blog_title,
                review_request.version,
                review_request.stage,
                yaml_path
            )
            
            # Get the next stage if there is one
            current_status = get_review_status(yaml_path)
            next_stage = current_status.get("current_stage")
            
            return ReviewResponse(
                status="success",
                message=f"Review stage {review_request.stage} processed successfully.",
                blog_title=review_request.blog_title,
                version=review_request.version,
                stage=review_request.stage,
                next_stage=next_stage,
                report_filename=result.get("report_filename"),
                yaml_path=yaml_path
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing review stage: {str(e)}")


@router.post("/release", response_model=ReviewResponse)
async def mark_blog_released(release_request: ReleaseRequest = Body(...)):
    """
    Mark a blog as released.
    
    Args:
        release_request: Release request data
        
    Returns:
        ReviewResponse with status and info
    """
    try:
        # Resolve YAML path
        yaml_path = resolve_yaml_path(release_request.blog_title, release_request.yaml_path)
        
        # Validate YAML
        yaml_data = load_yaml(yaml_path)
        validate_yaml_structure(yaml_data)
        
        # Verify all stages are complete
        status = get_review_status(yaml_path)
        if not status.get("all_stages_complete"):
            incomplete_stage = status.get("current_stage")
            raise HTTPException(
                status_code=400,
                detail=f"Cannot release blog: Review stage {incomplete_stage} is not complete"
            )
        
        # Mark as released
        agent = ReviewerAgent()
        result = agent.mark_blog_as_released(yaml_path)
        
        return ReviewResponse(
            status="success",
            message=f"Blog {release_request.blog_title} (v{release_request.version}) marked as released.",
            blog_title=release_request.blog_title,
            version=release_request.version,
            yaml_path=yaml_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking blog as released: {str(e)}")


@router.get("/status/{blog_title}", response_model=Dict[str, Any])
async def get_review_status_endpoint(blog_title: str, yaml_path: Optional[str] = None):
    """
    Get the current review status of a blog.
    
    Args:
        blog_title: Title of the blog
        yaml_path: Optional path to the YAML file
        
    Returns:
        Dict with review status information
    """
    try:
        # Resolve YAML path
        resolved_yaml_path = resolve_yaml_path(blog_title, yaml_path)
        
        # Get status
        status = get_review_status(resolved_yaml_path)
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving review status: {str(e)}")
