"""
Process endpoints for handling blog uploads and running the researcher agent pipeline.

This module:
1. Accepts blog uploads (ZIP files with markdown and images)
2. Extracts and validates content
3. Stores content in MongoDB
4. Triggers researcher agent for analysis
"""

import os
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents.researcher_agent import ResearcherAgent
from agents.utilities.file_ops import FileOpsError

# Create router
router = APIRouter()

# Response models
class ProcessResponse(BaseModel):
    """Response model for process endpoints."""
    status: str
    message: str
    blog_title: Optional[str] = None
    version: Optional[int] = None
    readiness_score: Optional[int] = None
    yaml_path: Optional[str] = None


async def process_blog_background(file_path: str) -> Dict[str, Any]:
    """
    Process a blog in the background.
    
    Args:
        file_path: Path to the uploaded file
        
    Returns:
        Dict with processing results
    """
    try:
        agent = ResearcherAgent()
        result = agent.process_blog(file_path)
        
        # Cleanup the temporary file if needed
        if os.path.exists(file_path) and file_path.startswith(tempfile.gettempdir()):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file_path}: {e}")
        
        return result
    except Exception as e:
        print(f"Error in background processing: {e}")
        raise


@router.post("/upload", response_model=ProcessResponse)
async def upload_blog(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_async: bool = Query(False, description="Process the blog asynchronously")
):
    """
    Upload a blog file (ZIP or markdown) for processing.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded file (ZIP or markdown)
        process_async: Whether to process the blog asynchronously
        
    Returns:
        ProcessResponse with status and info
    """
    # Validate file type
    filename = file.filename.lower()
    if not (filename.endswith('.zip') or filename.endswith('.md') or filename.endswith('.markdown')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Only ZIP or markdown files are supported."
        )
    
    # Create uploads directory if it doesn't exist
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    file_path = upload_dir / filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    try:
        if process_async:
            # Process in background
            background_tasks.add_task(process_blog_background, str(file_path))
            return ProcessResponse(
                status="accepted",
                message=f"File {filename} uploaded successfully. Processing in background."
            )
        else:
            # Process synchronously
            agent = ResearcherAgent()
            result = agent.process_blog(str(file_path))
            
            return ProcessResponse(
                status="success",
                message=f"File {filename} processed successfully.",
                blog_title=result.get("blog_title"),
                version=result.get("version"),
                readiness_score=result.get("readiness_score"),
                yaml_path=result.get("yaml_path")
            )
            
    except FileOpsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/status/{blog_title}", response_model=Dict[str, Any])
async def get_blog_status(blog_title: str):
    """
    Get the current status of a blog.
    
    Args:
        blog_title: Title of the blog
        
    Returns:
        Dict with blog status information
    """
    try:
        agent = ResearcherAgent()
        blog_status = agent.db_client.get_blog_status(blog_title)
        
        if not blog_status:
            raise HTTPException(status_code=404, detail=f"Blog '{blog_title}' not found")
        
        return blog_status
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error retrieving blog status: {str(e)}")
