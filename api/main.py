"""
FastAPI main application for Blog Accelerator Agent.

This module:
1. Sets up the FastAPI application
2. Registers API routes
3. Configures middleware
4. Provides health check endpoint
5. Configures static files and templates for the report viewer
"""

import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import route modules
from api.endpoints import process, review, reports  # Added reports

# Create FastAPI app
app = FastAPI(
    title="Blog Accelerator API",
    description="API for Blog Accelerator research and review pipeline",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure static files (for CSS, JS)
# Assuming a 'static' directory exists at the root
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates (shared instance, although endpoints might define their own)
# Base templates are in the 'templates' directory
templates = Jinja2Templates(directory="templates")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        Dict with status and version info
    """
    return {
        "status": "healthy",
        "version": app.version,
        "service": "blog-accelerator-api"
    }

# Register routes
app.include_router(process.router, prefix="/process", tags=["Process"])
app.include_router(review.router, prefix="/review", tags=["Review"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"]) # Added reports router

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Execute tasks on application startup.
    Verify environment variables and connections.
    Create static/template directories if they don't exist.
    """
    # Create required directories if they don't exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    required_vars = [
        "MONGODB_URI"
    ]
    
    # Check for required environment variables
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        # Don't fail startup, but log the warning
    
    # Additional startup tasks could include:
    # - Verifying MongoDB connection
    
    print("Blog Accelerator API started successfully")


if __name__ == "__main__":
    # For local development only
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8080, reload=True)
