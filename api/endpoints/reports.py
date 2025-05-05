import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymongo.database import Database

from agents.utilities.db import get_db_client

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Configure templates
# Assuming templates are in a directory named 'templates' at the root
# The path might need adjustment based on where main.py is run from
templates = Jinja2Templates(directory="templates")

def get_report_data_from_db(db: Database, blog_title: str, version: int) -> Optional[dict]:
    """Helper function to retrieve report data from MongoDB."""
    try:
        # First, try to find the pre-generated markdown report
        report_doc = db.review_files.find_one({
            "blog_title": blog_title,
            "version": version,
            "stage": "research"  # Assuming research report is stored with stage 'research'
        })
        
        if report_doc and "content" in report_doc:
            # If found, return it for direct rendering
            logger.info(f"Found pre-generated report for {blog_title} v{version}")
            return {"report_markdown": report_doc["content"]}
        else:
            # If not found, try to retrieve the raw research data
            # Note: This assumes a 'research_data' collection exists as planned
            research_doc = db.research_data.find_one({
                "blog_title": blog_title,
                "version": version
            })
            if research_doc:
                logger.info(f"Found raw research data for {blog_title} v{version}")
                # Remove MongoDB ObjectId if present before returning
                if "_id" in research_doc:
                    del research_doc["_id"]
                return {"research_data": research_doc}
            else:
                logger.warning(f"No report or research data found for {blog_title} v{version}")
                return None
                
    except Exception as e:
        logger.error(f"Database error fetching report for {blog_title} v{version}: {e}", exc_info=True)
        return None

@router.get("/view/{blog_title}/{version}", response_class=HTMLResponse)
async def view_research_report(
    request: Request, 
    blog_title: str, 
    version: int, 
    db: Database = Depends(get_db_client)
):
    """Serves the HTML page for viewing a specific research report."""
    logger.info(f"Request received for viewing report: {blog_title} v{version}")
    
    # Retrieve data from DB
    report_content = get_report_data_from_db(db, blog_title, version)
    
    if not report_content:
        logger.warning(f"Report data not found for {blog_title} v{version}. Returning 404.")
        raise HTTPException(status_code=404, detail="Report or research data not found")
        
    # Prepare context for the template
    context = {
        "request": request,
        "blog_title": blog_title,
        "version": version,
        "report_markdown": report_content.get("report_markdown"),
        "research_data": report_content.get("research_data")
    }
    
    # Render the HTML template
    try:
        response = templates.TemplateResponse("report_viewer.html", context)
        logger.info(f"Successfully rendered report template for {blog_title} v{version}")
        return response
    except Exception as e:
        logger.error(f"Error rendering template report_viewer.html: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error rendering report: {e}") 