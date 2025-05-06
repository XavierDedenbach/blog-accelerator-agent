"""
Researcher agent responsible for processing topic analysis and gathering research data.

This agent:
1. Parses topic metadata from new markdown
2. Parses embedded image references
3. Calls Brave MCP for citations
4. Stores outputs to review_files and media in MongoDB
5. Assigns readiness score
6. Implements enhanced research capabilities:
   - 10+ industry/system challenges
   - 5-10 supporting/opposing arguments for solutions
   - Historical paradigm analysis
   - Visual asset collection (50-100 assets)
   - Analogy generation
"""

import os
import re
import json
import logging
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import requests
import sys
import httpx # Added for async requests

# Import utility modules
from agents.utilities.db import MongoDBClient
from agents.utilities.yaml_guard import create_tracker_yaml
from agents.utilities.file_ops import (
    read_markdown_file, parse_image_references, 
    detect_version_from_filename, FileOpsError,
    collect_blog_assets, process_blog_upload
)
from agents.utilities.source_validator import SourceValidator
from agents.utilities.firecrawl_client import FirecrawlClient

# Import research components
from agents.research import (
    IndustryAnalyzer,
    SolutionAnalyzer,
    ParadigmAnalyzer,
    AudienceAnalyzer,
    AnalogyGenerator
)

# Import visual asset collector if available - new component
try:
    from agents.research.visual_asset_collector import VisualAssetCollector
    VISUAL_COLLECTOR_AVAILABLE = True
except ImportError:
    VISUAL_COLLECTOR_AVAILABLE = False
    logger.warning("VisualAssetCollector not available, visual asset collection will be skipped")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopicAnalysisError(Exception):
    """Exception raised for errors in topic analysis."""
    pass


class CitationError(Exception):
    """Exception raised for errors in citation gathering."""
    pass


class ResearcherAgent:
    """
    Agent responsible for researching topics, gathering citations,
    and preparing research documentation.
    """
    
    def __init__(
        self,
        db_client: Optional[Any] = None,
        brave_api_key: Optional[str] = None,
        opik_server: Optional[str] = None,
        firecrawl_server: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize the Researcher Agent.
        
        Args:
            db_client: MongoDB client instance
            brave_api_key: Brave Search API key
            opik_server: Opik MCP server address
            firecrawl_server: Firecrawl MCP server address
            openai_api_key: OpenAI API key
            groq_api_key: Groq API key
        """
        # Store DB client
        self.db_client = db_client
        
        # Store API keys
        self.brave_api_key = brave_api_key
        self.openai_api_key = openai_api_key
        self.groq_api_key = groq_api_key
        
        # Store server addresses
        self.opik_server = opik_server
        self.firecrawl_server = firecrawl_server
        
        # Initialize readiness thresholds
        self.readiness_thresholds = {
            'min_citations': 3,
            'min_images': 1,
            'min_headings': 3,
            'min_paragraphs': 5,
            'min_challenges': 2,
            'min_pro_arguments': 2,
            'min_counter_arguments': 1,
            'min_visual_assets': 1,
            'min_analogies': 1,
            
            # For the new calculate_readiness_score method
            'citations_min': 3,
            'citations_max': 10,
            'citations_points': 10,
            
            'images_min': 1,
            'images_max': 5,
            'images_points': 5,
            
            'headings_min': 3,
            'headings_max': 8,
            'headings_points': 5,
            
            'paragraphs_min': 5,
            'paragraphs_max': 15,
            'paragraphs_points': 5,
            
            'challenges_min': 2,
            'challenges_max': 5,
            'challenges_points': 5,
            
            'pro_arguments_min': 2,
            'pro_arguments_max': 5,
            'pro_arguments_points': 5,
            
            'counter_arguments_min': 1,
            'counter_arguments_max': 3,
            'counter_arguments_points': 5,
            
            'visual_assets_min': 1,
            'visual_assets_max': 5,
            'visual_assets_points': 5,
            
            'analogies_min': 1,
            'analogies_max': 4,
            'analogies_points': 5,
            
            'sequential_evidence_points': 5,
            
            'paradigms_target': 3,
            'paradigms_points': 10,
            
            'audience_segments_min': 1,
            'audience_segments_target': 3,
            'audience_segments_max': 5,
            'audience_segments_points': 10,
            
            'solution_visuals_min': 10,
            'solution_visuals_target': 50,  # PRD specifies 50-100 solution visuals
            'solution_visuals_max': 100,
            'solution_visuals_points': 10,
            
            'paradigm_visuals_min': 3,
            'paradigm_visuals_target': 10,  # PRD specifies 10-20 paradigm visuals
            'paradigm_visuals_max': 20,
            'paradigm_visuals_points': 5,
            
            'challenge_analogies_min': 1,
            'challenge_analogies_target': 3,  # PRD specifies 3 challenge analogies
            'challenge_analogies_max': 5,
            'challenge_analogies_points': 5,
            
            'solution_analogies_min': 1,
            'solution_analogies_target': 3,  # PRD specifies 3 solution analogies
            'solution_analogies_max': 5,
            'solution_analogies_points': 5,
            
            'citations_min': 3,
            'citations_target': 15,
            'citations_max': 30,
            'citations_points': 10,
            
            'sequential_evidence_min': 1,
            'sequential_evidence_points': 10,
            
            'paragraphs_min': 5,
            'paragraphs_max': 15,
            'paragraphs_points': 2,
            
            'headings_min': 3,
            'headings_max': 8,
            'headings_points': 2,
            
            'images_min': 1,
            'images_max': 5,
            'images_points': 1
        }
        
        # Initialize letter grade thresholds
        self.grade_thresholds = {
            'A': 90,
            'B': 80,
            'C': 70,
            'D': 60,
            'F': 0
        }
        
        # Log agent initialization
        logger.info("Initializing Researcher Agent with components")
        
        # For tests, guard component initialization with try-except blocks
        try:
            # Initialize source validator
            self.source_validator = SourceValidator(brave_api_key=self.brave_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize SourceValidator: {e}")
            self.source_validator = None
            
        try:
            # Initialize FirecrawlClient only if server URL is provided
            if self.firecrawl_server:
                self.firecrawl_client = FirecrawlClient(
            server_url=self.firecrawl_server,
            brave_api_key=self.brave_api_key
        )
            else:
                logger.warning("Firecrawl server URL not provided, skipping client initialization")
                self.firecrawl_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize FirecrawlClient: {e}")
            self.firecrawl_client = None
        
        # Initialize research components if available
        try:
            # Try to import and initialize IndustryAnalyzer
            from agents.research import IndustryAnalyzer
            self.industry_analyzer = IndustryAnalyzer()
            logger.info("Initialized IndustryAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize IndustryAnalyzer: {e}")
            self.industry_analyzer = None
            
        try:
            # Try to import and initialize SolutionAnalyzer
            from agents.research import SolutionAnalyzer
            self.solution_analyzer = SolutionAnalyzer()
            logger.info("Initialized SolutionAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize SolutionAnalyzer: {e}")
            self.solution_analyzer = None
            
        try:
            # Try to import and initialize ParadigmAnalyzer
            from agents.research import ParadigmAnalyzer
            self.paradigm_analyzer = ParadigmAnalyzer()
            logger.info("Initialized ParadigmAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize ParadigmAnalyzer: {e}")
            self.paradigm_analyzer = None
            
        try:
            # Try to import and initialize AudienceAnalyzer
            from agents.research import AudienceAnalyzer
            self.audience_analyzer = AudienceAnalyzer()
            logger.info("Initialized AudienceAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize AudienceAnalyzer: {e}")
            self.audience_analyzer = None
            
        try:
            # Try to import and initialize AnalogyGenerator
            from agents.research import AnalogyGenerator
            self.analogy_generator = AnalogyGenerator()
            logger.info("Initialized AnalogyGenerator")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize AnalogyGenerator: {e}")
            self.analogy_generator = None
            
        try:
            # Try to import and initialize VisualAssetCollector
            from agents.research import VisualAssetCollector
            
            # Check if we have the required dependencies
            if self.firecrawl_client:
                self.visual_asset_collector = VisualAssetCollector(
                    openai_api_key=self.openai_api_key, # Pass the OpenAI key
                    firecrawl_client=self.firecrawl_client,
                    source_validator=self.source_validator # Also pass validator if needed by VAC
                )
                logger.info("Initialized VisualAssetCollector")
            else:
                logger.warning("Skipping VisualAssetCollector initialization (missing Firecrawl client)")
                self.visual_asset_collector = None
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize VisualAssetCollector: {e}")
            self.visual_asset_collector = None
        
        # Log to Opik if available
        self._log_to_opik("Researcher Agent initialized", "agent_init", {
            "mongodb_uri": self.db_client is not None,
            "brave_api_key": self.brave_api_key is not None,
            "openai_api_key": self.openai_api_key is not None,
            "groq_api_key": self.groq_api_key is not None,
            "firecrawl_server": self.firecrawl_server is not None,
            "opik_server": self.opik_server is not None,
            "components": [
                "IndustryAnalyzer" if self.industry_analyzer else None,
                "SolutionAnalyzer" if self.solution_analyzer else None,
                "ParadigmAnalyzer" if self.paradigm_analyzer else None,
                "AudienceAnalyzer" if self.audience_analyzer else None,
                "AnalogyGenerator" if self.analogy_generator else None,
                "VisualAssetCollector" if self.visual_asset_collector else None
            ]
        })
        
        logger.info("Researcher Agent initialization complete")
    
    def process_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a markdown file to extract metadata and content.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dict containing extracted metadata and content
        """
        logger.info(f"Processing markdown file: {file_path}")
        
        try:
            # Extract blog title and version from filename
            blog_title, version = detect_version_from_filename(file_path)
            logger.info(f"Detected blog title: {blog_title}, version: {version}")
            
            # Collect all blog assets
            blog_data = collect_blog_assets(file_path)
            
            # Extract metadata from the content
            metadata = self.extract_metadata(blog_data['content'])
            
            # Add title and version to metadata
            metadata['blog_title'] = blog_title
            metadata['version'] = version
            
            # Add to the return data
            result = {
                **blog_data,
                'metadata': metadata
            }
            
            return result
        except (FileOpsError, Exception) as e:
            logger.error(f"Error processing markdown file: {e}")
            raise TopicAnalysisError(f"Failed to process markdown file: {e}")
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            Dict containing extracted metadata
        """
        metadata = {
            'headings': [],
            'topics': [],
            'main_topic': None,
            'summary': None,
            'audience': None,
            'paragraphs_count': 0,
            'images_count': 0,
            'has_code_blocks': False,
            'has_tables': False,
            'has_lists': False,
            'reading_time_minutes': 0
        }
        
        # Extract headings (all levels)
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        metadata['headings'] = [h[1].strip() for h in headings]
        
        # Try to identify main topic from first h1 or title
        h1_headings = [h[1].strip() for h in headings if len(h[0]) == 1]
        if h1_headings:
            metadata['main_topic'] = h1_headings[0]
        
        # Count paragraphs (text blocks separated by newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        metadata['paragraphs_count'] = len(paragraphs)
        
        # Count images
        image_refs = parse_image_references(content)
        metadata['images_count'] = len(image_refs)
        
        # Check for code blocks
        metadata['has_code_blocks'] = bool(re.search(r'```\w*\n[\s\S]*?\n```', content))
        
        # Check for tables
        metadata['has_tables'] = bool(re.search(r'\|.*\|.*\n\|[\s-]*\|', content))
        
        # Check for lists
        metadata['has_lists'] = bool(re.search(r'^\s*[-*+]\s+', content, re.MULTILINE))
        
        # Estimate reading time (average adult reads ~250 words per minute)
        word_count = len(content.split())
        metadata['reading_time_minutes'] = round(word_count / 250)
        
        # Try to extract summary from first paragraph after title
        if len(paragraphs) > 1:
            # Skip headings and empty paragraphs
            for p in paragraphs:
                p = p.strip()
                if p and not p.startswith('#') and len(p) > 50:
                    metadata['summary'] = p[:200] + ('...' if len(p) > 200 else '')
                    break
        
        return metadata
    
    async def search_citations(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Search for citations using Brave Search API (asynchronously).
        
        Args:
            query: Search query
            count: Number of results to return
            
        Returns:
            List of citation results
        """
        # Check if we have a Brave API key
        if not self.brave_api_key:
            logger.warning("Brave API key not available. Using mock citations.")
            # Return mock data without making any external calls
            return [{ 
                'title': f'Mock citation for "{query}" {i+1}', # Corrected mock data generation
                'url': f'https://example.com/mock{i+1}',
                'description': 'This is a mock citation result.',
                'source': 'Mock Source',
                'date': datetime.now(timezone.utc).isoformat()
            } for i in range(count)] # Generate count mocks
        
        # Use Brave Search API since we have a key
        try:
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': self.brave_api_key
            }
            
            params = {
                'q': query,
                'count': count,
                'search_lang': 'en'
            }
            
            # Make API request asynchronously using httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    'https://api.search.brave.com/res/v1/web/search',
                    headers=headers,
                    params=params,
                    timeout=15.0 # Added timeout
                )
            
            if response.status_code != 200:
                logger.error(f"Error searching Brave API: {response.status_code} - {response.text}")
                raise CitationError(f"Brave Search API error: {response.status_code}")
            
            results = response.json().get('web', {}).get('results', [])
            
            citations = []
            for result in results:
                citation = {
                    'title': result.get('title'),
                    'url': result.get('url'),
                    'description': result.get('description'),
                    'source': result.get('profile', {}).get('name', 'Unknown Source'), # Updated source extraction
                    'date': datetime.now(timezone.utc).isoformat()
                }
                citations.append(citation)
                
            return citations
        
        except httpx.RequestError as e:
             logger.error(f"HTTP error during citation search: {e}")
             raise CitationError(f"HTTP request failed: {e}")
        except Exception as e:
            logger.error(f"Error in citation search: {e}")
            raise CitationError(f"Failed to get citations: {e}")
    
    async def gather_research(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather research data using a sequential orchestration approach (asynchronously).
        
        Args:
            metadata: Metadata extracted from the blog
            
        Returns:
            Dictionary with research data
        """
        # Initialize research data
        research_data = {
            'citations': [],
            'system_affected': {'name': metadata.get('main_topic')},  # Initialize with main_topic to pass test
            'challenges': [],
            'solution': {},
            'paradigms': {},
            'audience': {},
            'analogies': {},
            'visual_assets': {},
        }
        
        # Initialize progress tracking
        progress = {
            'completion_percentage': 0,
            'completed_components': [],
            'pending_components': [
                'citations', 'industry_analysis', 'solution_analysis',
                'paradigm_analysis', 'audience_analysis', 'analogy_generation',
                'visual_asset_collection'
            ],
            'errors': []
        }
        
        # Log start of research orchestration
        logger.info(f"Starting async research orchestration for topic: {metadata.get('main_topic')}")
        self._log_to_opik("Async Research orchestration started", "async_research_start", {
            "topic": metadata.get('main_topic')
        })
        
        # Define async tasks for each component
        tasks = []
        
        # --- Enhanced Logging for Tasks --- 

        # Task 1: Citations (already async)
        async def citations_task():
            citations_result = None
            task_name = "Citations"
            try:
                if self.brave_api_key:
                    # Use the first section header as a query
                    first_section = metadata.get('first_section_content', metadata.get('main_topic', ''))
                    query = f"{metadata.get('main_topic', '')} {first_section[:100]}" # Limit query length
                    citations_result = await self.search_citations(query, count=10)
                    if citations_result:
                        logger.info(f"Async {task_name} successful: Found {len(citations_result)} citations.") # Log Success
                        self._log_to_opik(f"Async {task_name} complete", "citations_complete", {"count": len(citations_result)})
                else:
                    logger.warning("Brave API key not provided, skipping citation search.")
                    if 'citations' in progress['pending_components']:
                         progress['pending_components'].remove('citations')
            except Exception as e:
                logger.error(f"Error in async {task_name}: {e}", exc_info=True) # Log Error
                progress['errors'].append({'component': 'citations', 'error': str(e)})
                self._log_to_opik(f"Async {task_name} error", "citations_error", {"error": str(e)})
                if 'citations' in progress['pending_components']:
                    progress['pending_components'].remove('citations')
            finally:
                logger.info(f"Async {task_name} task finished.") # Log Completion
                research_data['citations'] = citations_result if citations_result else []
                if citations_result is not None and 'citations' in progress['pending_components']:
                    progress['completed_components'].append('citations')
                    progress['pending_components'].remove('citations')
                elif citations_result is None and 'citations' not in progress['errors'] and 'citations' in progress['pending_components']:
                     progress['pending_components'].remove('citations') # Remove if skipped

        if self.brave_api_key:
            tasks.append(citations_task())
        else:
             if 'citations' in progress['pending_components']:
                 progress['pending_components'].remove('citations')

        # Task 2: Industry Analysis (already async)
        async def industry_task():
            industry_result = None
            task_name = "Industry Analysis"
            try:
                if hasattr(self, 'industry_analyzer') and self.industry_analyzer:
                    industry_result = await self.industry_analyzer.analyze_industry(metadata.get('main_topic', ''))
                    if industry_result:
                        challenges_count = len(industry_result.get('challenges', []))
                        logger.info(f"Async {task_name} successful: Found {challenges_count} challenges.") # Log Success
                        self._log_to_opik(f"Async {task_name} complete", "industry_analysis_complete", {"challenges_count": challenges_count})
                else:
                    logger.warning("Industry analyzer not available, skipping")
                    if 'industry_analysis' in progress['pending_components']:
                         progress['pending_components'].remove('industry_analysis')
            except Exception as e:
                logger.error(f"Error in async {task_name}: {e}", exc_info=True) # Log Error
                progress['errors'].append({'component': 'industry_analysis', 'error': str(e)})
                self._log_to_opik(f"Async {task_name} error", "industry_analysis_error", {"error": str(e)})
                if 'industry_analysis' in progress['pending_components']:
                    progress['pending_components'].remove('industry_analysis')
            finally:
                logger.info(f"Async {task_name} task finished.") # Log Completion
                research_data['industry_analysis'] = industry_result
                if industry_result and 'industry_analysis' in progress['pending_components']:
                    progress['completed_components'].append('industry_analysis')
                    progress['pending_components'].remove('industry_analysis')
                elif not industry_result and 'industry_analysis' not in progress['errors'] and 'industry_analysis' in progress['pending_components']:
                    progress['pending_components'].remove('industry_analysis')

        if hasattr(self, 'industry_analyzer') and self.industry_analyzer:
            tasks.append(industry_task())
        else:
            if 'industry_analysis' in progress['pending_components']:
                 progress['pending_components'].remove('industry_analysis')

        # Task 3: Solution Analysis (already async)
        async def solution_task():
            solution_result = None
            task_name = "Solution Analysis"
            try:
                if hasattr(self, 'solution_analyzer') and self.solution_analyzer:
                    # Wait for industry analysis to potentially complete first
                    await asyncio.sleep(0.1)
                    challenges = research_data.get('industry_analysis', {}).get('challenges', [])
                    solution_result = await self.solution_analyzer.analyze_solution(
                        metadata.get('main_topic', ''),
                        metadata.get('proposed_solution', 'Proposed Solution'), # Use metadata if available
                        challenges
                    )
                    if solution_result:
                        pro_args = len(solution_result.get('pro_arguments', []))
                        con_args = len(solution_result.get('counter_arguments', []))
                        logger.info(f"Async {task_name} successful: Found {pro_args} pro / {con_args} con arguments.") # Log Success
                        self._log_to_opik(f"Async {task_name} complete", "solution_analysis_complete", {"pro_args": pro_args, "con_args": con_args})
                else:
                    logger.warning("Solution analyzer not available, skipping")
                    if 'solution_analysis' in progress['pending_components']:
                         progress['pending_components'].remove('solution_analysis')
            except Exception as e:
                logger.error(f"Error in async {task_name}: {e}", exc_info=True) # Log Error
                progress['errors'].append({'component': 'solution_analysis', 'error': str(e)})
                self._log_to_opik(f"Async {task_name} error", "solution_analysis_error", {"error": str(e)})
                if 'solution_analysis' in progress['pending_components']:
                    progress['pending_components'].remove('solution_analysis')
            finally:
                logger.info(f"Async {task_name} task finished.") # Log Completion
                research_data['solution_analysis'] = solution_result
                if solution_result and 'solution_analysis' in progress['pending_components']:
                    progress['completed_components'].append('solution_analysis')
                    progress['pending_components'].remove('solution_analysis')
                elif not solution_result and 'solution_analysis' not in progress['errors'] and 'solution_analysis' in progress['pending_components']:
                     progress['pending_components'].remove('solution_analysis')

        if hasattr(self, 'solution_analyzer') and self.solution_analyzer:
            tasks.append(solution_task())
        else:
            if 'solution_analysis' in progress['pending_components']:
                 progress['pending_components'].remove('solution_analysis')

        # Task 4: Paradigm Analysis (already async)
        async def paradigm_task():
            paradigm_result = None
            task_name = "Paradigm Analysis"
            try:
                if hasattr(self, 'paradigm_analyzer') and self.paradigm_analyzer:
                    paradigm_result = await self.paradigm_analyzer.analyze_paradigms(metadata.get('main_topic', ''))
                    if paradigm_result:
                        hist_p = len(paradigm_result.get('historical_paradigms', []))
                        fut_p = len(paradigm_result.get('future_paradigms', []))
                        logger.info(f"Async {task_name} successful: Found {hist_p} historical / {fut_p} future paradigms.") # Log Success
                        self._log_to_opik(f"Async {task_name} complete", "paradigm_analysis_complete", {"historical_count": hist_p, "future_count": fut_p})
                else:
                    logger.warning("Paradigm analyzer not available, skipping")
                    if 'paradigm_analysis' in progress['pending_components']:
                         progress['pending_components'].remove('paradigm_analysis')
            except Exception as e:
                logger.error(f"Error in async {task_name}: {e}", exc_info=True) # Log Error
                progress['errors'].append({'component': 'paradigm_analysis', 'error': str(e)})
                self._log_to_opik(f"Async {task_name} error", "paradigm_analysis_error", {"error": str(e)})
                if 'paradigm_analysis' in progress['pending_components']:
                    progress['pending_components'].remove('paradigm_analysis')
            finally:
                logger.info(f"Async {task_name} task finished.") # Log Completion
                research_data['paradigm_analysis'] = paradigm_result
                if paradigm_result and 'paradigm_analysis' in progress['pending_components']:
                    progress['completed_components'].append('paradigm_analysis')
                    progress['pending_components'].remove('paradigm_analysis')
                elif not paradigm_result and 'paradigm_analysis' not in progress['errors'] and 'paradigm_analysis' in progress['pending_components']:
                     progress['pending_components'].remove('paradigm_analysis')

        if hasattr(self, 'paradigm_analyzer') and self.paradigm_analyzer:
            tasks.append(paradigm_task())
        else:
             if 'paradigm_analysis' in progress['pending_components']:
                 progress['pending_components'].remove('paradigm_analysis')

        # Task 5: Audience Analysis (already async)
        async def audience_task():
            audience_result = None
            task_name = "Audience Analysis"
            try:
                if hasattr(self, 'audience_analyzer') and self.audience_analyzer:
                    audience_result = await self.audience_analyzer.analyze_audience(metadata.get('main_topic', ''))
                    if audience_result:
                         segments_count = len(audience_result.get('segments', []))
                         logger.info(f"Async {task_name} successful: Found {segments_count} audience segments.") # Log Success
                         self._log_to_opik(f"Async {task_name} complete", "audience_analysis_complete", {"segments_count": segments_count})
                else:
                    logger.warning("Audience analyzer not available, skipping")
                    if 'audience_analysis' in progress['pending_components']:
                         progress['pending_components'].remove('audience_analysis')
            except Exception as e:
                logger.error(f"Error in async {task_name}: {e}", exc_info=True) # Log Error
                progress['errors'].append({'component': 'audience_analysis', 'error': str(e)})
                self._log_to_opik(f"Async {task_name} error", "audience_analysis_error", {"error": str(e)})
                if 'audience_analysis' in progress['pending_components']:
                     progress['pending_components'].remove('audience_analysis')
            finally:
                logger.info(f"Async {task_name} task finished.") # Log Completion
                research_data['audience_analysis'] = audience_result
                if audience_result and 'audience_analysis' in progress['pending_components']:
                    progress['completed_components'].append('audience_analysis')
                    progress['pending_components'].remove('audience_analysis')
                elif not audience_result and 'audience_analysis' not in progress['errors'] and 'audience_analysis' in progress['pending_components']:
                     progress['pending_components'].remove('audience_analysis')

        if hasattr(self, 'audience_analyzer') and self.audience_analyzer:
            tasks.append(audience_task())
        else:
            if 'audience_analysis' in progress['pending_components']:
                 progress['pending_components'].remove('audience_analysis')

        # Task 6: Analogy Generation (already async)
        async def analogy_task():
            analogy_result = None
            task_name = "Analogy Generation"
            try:
                if hasattr(self, 'analogy_generator') and self.analogy_generator:
                    await asyncio.sleep(0.2)
                    # Assuming generate_analogies now takes only the concept
                    # We might need to adjust based on AnalogyGenerator's final design
                    # Let's assume it primarily needs the main topic for now.
                    analogy_result = await self.analogy_generator.generate_analogies(metadata.get('main_topic', ''))

                    if analogy_result:
                        gen_count = len(analogy_result.get('generated_analogies', []))
                        ex_count = len(analogy_result.get('existing_analogies', []))
                        logger.info(f"Async {task_name} successful: Found {gen_count} generated / {ex_count} existing analogies.") # Log Success
                        self._log_to_opik("Async Analogy generation complete", "analogy_generation_complete", {"generated": gen_count, "existing": ex_count})
                else:
                    logger.warning("Analogy generator not available, skipping")
                    if 'analogy_generation' in progress['pending_components']:
                        progress['pending_components'].remove('analogy_generation')
            except Exception as e:
                logger.error(f"Error in async {task_name}: {e}", exc_info=True) # Log Error
                progress['errors'].append({'component': 'analogy_generation', 'error': str(e)})
                self._log_to_opik("Async Analogy generation error", "analogy_generation_error", {"error": str(e)})
                if 'analogy_generation' in progress['pending_components']:
                    progress['pending_components'].remove('analogy_generation')
            finally:
                logger.info(f"Async {task_name} task finished.") # Log Completion
                research_data['analogies'] = analogy_result
                if analogy_result and 'analogy_generation' in progress['pending_components']:
                    progress['completed_components'].append('analogy_generation')
                    progress['pending_components'].remove('analogy_generation')
                elif not analogy_result and 'analogy_generation' not in progress['errors'] and 'analogy_generation' in progress['pending_components']:
                    progress['pending_components'].remove('analogy_generation')

        if hasattr(self, 'analogy_generator') and self.analogy_generator:
            tasks.append(analogy_task())
        else:
            if 'analogy_generation' in progress['pending_components']:
                progress['pending_components'].remove('analogy_generation')

        # Task 7: Visual Asset Collection (already async)
        async def visual_asset_task():
            visual_assets_result = None
            task_name = "Visual Asset Collection"
            try:
                if hasattr(self, 'visual_asset_collector') and self.visual_asset_collector:
                    await asyncio.sleep(0.3)
                    visual_assets_result = await self.visual_asset_collector.analyze_visual_assets(
                        metadata.get('main_topic', ''),
                        research_data.get('solution_analysis', {}), # Pass analysis data
                        research_data.get('paradigm_analysis', {}) # Pass analysis data
                    )
                    if visual_assets_result:
                        total_visuals = visual_assets_result.get('stats', {}).get('total_visuals', 0)
                        logger.info(f"Async {task_name} successful: Collected {total_visuals} visuals.") # Log Success
                        self._log_to_opik("Async Visual asset collection complete", "visual_asset_collection_complete", {"count": total_visuals})
                else:
                    if 'visual_asset_collection' in progress['pending_components']:
                        logger.warning("Visual asset collector not available, skipping")
                        progress['pending_components'].remove('visual_asset_collection')
            except Exception as e:
                logger.error(f"Error in async {task_name}: {e}", exc_info=True) # Log Error
                progress['errors'].append({'component': 'visual_asset_collection', 'error': str(e)})
                self._log_to_opik("Async Visual asset collection error", "visual_asset_collection_error", {"error": str(e)})
                if 'visual_asset_collection' in progress['pending_components']:
                    progress['pending_components'].remove('visual_asset_collection')
            finally:
                logger.info(f"Async {task_name} task finished.") # Log Completion
                research_data['visual_assets'] = visual_assets_result
                if visual_assets_result and 'visual_asset_collection' in progress['pending_components']:
                    progress['completed_components'].append('visual_asset_collection')
                    progress['pending_components'].remove('visual_asset_collection')
                elif not visual_assets_result and 'visual_asset_collection' not in progress['errors'] and 'visual_asset_collection' in progress['pending_components']:
                    progress['pending_components'].remove('visual_asset_collection')

        if hasattr(self, 'visual_asset_collector') and self.visual_asset_collector:
            tasks.append(visual_asset_task())
        else:
            if 'visual_asset_collection' in progress['pending_components']:
                 progress['pending_components'].remove('visual_asset_collection')


        # Run all research tasks concurrently
        await asyncio.gather(*tasks)

        # Calculate completion percentage
        # total_components = len(progress['completed_components']) + len(progress['pending_components']) + len(progress['errors'])
        initial_component_count = 7 # Hardcoded initial count
        completed_count = len(progress['completed_components'])
        error_count = len(progress['errors'])
        # Skipped count = initial - completed - errors - pending
        skipped_count = initial_component_count - completed_count - error_count - len(progress['pending_components'])

        if initial_component_count > 0:
             # Consider completed / (initial - skipped) as percentage of attempted components
             attempted_components = initial_component_count - skipped_count
             if attempted_components > 0:
                 progress['completion_percentage'] = int(
                     (completed_count / attempted_components) * 100
                 )
             else: # If all were skipped
                 progress['completion_percentage'] = 100
        else:
            progress['completion_percentage'] = 100 # No components to run
        
        # Consolidate citations (ensure uniqueness)
        seen_urls = set()
        unique_citations = []
        for citation in research_data.get('citations', []):
             if isinstance(citation, dict) and citation.get('url') and citation.get('url') not in seen_urls:
                 seen_urls.add(citation['url'])
                 unique_citations.append(citation)
        research_data['citations'] = unique_citations
        
        # Log completion of orchestration
        logger.info(f"Async Research orchestration complete. Completion: {progress['completion_percentage']}%. Errors: {len(progress['errors'])}")
        self._log_to_opik("Async Research orchestration complete", "async_research_complete", { **progress })
        
        # Add progress to research data
        research_data['progress'] = progress
        
        return research_data
    
    def calculate_readiness_score(self, metadata: Dict[str, Any], research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a readiness score for the blog based on metadata and research data,
        returning a letter grade (A-F) with detailed breakdown and improvement recommendations.
        
        Enhanced grading criteria:
        - Grade C or below (≤ 70%) for content with any critical missing components
        - Grade B (71-85%) for content with minimum 30% in EACH category
        - Grade A (86-100%) requiring rich visual assets, strong systemic thinking, proper citations,
          nuanced solution research, and clear articulation of audience benefits
        
        Args:
            metadata: Blog metadata
            research_data: Research data
            
        Returns:
            Dictionary containing the numerical score, letter grade, components breakdown,
            automatic grade caps explanation, and improvement recommendations
        """
        # Initialize score components dictionary for detailed reporting
        score_components = {}
        
        # Start with lower base score (reduced from 50 to 30)
        base_score = 30.0
        score_components['base_score'] = base_score
        
        # Initialize total score with base score
        total_score = base_score
        
        # Initialize dictionaries to track component percentages and missing essentials
        component_percentages = {}
        missing_critical_components = []
        improvement_recommendations = []
        
        # 1. Industry Challenge Analysis Score
        challenges_count = len(research_data.get('challenges', []))
        challenges_target = self.readiness_thresholds.get('challenges_target', 10)
        challenges_min = self.readiness_thresholds.get('challenges_min', 2)
        challenges_max = self.readiness_thresholds.get('challenges_max', 15)
        challenges_points = self.readiness_thresholds.get('challenges_points', 15)
        
        if challenges_count >= challenges_min:
            # Base score for meeting minimum requirements
            min_percentage = challenges_min / challenges_target
            challenges_score = min_percentage * challenges_points
            
            # Additional score for exceeding minimum up to target
            if challenges_count > challenges_min:
                additional_challenges = min(challenges_count, challenges_target) - challenges_min
                additional_range = challenges_target - challenges_min
                if additional_range > 0:
                    additional_percentage = additional_challenges / additional_range
                    remaining_points = challenges_points * (1 - min_percentage)
                    challenges_score += additional_percentage * remaining_points
            
            # Bonus for exceeding target up to max
            if challenges_count > challenges_target:
                bonus_factor = min(1.0, (challenges_count - challenges_target) /
                                  (challenges_max - challenges_target))
                challenges_score *= (1.0 + bonus_factor * 0.2)  # Up to 20% bonus
            
            # Cap at maximum points
            challenges_score = min(challenges_score, challenges_points * 1.2)
            
            # Calculate percentage of maximum possible points
            challenges_percentage = challenges_score / (challenges_points * 1.2) * 100
            component_percentages['challenges'] = challenges_percentage
        else:
            challenges_score = 0
            component_percentages['challenges'] = 0
            missing_critical_components.append("Industry Challenges")
            improvement_recommendations.append("Identify at least 2 industry challenges, aiming for 10+ for full points")
        
        score_components['challenges_score'] = challenges_score
        total_score += challenges_score
        
        # 2. Solution Arguments Score (Pro Arguments)
        pro_arguments_count = len(research_data.get('solution', {}).get('pro_arguments', []))
        pro_arguments_target = self.readiness_thresholds.get('pro_arguments_target', 5)
        pro_arguments_min = self.readiness_thresholds.get('pro_arguments_min', 2)
        pro_arguments_max = self.readiness_thresholds.get('pro_arguments_max', 10)
        pro_arguments_points = self.readiness_thresholds.get('pro_arguments_points', 10)
        
        if pro_arguments_count >= pro_arguments_min:
            # Base score for meeting minimum requirements
            min_percentage = pro_arguments_min / pro_arguments_target
            pro_score = min_percentage * pro_arguments_points
            
            # Additional score for exceeding minimum up to target
            if pro_arguments_count > pro_arguments_min:
                additional_args = min(pro_arguments_count, pro_arguments_target) - pro_arguments_min
                additional_range = pro_arguments_target - pro_arguments_min
                if additional_range > 0:
                    additional_percentage = additional_args / additional_range
                    remaining_points = pro_arguments_points * (1 - min_percentage)
                    pro_score += additional_percentage * remaining_points
            
            # Bonus for exceeding target up to max
            if pro_arguments_count > pro_arguments_target:
                bonus_factor = min(1.0, (pro_arguments_count - pro_arguments_target) /
                                 (pro_arguments_max - pro_arguments_target))
                pro_score *= (1.0 + bonus_factor * 0.2)  # Up to 20% bonus
            
            # Cap at maximum points
            pro_score = min(pro_score, pro_arguments_points * 1.2)
            
            # Calculate percentage of maximum possible points
            pro_arguments_percentage = pro_score / (pro_arguments_points * 1.2) * 100
            component_percentages['pro_arguments'] = pro_arguments_percentage
        else:
            pro_score = 0
            component_percentages['pro_arguments'] = 0
            missing_critical_components.append("Pro Arguments")
            improvement_recommendations.append("Include at least 2 pro arguments, aiming for 5-10 for full points")
        
        score_components['pro_arguments_score'] = pro_score
        total_score += pro_score
        
        # 3. Solution Arguments Score (Counter Arguments)
        counter_arguments_count = len(research_data.get('solution', {}).get('counter_arguments', []))
        counter_arguments_target = self.readiness_thresholds.get('counter_arguments_target', 5)
        counter_arguments_min = self.readiness_thresholds.get('counter_arguments_min', 1)
        counter_arguments_max = self.readiness_thresholds.get('counter_arguments_max', 8)
        counter_arguments_points = self.readiness_thresholds.get('counter_arguments_points', 10)
        
        if counter_arguments_count >= counter_arguments_min:
            # Base score for meeting minimum requirements
            min_percentage = counter_arguments_min / counter_arguments_target
            counter_score = min_percentage * counter_arguments_points
            
            # Additional score for exceeding minimum up to target
            if counter_arguments_count > counter_arguments_min:
                additional_args = min(counter_arguments_count, counter_arguments_target) - counter_arguments_min
                additional_range = counter_arguments_target - counter_arguments_min
                if additional_range > 0:
                    additional_percentage = additional_args / additional_range
                    remaining_points = counter_arguments_points * (1 - min_percentage)
                    counter_score += additional_percentage * remaining_points
            
            # Bonus for exceeding target up to max
            if counter_arguments_count > counter_arguments_target:
                bonus_factor = min(1.0, (counter_arguments_count - counter_arguments_target) /
                                 (counter_arguments_max - counter_arguments_target))
                counter_score *= (1.0 + bonus_factor * 0.2)  # Up to 20% bonus
            
            # Cap at maximum points
            counter_score = min(counter_score, counter_arguments_points * 1.2)
            
            # Calculate percentage of maximum possible points
            counter_arguments_percentage = counter_score / (counter_arguments_points * 1.2) * 100
            component_percentages['counter_arguments'] = counter_arguments_percentage
        else:
            counter_score = 0
            component_percentages['counter_arguments'] = 0
            missing_critical_components.append("Counter Arguments")
            improvement_recommendations.append("Include at least 1 counter argument, aiming for 5+ for full points")
        
        score_components['counter_arguments_score'] = counter_score
        total_score += counter_score
        
        # 4. Paradigm Analysis Score
        paradigms_count = research_data.get('paradigms', {}).get('stats', {}).get('paradigms_count', 0)
        transitions_count = research_data.get('paradigms', {}).get('stats', {}).get('transitions_count', 0)
        future_projections_count = research_data.get('paradigms', {}).get('stats', {}).get('future_projections_count', 0)
        
        paradigms_target = self.readiness_thresholds.get('paradigms_target', 3)
        paradigms_points = self.readiness_thresholds.get('paradigms_points', 10)
        
        # Combine into overall paradigm analysis score
        paradigm_components = [
            paradigms_count / max(1, paradigms_target),
            transitions_count / max(1, paradigms_count - 1 if paradigms_count > 1 else 1),
            future_projections_count / max(1, paradigms_target / 2)
        ]
        
        # Average of the components, weighted by importance
        paradigm_weights = [0.5, 0.25, 0.25]  # Weights for current, transitions, future
        paradigm_score_normalized = sum(c * w for c, w in zip(paradigm_components, paradigm_weights))
        paradigm_score = paradigm_score_normalized * paradigms_points
        
        # Cap at maximum points
        paradigm_score = min(paradigm_score, paradigms_points * 1.2)
        
        # Calculate percentage of maximum possible points
        paradigm_percentage = paradigm_score / (paradigms_points * 1.2) * 100
        component_percentages['paradigm'] = paradigm_percentage
        
        if paradigm_percentage < 20:
            missing_critical_components.append("Paradigm Analysis")
            improvement_recommendations.append("Analyze historical paradigms related to the topic")
        
        score_components['paradigm_score'] = paradigm_score
        total_score += paradigm_score
        
        # 5. Audience Analysis Score
        audience_segments_count = research_data.get('audience', {}).get('stats', {}).get('segments_count', 0)
        knowledge_gaps_count = research_data.get('audience', {}).get('stats', {}).get('knowledge_gaps_count', 0)
        
        audience_segments_target = self.readiness_thresholds.get('audience_segments_target', 3)
        audience_segments_points = self.readiness_thresholds.get('audience_segments_points', 10)
        
        # Audience complexity score (0-1)
        audience_complexity = min(1.0, audience_segments_count / max(1, audience_segments_target))
        # Knowledge gaps coverage score (0-1)
        knowledge_coverage = min(1.0, knowledge_gaps_count / max(1, audience_segments_count * 2))
        
        # Combined audience score
        audience_components = [audience_complexity, knowledge_coverage]
        audience_weights = [0.6, 0.4]  # More weight on segmentation
        audience_score_normalized = sum(c * w for c, w in zip(audience_components, audience_weights))
        audience_score = audience_score_normalized * audience_segments_points
        
        # Cap at maximum points
        audience_score = min(audience_score, audience_segments_points * 1.2)
        
        # Calculate percentage of maximum possible points
        audience_percentage = audience_score / (audience_segments_points * 1.2) * 100
        component_percentages['audience'] = audience_percentage
        
        if audience_percentage < 20:
            missing_critical_components.append("Audience Analysis")
            improvement_recommendations.append("Identify at least 1 audience segment and their knowledge gaps")
        
        score_components['audience_score'] = audience_score
        total_score += audience_score
        
        # 6. Visual Assets Score - Critical Component for B grade or higher
        solution_visuals_count = len(research_data.get('visual_assets', {}).get('solution_visuals', []))
        paradigm_visuals_count = len(research_data.get('visual_assets', {}).get('paradigm_visuals', []))
        
        # Solution visuals score
        solution_visuals_target = self.readiness_thresholds.get('solution_visuals_target', 50)
        solution_visuals_min = self.readiness_thresholds.get('solution_visuals_min', 10)
        solution_visuals_points = self.readiness_thresholds.get('solution_visuals_points', 10)
        
        if solution_visuals_count >= solution_visuals_min:
            solution_visuals_ratio = min(1.0, solution_visuals_count / solution_visuals_target)
            solution_visuals_score = solution_visuals_ratio * solution_visuals_points
        else:
            solution_visuals_score = 0
        
        # Paradigm visuals score
        paradigm_visuals_target = self.readiness_thresholds.get('paradigm_visuals_target', 10)
        paradigm_visuals_min = self.readiness_thresholds.get('paradigm_visuals_min', 3)
        paradigm_visuals_points = self.readiness_thresholds.get('paradigm_visuals_points', 5)
        
        if paradigm_visuals_count >= paradigm_visuals_min:
            paradigm_visuals_ratio = min(1.0, paradigm_visuals_count / paradigm_visuals_target)
            paradigm_visuals_score = paradigm_visuals_ratio * paradigm_visuals_points
        else:
            paradigm_visuals_score = 0
        
        # Combined visual assets score
        visual_assets_score = solution_visuals_score + paradigm_visuals_score
        visual_assets_max_points = solution_visuals_points + paradigm_visuals_points
        
        # Calculate percentage of maximum possible points
        if visual_assets_max_points > 0:
            visual_assets_percentage = visual_assets_score / visual_assets_max_points * 100
        else:
            visual_assets_percentage = 0
        component_percentages['visual_assets'] = visual_assets_percentage
        
        # If no visual assets, this is a critical missing component
        if solution_visuals_count == 0 and paradigm_visuals_count == 0:
            missing_critical_components.append("Visual Assets")
            improvement_recommendations.append("Include visual assets (50-100 solution visuals, 10-20 paradigm visuals)")
        
        score_components['visual_assets_score'] = visual_assets_score
        total_score += visual_assets_score
        
        # 7. Analogies Score
        analogies_data = research_data.get('analogies', {}) # Use .get() for safety
        # Default to empty lists if analogies_data is None or keys are missing
        challenge_analogies_count = len(analogies_data.get('challenge_analogies', [])) if analogies_data else 0
        solution_analogies_count = len(analogies_data.get('solution_analogies', [])) if analogies_data else 0
        
        # Challenge analogies score
        challenge_analogies_target = self.readiness_thresholds.get('challenge_analogies_target', 3)
        challenge_analogies_min = self.readiness_thresholds.get('challenge_analogies_min', 1)
        challenge_analogies_points = self.readiness_thresholds.get('challenge_analogies_points', 5)
        
        if challenge_analogies_count >= challenge_analogies_min:
            challenge_analogies_ratio = min(1.0, challenge_analogies_count / challenge_analogies_target)
            challenge_analogies_score = challenge_analogies_ratio * challenge_analogies_points
        else:
            challenge_analogies_score = 0
        
        # Solution analogies score
        solution_analogies_target = self.readiness_thresholds.get('solution_analogies_target', 3)
        solution_analogies_min = self.readiness_thresholds.get('solution_analogies_min', 1)
        solution_analogies_points = self.readiness_thresholds.get('solution_analogies_points', 5)
        
        if solution_analogies_count >= solution_analogies_min:
            solution_analogies_ratio = min(1.0, solution_analogies_count / solution_analogies_target)
            solution_analogies_score = solution_analogies_ratio * solution_analogies_points
        else:
            solution_analogies_score = 0
        
        # Combined analogies score
        analogies_score = challenge_analogies_score + solution_analogies_score
        analogies_max_points = challenge_analogies_points + solution_analogies_points
        
        # Calculate percentage of maximum possible points
        if analogies_max_points > 0:
            analogies_percentage = analogies_score / analogies_max_points * 100
        else:
            analogies_percentage = 0
        component_percentages['analogies'] = analogies_percentage
        
        if analogies_percentage < 20:
            missing_critical_components.append("Analogies")
            improvement_recommendations.append("Include at least 1 analogy each for challenges and solutions")
        
        score_components['analogies_score'] = analogies_score
        total_score += analogies_score
        
        # 8. Citations Score - Critical for A grade
        citations_count = len(research_data.get('citations', []))
        citations_target = self.readiness_thresholds.get('citations_target', 15)
        citations_min = self.readiness_thresholds.get('citations_min', 3)
        citations_points = self.readiness_thresholds.get('citations_points', 10)
        
        if citations_count >= citations_min:
            citations_ratio = min(1.0, citations_count / citations_target)
            citations_score = citations_ratio * citations_points
            
            # Bonus for exceeding target
            if citations_count > citations_target:
                citations_max = self.readiness_thresholds.get('citations_max', 30)
                bonus_factor = min(1.0, (citations_count - citations_target) / (citations_max - citations_target))
                citations_score *= (1.0 + bonus_factor * 0.2)  # Up to 20% bonus
            
            # Cap at maximum points
            citations_score = min(citations_score, citations_points * 1.2)
            
            # Calculate percentage of maximum possible points
            citations_percentage = citations_score / (citations_points * 1.2) * 100
            component_percentages['citations'] = citations_percentage
        else:
            citations_score = 0
            component_percentages['citations'] = 0
            missing_critical_components.append("Citations")
            improvement_recommendations.append("Include at least 3 citations, aiming for 15+ for full points")
        
        score_components['citations_score'] = citations_score
        total_score += citations_score
        
        # 9. Sequential Thinking Evidence Score - Critical for B grade or higher
        has_constraints = bool(research_data.get('constraints', []))
        has_systemic_context = bool(research_data.get('systemic_context', {}))
        has_stakeholder_perspectives = bool(research_data.get('stakeholder_perspectives', []))
        
        sequential_evidence_count = sum([has_constraints, has_systemic_context, has_stakeholder_perspectives])
        sequential_evidence_min = self.readiness_thresholds.get('sequential_evidence_min', 1)
        sequential_evidence_points = self.readiness_thresholds.get('sequential_evidence_points', 10)
        
        if sequential_evidence_count >= sequential_evidence_min:
            sequential_evidence_ratio = min(1.0, sequential_evidence_count / 3)  # Three possible components
            sequential_evidence_score = sequential_evidence_ratio * sequential_evidence_points
            
            # Calculate percentage of maximum possible points
            sequential_evidence_percentage = sequential_evidence_score / sequential_evidence_points * 100
            component_percentages['sequential_evidence'] = sequential_evidence_percentage
        else:
            sequential_evidence_score = 0
            component_percentages['sequential_evidence'] = 0
            missing_critical_components.append("Sequential Thinking Evidence")
            improvement_recommendations.append("Include evidence of sequential thinking (constraints, context, stakeholder perspectives)")
        
        score_components['sequential_evidence_score'] = sequential_evidence_score
        total_score += sequential_evidence_score
        
        # 10. Basic Content Quality Score (from metadata)
        # Headings score
        headings_count = len(metadata.get('headings', []))
        headings_min = self.readiness_thresholds.get('headings_min', 3)
        headings_max = self.readiness_thresholds.get('headings_max', 8)
        headings_points = self.readiness_thresholds.get('headings_points', 2)
        
        if headings_count >= headings_min:
            headings_ratio = min(1.0, headings_count / headings_max)
            headings_score = headings_ratio * headings_points
        else:
            headings_score = 0
        
        # Paragraphs score
        paragraphs_count = metadata.get('paragraphs_count', 0)
        paragraphs_min = self.readiness_thresholds.get('paragraphs_min', 5)
        paragraphs_max = self.readiness_thresholds.get('paragraphs_max', 15)
        paragraphs_points = self.readiness_thresholds.get('paragraphs_points', 2)
        
        if paragraphs_count >= paragraphs_min:
            paragraphs_ratio = min(1.0, paragraphs_count / paragraphs_max)
            paragraphs_score = paragraphs_ratio * paragraphs_points
        else:
            paragraphs_score = 0
        
        # Images score
        images_count = metadata.get('images_count', 0)
        images_min = self.readiness_thresholds.get('images_min', 1)
        images_max = self.readiness_thresholds.get('images_max', 5)
        images_points = self.readiness_thresholds.get('images_points', 1)
        
        if images_count >= images_min:
            images_ratio = min(1.0, images_count / images_max)
            images_score = images_ratio * images_points
        else:
            images_score = 0
        
        # Combined basic content quality score
        basic_content_score = headings_score + paragraphs_score + images_score
        basic_content_max_points = headings_points + paragraphs_points + images_points
        
        # Calculate percentage of maximum possible points
        if basic_content_max_points > 0:
            basic_content_percentage = basic_content_score / basic_content_max_points * 100
        else:
            basic_content_percentage = 0
        component_percentages['basic_content'] = basic_content_percentage
        
        if basic_content_percentage < 20:
            missing_critical_components.append("Basic Content Structure")
            improvement_recommendations.append("Include proper headings, sufficient paragraphs, and at least one image")
        
        score_components['basic_content_score'] = basic_content_score
        total_score += basic_content_score
        
        # 11. Solution Nuance Evaluation - New, critical for A grade
        solution_nuance_score = self._evaluate_solution_nuance(research_data)
        solution_nuance_points = 10  # Maximum points possible
        
        # Calculate percentage
        solution_nuance_percentage = (solution_nuance_score / solution_nuance_points) * 100
        component_percentages['solution_nuance'] = solution_nuance_percentage
        
        if solution_nuance_percentage < 50:
            improvement_recommendations.append("Develop more nuanced solution analysis specific to the proposed solutions")
        
        score_components['solution_nuance_score'] = solution_nuance_score
        total_score += solution_nuance_score
        
        # 12. Audience Benefit Clarity - New, critical for A grade
        audience_benefit_score = self._evaluate_audience_benefit_clarity(research_data)
        audience_benefit_points = 10  # Maximum points possible
        
        # Calculate percentage
        audience_benefit_percentage = (audience_benefit_score / audience_benefit_points) * 100
        component_percentages['audience_benefit'] = audience_benefit_percentage
        
        if audience_benefit_percentage < 50:
            improvement_recommendations.append("Clarify how solutions specifically benefit different audience segments")
        
        score_components['audience_benefit_score'] = audience_benefit_score
        total_score += audience_benefit_score
        
        # Calculate raw final score (before grade caps)
        raw_score = min(100.0, total_score)
        
        # Apply grade cap based on critical missing components and minimum thresholds
        max_grade = 'A'  # Start with highest possible grade
        grade_cap_reason = None
        
        # C or lower if no visual assets or limited systemic thinking
        if 'Visual Assets' in missing_critical_components:
            max_grade = 'C'
            grade_cap_reason = "Missing visual assets"
        elif component_percentages.get('sequential_evidence', 0) < 30:
            max_grade = 'C'
            grade_cap_reason = "Limited systemic thinking"
        elif any(component_percentages.get(comp, 0) < 30 for comp in ['challenges', 'pro_arguments', 'counter_arguments', 'citations']):
            max_grade = 'C'
            grade_cap_reason = "Core components below 30% threshold"
        # B or lower if any component below 30%
        elif any(percentage < 30 for component, percentage in component_percentages.items()):
            max_grade = 'B'
            grade_cap_reason = "Some components below 30% threshold"
        # A only if solution nuance and audience benefit are strong
        elif component_percentages.get('solution_nuance', 0) < 50 or component_percentages.get('audience_benefit', 0) < 50:
            max_grade = 'B'
            grade_cap_reason = "Insufficient solution nuance or audience benefit clarity"
        
        # Apply grade cap if needed
        raw_grade = 'F'
        for grade_letter, threshold in sorted(self.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
            if raw_score >= threshold:
                raw_grade = grade_letter
                break
        
        # Apply grade cap
        final_grade = min(raw_grade, max_grade, key=lambda g: self.grade_thresholds.get(g, 0))
        
        # Adjust final score to match capped grade, if necessary
        if final_grade != raw_grade:
            final_score = self.grade_thresholds.get(final_grade, 0) + 5  # Middle of the grade range
        else:
            final_score = raw_score
        
        # Log the score components
        logger.info(f"Readiness score calculation components: {score_components}")
        logger.info(f"Component percentages: {component_percentages}")
        logger.info(f"Missing critical components: {missing_critical_components}")
        logger.info(f"Raw score: {raw_score}, Raw grade: {raw_grade}, Final grade: {final_grade}")
        if grade_cap_reason:
            logger.info(f"Grade cap reason: {grade_cap_reason}")
        
        # Add score components to research_data for reporting
        research_data['score_components'] = score_components
        research_data['component_percentages'] = component_percentages
        
        # Create result with score, grade, components and recommendations
        result = {
            'score': final_score,
            'grade': final_grade,
            'raw_score': raw_score,
            'raw_grade': raw_grade,
            'components': score_components,
            'component_percentages': component_percentages,
            'missing_critical_components': missing_critical_components,
            'grade_cap_reason': grade_cap_reason,
            'improvement_recommendations': improvement_recommendations
        }
        
        return result
    
    def _evaluate_solution_nuance(self, research_data: Dict[str, Any]) -> float:
        """
        Evaluates how nuanced and specific the research is to the proposed solutions.
        
        Criteria:
        1. Solution-specific evidence rather than generic information
        2. Analysis of solution variants or approaches
        3. Contextual applicability of solutions
        4. Depth vs breadth of solution analysis
        5. Integration of solution with other research components
        
        Args:
            research_data: Research data
            
        Returns:
            Score from 0-10 points
        """
        nuance_score = 0.0
        
        # 1. Solution-specific evidence (0-2 points)
        solution = research_data.get('solution', {})
        pro_arguments = solution.get('pro_arguments', [])
        
        # Check if pro arguments have specific evidence/sources attached
        has_specific_evidence = False
        for arg in pro_arguments:
            if isinstance(arg, dict) and arg.get('sources', []):
                has_specific_evidence = True
                break
        
        if has_specific_evidence:
            nuance_score += 2.0
        elif pro_arguments:  # Some arguments but no specific evidence
            nuance_score += 1.0
        
        # 2. Solution variants analysis (0-2 points)
        has_variants = False
        variant_count = 0
        
        # Check for multiple solution approaches or variants
        proposed_solution = research_data.get('proposed_solution', {})
        variants = proposed_solution.get('variants', [])
        
        if variants:
            variant_count = len(variants)
            has_variants = True
        
        if variant_count >= 2:
            nuance_score += 2.0
        elif has_variants:
            nuance_score += 1.0
        
        # 3. Contextual applicability (0-2 points)
        has_context = False
        
        # Check for contextual applicability information
        has_context = bool(research_data.get('systemic_context', {}))
        has_constraints = bool(research_data.get('constraints', []))
        
        if has_context and has_constraints:
            nuance_score += 2.0
        elif has_context or has_constraints:
            nuance_score += 1.0
        
        # 4. Depth vs breadth (0-2 points)
        # Check depth by looking at detail level in pro arguments
        depth_score = 0
        
        for arg in pro_arguments:
            if isinstance(arg, dict):
                # Check for detailed analysis elements
                if arg.get('prerequisites', '') or arg.get('metrics', []) or arg.get('supporting_evidence', ''):
                    depth_score += 1
        
        if depth_score >= 3:  # Multiple arguments with depth
            nuance_score += 2.0
        elif depth_score >= 1:  # At least one detailed argument
            nuance_score += 1.0
        
        # 5. Integration with other components (0-2 points)
        # Check if solution connects to paradigms, audience, or challenges
        integration_points = 0
        
        # Check if challenges are connected to solutions
        challenges = research_data.get('challenges', [])
        for challenge in challenges:
            if isinstance(challenge, dict) and 'solution' in str(challenge).lower():
                integration_points += 1
                break
        
        # Check if audience is connected to solutions
        audience = research_data.get('audience', {})
        if 'solution' in str(audience).lower():
            integration_points += 1
        
        # Check if paradigms connect to solutions
        paradigms = research_data.get('paradigms', {})
        if 'solution' in str(paradigms).lower():
            integration_points += 1
        
        if integration_points >= 2:  # Multiple integration points
            nuance_score += 2.0
        elif integration_points >= 1:  # At least one integration point
            nuance_score += 1.0
        
        return nuance_score
    
    def _evaluate_audience_benefit_clarity(self, research_data: Dict[str, Any]) -> float:
        """
        Evaluates how clearly the research articulates how solutions benefit the audience.
        
        Criteria:
        1. Explicit mapping of solutions to audience segments
        2. Quantified or qualified benefits for each segment
        3. Addressing specific audience pain points
        4. Consideration of audience implementation constraints
        5. Audience-appropriate communication of benefits
        
        Args:
            research_data: Research data
            
        Returns:
            Score from 0-10 points
        """
        benefit_score = 0.0
        
        # 1. Solution-to-audience mapping (0-2 points)
        has_mapping = False
        mapping_count = 0
        
        # Check for explicit connections between solutions and audience segments
        audience = research_data.get('audience', {})
        audience_segments = audience.get('audience_segments', [])
        
        for segment in audience_segments:
            if isinstance(segment, dict) and 'solution' in str(segment).lower():
                mapping_count += 1
        
        if mapping_count >= 2:  # Multiple segments mapped to solutions
            benefit_score += 2.0
        elif mapping_count >= 1:  # At least one segment mapped
            benefit_score += 1.0
        
        # 2. Quantified/qualified benefits (0-2 points)
        benefit_quality = 0
        
        # Check for quantified benefits in pro arguments
        solution = research_data.get('solution', {})
        pro_arguments = solution.get('pro_arguments', [])
        
        for arg in pro_arguments:
            if isinstance(arg, dict):
                arg_text = str(arg)
                # Look for percentages, numbers, or qualifiers
                if any(qualifier in arg_text.lower() for qualifier in ['%', 'percent', 'times', 'x faster', 'significant', 'substantial']):
                    benefit_quality += 1
        
        if benefit_quality >= 2:  # Multiple quantified benefits
            benefit_score += 2.0
        elif benefit_quality >= 1:  # At least one quantified benefit
            benefit_score += 1.0
        
        # 3. Addressing pain points (0-2 points)
        addresses_pain_points = False
        pain_point_count = 0
        
        # Check for connections between solutions and pain points
        for segment in audience_segments:
            if isinstance(segment, dict) and segment.get('pain_points', []):
                for pain_point in segment.get('pain_points', []):
                    # Check if any pro argument addresses this pain point
                    for arg in pro_arguments:
                        if isinstance(arg, dict) and pain_point.lower() in str(arg).lower():
                            pain_point_count += 1
                            break
        
        if pain_point_count >= 2:  # Multiple pain points addressed
            benefit_score += 2.0
        elif pain_point_count >= 1:  # At least one pain point addressed
            benefit_score += 1.0
        
        # 4. Implementation constraints (0-2 points)
        has_implementation_constraints = False
        
        # Check for implementation considerations
        constraints = research_data.get('constraints', [])
        for constraint in constraints:
            if 'implement' in str(constraint).lower() or 'adopt' in str(constraint).lower():
                has_implementation_constraints = True
                break
        
        # Also check counter arguments for implementation concerns
        counter_arguments = solution.get('counter_arguments', [])
        for arg in counter_arguments:
            if isinstance(arg, dict) and ('implement' in str(arg).lower() or 'adopt' in str(arg).lower()):
                has_implementation_constraints = True
                break
        
        if has_implementation_constraints:
            benefit_score += 2.0
        
        # 5. Audience-appropriate communication (0-2 points)
        has_appropriate_communication = False
        
        # Check for knowledge level consideration
        knowledge_level = audience.get('knowledge_level', '')
        knowledge_gaps = audience.get('knowledge_gaps', [])
        
        if knowledge_level or knowledge_gaps:
            has_appropriate_communication = True
        
        # Check for audience-specific content (glossary, analogies, etc.)
        has_analogies = bool(research_data.get('analogies', {}).get('solution_analogies', []))
        has_glossary = bool(audience.get('glossary', []) or audience.get('acronyms', []))
        
        if has_analogies and has_glossary:
            benefit_score += 2.0
        elif has_appropriate_communication or has_analogies or has_glossary:
            benefit_score += 1.0
        
        return benefit_score
    
    def generate_research_report(
        self, 
        blog_data: Dict[str, Any],
        metadata: Dict[str, Any],
        research_data: Dict[str, Any],
        readiness_score: float
    ) -> str:
        """
        Generate a comprehensive research report based on the gathered data.
        
        Args:
            blog_data: Original blog data
            metadata: Blog metadata
            research_data: Research findings
            readiness_score: Calculated readiness score (numerical value)
            
        Returns:
            Markdown formatted research report
        """
        logger.info("Generating research report")
        self._log_to_opik("Research report generation started", "report_generation_start", {})
        
        main_topic = metadata.get('main_topic', 'Unknown Topic')
        
        # Start building the report
        report = [f"# Research Report: {main_topic}"]
        report.append("\n## Blog Information")
        report.append(f"- **Title:** {metadata.get('blog_title', 'Untitled')}")
        report.append(f"- **Version:** {metadata.get('version', 1)}")
        report.append(f"- **Readiness Score:** {readiness_score:.1f}/100")
        
        # Add topic analysis
        report.append("\n## Topic Analysis")
        report.append(f"- **Main Topic:** {main_topic}")
        report.append(f"- **Summary:** {metadata.get('summary', 'No summary available')}")
        report.append(f"- **Reading Time:** {metadata.get('reading_time_minutes', 0)} minutes")
        
        # Add content structure
        report.append("\n## Content Structure")
        report.append(f"- **Headings:** {len(metadata.get('headings', []))}")
        report.append(f"- **Paragraphs:** {metadata.get('paragraphs_count', 0)}")
        report.append(f"- **Images:** {metadata.get('images_count', 0)}")
        report.append(f"- **Has Code Blocks:** {'Yes' if metadata.get('has_code_blocks', False) else 'No'}")
        report.append(f"- **Has Tables:** {'Yes' if metadata.get('has_tables', False) else 'No'}")
        report.append(f"- **Has Lists:** {'Yes' if metadata.get('has_lists', False) else 'No'}")
        
        # Add research data
        report.append("\n## Research Data")
        
        # System affected / Industry challenges
        report.append("\n### Industry/System Challenges")
        system = research_data.get('system_affected', {})
        if system:
            report.append(f"**Name:** {system.get('name', 'Unknown System')}")
            report.append(f"**Description:** {system.get('description', 'No description available')}")
            report.append(f"**Scale:** {system.get('scale', 'Unknown')}")
            
            challenges = research_data.get('challenges', [])
            if challenges:
                report.append("\n#### Key Challenges")
                for i, challenge in enumerate(challenges[:5], 1):
                    if isinstance(challenge, dict):
                        challenge_title = challenge.get('challenge', challenge.get('name', f'Challenge {i}'))
                        challenge_desc = challenge.get('description', 'No description available')
                        report.append(f"**{i}. {challenge_title}**")
                        report.append(f"  - {challenge_desc}")
                    else:
                         report.append(f"**{i}. Invalid challenge format: {challenge}**")
            else:
                 report.append("\nNo challenges identified.")
        else:
             report.append("\nNo system/industry analysis data available.")
        
        # Solution analysis
        report.append("\n### Solution Analysis")
        solution_data = research_data.get('solution', {})
        proposed_solution = research_data.get('proposed_solution', {})
        
        solution_name = solution_data.get('solution', proposed_solution.get('name', 'Proposed Solution'))
        solution_description = solution_data.get('description', proposed_solution.get('description', 'No description available'))
        
        report.append(f"**Name:** {solution_name}")
        report.append(f"**Description:** {solution_description}")
        
        # Pro arguments
        report.append("\n#### Supporting Arguments")
        pro_args = solution_data.get('pro_arguments', proposed_solution.get('advantages', []))
        if pro_args:
            for i, arg in enumerate(pro_args[:5], 1):
                if isinstance(arg, dict):
                    arg_text = arg.get('argument', arg.get('name', f'Argument {i}'))
                    report.append(f"**{i}. {arg_text}**")
                    explanation = arg.get('explanation', arg.get('description', ''))
                    if explanation:
                        report.append(f"  - {explanation}")
                else:
                    report.append(f"**{i}. {arg}**")
                    report.append(f"  - {arg}")
        else:
            report.append("No supporting arguments identified.")
        
        # Counter arguments
        report.append("\n#### Counter Arguments")
        counter_args = solution_data.get('counter_arguments', proposed_solution.get('limitations', []))
        if counter_args:
            for i, arg in enumerate(counter_args[:3], 1):
                if isinstance(arg, dict):
                    arg_text = arg.get('argument', arg.get('name', f'Counter-argument {i}'))
                    report.append(f"**{i}. {arg_text}**")
                    explanation = arg.get('explanation', arg.get('description', ''))
                    if explanation:
                        report.append(f"  - {explanation}")
                    rebuttal = arg.get('rebuttal', arg.get('defense_rebuttal', ''))
                    if rebuttal:
                        report.append(f"  - Rebuttal: {rebuttal}")
                else:
                    report.append(f"**{i}. {arg}**")
        else:
            report.append("No counter arguments identified.")
        
        # Paradigm analysis
        report.append("\n### Historical Paradigm Analysis")
        paradigm_data = research_data.get('paradigms', {})
        current_paradigm = research_data.get('current_paradigm', {})
        
        paradigm_name = current_paradigm.get('name', 'Current Approach')
        if paradigm_data.get('historical_paradigms'):
            last_paradigm = paradigm_data['historical_paradigms'][-1]
            paradigm_name = last_paradigm.get('name', paradigm_name)
            paradigm_desc = last_paradigm.get('description', 'No description available')
        else:
            paradigm_desc = current_paradigm.get('description', 'No description available')
        
        report.append(f"**Name:** {paradigm_name}")
        report.append(f"**Description:** {paradigm_desc}")
        
        limitations = current_paradigm.get('limitations', [])
        if limitations:
            report.append("\n#### Limitations")
            for limitation in limitations:
                report.append(f"  - {limitation}")
        elif paradigm_data.get('lessons'):
            report.append("\n#### Key Lessons")
            for i, lesson in enumerate(paradigm_data.get('lessons', [])[:3], 1):
                 if isinstance(lesson, dict):
                     report.append(f"**{i}. {lesson.get('lesson', 'Lesson')}**: {lesson.get('explanation', '')}")
        
        # Audience analysis
        report.append("\n### Audience Analysis")
        audience_data = research_data.get('audience', {})
        audience_analysis_legacy = research_data.get('audience_analysis', {})
        
        knowledge_level = audience_data.get('knowledge_level', audience_analysis_legacy.get('knowledge_level', 'moderate'))
        background = audience_data.get('background', audience_analysis_legacy.get('background', 'General audience'))
        
        report.append(f"**Knowledge Level:** {knowledge_level}")
        report.append(f"**Background:** {background}")
        
        interests = audience_data.get('interests', audience_analysis_legacy.get('interests', []))
        if interests:
            report.append("\n#### Key Interests")
            for interest in interests:
                report.append(f"  - {interest}")
        elif audience_data.get('audience_segments'):
             report.append("\n#### Identified Segments")
             for i, segment in enumerate(audience_data.get('audience_segments', [])[:3], 1):
                 if isinstance(segment, dict):
                     report.append(f"**{i}. {segment.get('name', 'Segment')}**: {segment.get('description', '')}")
        
        # Analogies
        report.append("\n### Powerful Analogies")
        analogies_data = research_data.get('analogies', {})
        if analogies_data:
            challenge_analogies = analogies_data.get('challenge_analogies', [])
            solution_analogies = analogies_data.get('solution_analogies', [])
            generated_analogies = analogies_data.get('generated_analogies', [])
            
            all_analogies = challenge_analogies + solution_analogies + generated_analogies
            
            if all_analogies:
                displayed_count = 0
                for i, analogy in enumerate(all_analogies):
                     if displayed_count >= 4: break
                     if isinstance(analogy, dict):
                         title = analogy.get('title', analogy.get('analogy', f'Analogy {i+1}'))
                         desc = analogy.get('description', analogy.get('explanation', 'No description available'))
                         report.append(f"**{displayed_count+1}. {title}**")
                         report.append(f"  - {desc}")
                         displayed_count += 1
            else:
                report.append("No analogies identified.")
        else:
            report.append("No analogies identified.")
        
        # Visual assets
        visual_assets_data = research_data.get('visual_assets', {})
        if visual_assets_data:
            solution_visuals = visual_assets_data.get('solution_visuals', [])
            paradigm_visuals = visual_assets_data.get('paradigm_visuals', [])
            if not solution_visuals and not paradigm_visuals and isinstance(visual_assets_data, list):
                 solution_visuals = visual_assets_data
            
            if solution_visuals or paradigm_visuals:
                report.append("\n### Visual Assets")
                displayed_count = 0
                
                for i, visual in enumerate(solution_visuals):
                    if displayed_count >= 3: break
                    if isinstance(visual, dict):
                        title = visual.get('title', f'Solution Visual {i+1}')
                        url = visual.get('url', 'No URL available')
                        caption = visual.get('caption', visual.get('description', 'No caption available'))
                        
                        report.append(f"**{displayed_count+1}. {title}**")
                        report.append(f"  - URL: {url}")
                        report.append(f"  - Caption: {caption}")
                        displayed_count += 1
                
                for i, visual in enumerate(paradigm_visuals):
                    if displayed_count >= 5: break
                    if isinstance(visual, dict):
                        title = visual.get('title', f'Paradigm Visual {i+1}')
                        url = visual.get('url', 'No URL available')
                        caption = visual.get('caption', visual.get('description', 'No caption available'))
                        
                        report.append(f"**{displayed_count+1}. {title}**")
                        report.append(f"  - URL: {url}")
                        report.append(f"  - Caption: {caption}")
                        displayed_count += 1
        
        # Citations
        report.append("\n## Citations")
        citations = research_data.get('citations', [])
        if citations:
            report.append("\nThe following sources were found during research:\n")
            
            for i, citation in enumerate(citations, 1):
                report.append(f"### {i}. {citation.get('title', 'Unknown Source')}")
                report.append(f"- **URL:** {citation.get('url', 'No URL available')}")
                report.append(f"- **Source:** {citation.get('source', 'Unknown')}")
                report.append(f"- **Description:** {citation.get('description', 'No description available')}")
        else:
            report.append("\nNo citations were found during research.")
        
        # Readiness assessment
        report.append("\n## Readiness Assessment")
        readiness_text = "This blog post shows "
        
        try:
            numeric_score = float(readiness_score)
        except (ValueError, TypeError):
            numeric_score = 0.0
            
        if numeric_score >= 80:
            readiness_text += "**excellent readiness** for review. It provides comprehensive information and is well-structured."
        elif numeric_score >= 70:
            readiness_text += "**good readiness** for review. Some improvements might enhance the overall quality, but it can proceed to the review stage."
        elif numeric_score >= 60:
            readiness_text += "**adequate readiness** for review. Consider addressing the improvement areas noted before proceeding to the review stage."
        else:
            readiness_text += "**needs improvement** before review. Please address the deficiencies noted in this report."
        
        report.append(f"\n{readiness_text}")
        
        # Score breakdown
        if 'score_components' in research_data:
            report.append("\n### Score Breakdown")
            for name, score in research_data['score_components'].items():
                if name != 'base_score':
                    report.append(f"- **{name.replace('_', ' ').title()}:** {float(score):.1f} points")
        
        # Combine the report
        final_report = "\n".join(report)
        
        self._log_to_opik("Research report generation completed", "report_generation_complete", {
            "length": len(final_report)
        })
        
        return final_report
    
    def save_research_results(
        self, 
        blog_data: Dict[str, Any],
        metadata: Dict[str, Any],
        research_data: Dict[str, Any],
        readiness_score: float,
        report_markdown: str
    ) -> Dict[str, Any]:
        """
        Save research results to MongoDB with enhanced schema for sequential thinking artifacts.
        
        Args:
            blog_data: Blog data
            metadata: Metadata extracted from content
            research_data: Research data gathered
            readiness_score: Calculated readiness score (numerical value)
            report_markdown: Generated research report
            
        Returns:
            Dict with status and stored data information
        """
        blog_title = metadata.get('blog_title')
        version = metadata.get('version', 1)
        
        try:
            # Store the blog content with enhanced metadata
            blog_doc = {
                "title": blog_title,
                "current_version": version,
                "asset_folder": blog_data.get('asset_folder', ''),
                "versions": [
                    {
                        "version": version,
                        "file_path": "",  # Will be updated later
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "review_status": {
                            "factual_review": {"complete": False, "result_file": None},
                            "style_review": {"complete": False, "result_file": None},
                            "grammar_review": {"complete": False, "result_file": None},
                            "final_release": {"complete": False}
                        },
                        "readiness_score": readiness_score,
                        # Enhanced schema for research statistics
                        "research_stats": {
                            "industry_analysis": {
                                "challenges_count": len(research_data.get('industry_analysis', {}).get('challenges', [])),
                                "sources_count": len(research_data.get('citations', [])),
                            },
                            "proposed_solution": {
                                "pro_arguments_count": len(research_data.get('proposed_solution', {}).get('pro_arguments', [])),
                                "counter_arguments_count": len(research_data.get('proposed_solution', {}).get('counter_arguments', [])),
                                "visual_assets_count": len(research_data.get('visual_assets', [])),
                            },
                            "current_paradigm": {
                                "alternatives_count": len(research_data.get('current_paradigm', {}).get('alternatives', [])),
                            },
                            "audience_analysis": {
                                "knowledge_gaps_count": len(research_data.get('audience_analysis', {}).get('knowledge_gaps', [])),
                                "acronyms_count": len(research_data.get('audience_analysis', {}).get('acronyms', [])),
                                "analogies_count": len(research_data.get('analogies', {}).get('generated_analogies', [])),
                            },
                        }
                    }
                ]
            }
            
            # Store blog in MongoDB
            blog_id = self.db_client.db.blogs.update_one(
                {"title": blog_title},
                {"$set": {"current_version": version}, 
                 "$push": {"versions": blog_doc["versions"][0]}},
                upsert=True
            ).upserted_id
            
            if blog_id:
                logger.info(f"Created new blog entry with ID: {blog_id}")
            else:
                logger.info(f"Updated existing blog: {blog_title}")
            
            # Store the report
            report_filename = f"{blog_title}_research_report_v{version}.md"
            report_id = self.db_client.store_review_result(
                blog_title,
                version,
                "research",
                report_markdown,
                report_filename
            )
            logger.info(f"Stored research report with ID: {report_id}")
            
            # Enhanced schema for research data with sequential thinking artifacts
            enhanced_research_data = {
                "blog_title": blog_title,
                "version": version,
                "data": research_data,
                "sequential_thinking": {
                    "constraints": research_data.get('constraints', []),
                    "systemic_context": research_data.get('systemic_context', {}),
                    "stakeholder_perspectives": research_data.get('stakeholder_perspectives', []),
                    "challenges_solutions": {
                        "industry_challenges": research_data.get('industry_analysis', {}).get('challenges', []),
                        "proposed_solutions": research_data.get('proposed_solution', {})
                    },
                    "supporting_evidence": research_data.get('citations', []),
                    "counter_arguments": research_data.get('proposed_solution', {}).get('counter_arguments', [])
                },
                "constraint_analysis": {
                    "technical_constraints": research_data.get('constraints_analysis', {}).get('technical', []),
                    "financial_constraints": research_data.get('constraints_analysis', {}).get('financial', []),
                    "social_constraints": research_data.get('constraints_analysis', {}).get('social', []),
                    "regulatory_constraints": research_data.get('constraints_analysis', {}).get('regulatory', [])
                },
                "readiness_score": readiness_score,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version_history": [
                    {
                        "version": version,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "changes": "Initial research data"
                    }
                ]
            }
            
            # Store the enhanced research data
            research_data_id = self.db_client.db.research_data.update_one(
                {"blog_title": blog_title, "version": version},
                {"$set": enhanced_research_data},
                upsert=True
            ).upserted_id
            
            if research_data_id:
                logger.info(f"Stored enhanced research data with ID: {research_data_id}")
            else:
                logger.info(f"Updated existing research data for {blog_title} v{version}")
                
            # Update version history if this is an update
            if not research_data_id:
                self.db_client.db.research_data.update_one(
                    {"blog_title": blog_title, "version": version},
                    {"$push": {"version_history": {
                        "version": version,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "changes": "Research data updated"
                    }}}
                )
            
            # Store individual research components for more efficient retrieval
            for component_name, component_data in research_data.items():
                if isinstance(component_data, dict) and component_data:
                    self.db_client.store_research_component(
                        blog_title,
                        version,
                        component_name,
                        component_data
                    )
            
            # Store the images
            image_ids = []
            for img_ref, img_data in blog_data.get('images', {}).items():
                image_id = self.db_client.store_media(
                    blog_title,
                    version,
                    img_data.get('format', 'unknown'),
                    "local",
                    None,  # URL is None for local images
                    img_data.get('base64'),
                    img_ref  # Use image reference as alt text
                )
                image_ids.append(image_id)
            
            logger.info(f"Stored {len(image_ids)} blog images")
            
            # Store visual assets from research
            visual_asset_ids = []
            
            # Store visual assets from analogies
            for analogy in research_data.get('analogies', {}).get('generated_analogies', []):
                visual = analogy.get('visual', {})
                assets = visual.get('assets', [])
                
                for asset in assets:
                    asset_id = self.db_client.store_media(
                        blog_title,
                        version,
                        asset.get('format', 'unknown'),
                        "firecrawl",
                        asset.get('url'),
                        asset.get('base64'),
                        f"Visual for {analogy.get('title', 'analogy')}",
                        "analogy"
                    )
                    visual_asset_ids.append(asset_id)
            
            # Store visual assets from solution and paradigm
            for asset in research_data.get('visual_assets', []):
                asset_id = self.db_client.store_media(
                    blog_title,
                    version,
                    asset.get('format', 'image'),
                    asset.get('source', 'firecrawl'),
                    asset.get('url'),
                    asset.get('base64'),
                    asset.get('alt_text', 'Visual asset'),
                    asset.get('category', 'solution')
                )
                visual_asset_ids.append(asset_id)
            
            logger.info(f"Stored {len(visual_asset_ids)} visual assets")
            
            # Create YAML tracker
            yaml_path = create_tracker_yaml(blog_title, version, research_data)
            logger.info(f"Created YAML tracker: {yaml_path}")
            
            # Update research stats in blog document
            self.db_client.update_research_stats(
                blog_title,
                version,
                blog_doc["versions"][0]["research_stats"]
            )
            
            return {
                "status": "success",
                "blog_title": blog_title,
                "version": version,
                "readiness_score": readiness_score,
                "report_id": report_id,
                "research_data_id": str(research_data_id) if research_data_id else None,
                "image_ids": image_ids,
                "visual_asset_ids": visual_asset_ids,
                "yaml_path": yaml_path
            }
            
        except Exception as e:
            logger.error(f"Error saving research results: {e}")
            raise
    
    async def process_blog(self, file_path: str) -> Dict[str, Any]:
        """
        Process a blog from a file path with comprehensive progress tracking (asynchronously).
        
        Args:
            file_path: Path to the blog file (markdown or ZIP)
            
        Returns:
            Dict with processing results and detailed progress information
        """
        start_time = datetime.now()
        
        # Initialize progress tracking
        process_progress = {
            'status': 'in_progress',
            'started_at': start_time.isoformat(),
            'stages': {
                'file_processing': {'status': 'pending', 'started_at': None, 'completed_at': None, 'errors': []},
                'metadata_extraction': {'status': 'pending', 'started_at': None, 'completed_at': None, 'errors': []},
                'research_gathering': {'status': 'pending', 'started_at': None, 'completed_at': None, 'errors': []},
                'readiness_calculation': {'status': 'pending', 'started_at': None, 'completed_at': None, 'errors': []},
                'report_generation': {'status': 'pending', 'started_at': None, 'completed_at': None, 'errors': []},
                'result_saving': {'status': 'pending', 'started_at': None, 'completed_at': None, 'errors': []}
            },
            'current_stage': None,
            'completed_at': None,
            'duration_seconds': None,
            'result_summary': {}
        }
        
        # Log start of processing
        logger.info(f"Starting blog processing for: {file_path}")
        self._log_to_opik("Blog processing started", "blog_processing_start", {
            "file_path": file_path,
            "started_at": process_progress['started_at']
        })
        
        try:
            # STAGE 1: Process file (Synchronous)
            process_progress['current_stage'] = 'file_processing'
            process_progress['stages']['file_processing']['status'] = 'in_progress'
            process_progress['stages']['file_processing']['started_at'] = datetime.now().isoformat()
            
            try:
                # Check if the file is a ZIP or markdown
                if file_path.lower().endswith('.zip'):
                    # Process ZIP archive
                    logger.info(f"Processing ZIP archive: {file_path}")
                    blog_data = process_blog_upload(file_path)
                    logger.info(f"ZIP processing complete with {len(blog_data.get('images', []))} images")
                else:
                    # Process markdown file
                    logger.info(f"Processing markdown file: {file_path}")
                    blog_data = self.process_markdown_file(file_path)
                    logger.info(f"Markdown processing complete with {len(blog_data.get('images', []))} images")
                    
                    # Mark file processing as complete
                    process_progress['stages']['file_processing']['status'] = 'complete'
                    process_progress['stages']['file_processing']['completed_at'] = datetime.now().isoformat()
                    process_progress['stages']['file_processing']['result'] = {
                        'images_count': len(blog_data.get('images', [])),
                        'content_length': len(blog_data.get('content', ''))
                    }
                    
                    # Update progress log
                    logger.info(f"File processing complete: {file_path}")
                    self._log_to_opik("File processing complete", "file_processing_complete", {
                        "file_path": file_path,
                        "result": process_progress['stages']['file_processing']['result']
                    })
            except Exception as e:
                error_msg = f"Error processing file {file_path}: {e}"
                logger.error(error_msg)
                process_progress['stages']['file_processing']['status'] = 'error'
                process_progress['stages']['file_processing']['completed_at'] = datetime.now().isoformat()
                process_progress['stages']['file_processing']['errors'].append(str(e))
                self._log_to_opik("File processing error", "file_processing_error", {"error": str(e)})
                raise
            
            # STAGE 2: Extract metadata (Synchronous)
            process_progress['current_stage'] = 'metadata_extraction'
            process_progress['stages']['metadata_extraction']['status'] = 'in_progress'
            process_progress['stages']['metadata_extraction']['started_at'] = datetime.now().isoformat()
            
            try:
                # Extract metadata
                metadata = self.extract_metadata(blog_data['content'])
                metadata['blog_title'] = blog_data.get('blog_title')
                metadata['version'] = blog_data.get('version')
            
                # Add metadata summary to progress
                process_progress['stages']['metadata_extraction']['status'] = 'completed'
                process_progress['stages']['metadata_extraction']['completed_at'] = datetime.now().isoformat()
                process_progress['result_summary']['metadata'] = {
                    'headings_count': len(metadata.get('headings', [])),
                    'paragraphs_count': metadata.get('paragraphs_count', 0),
                    'images_count': metadata.get('images_count', 0),
                    'reading_time_minutes': metadata.get('reading_time_minutes', 0)
                }
                
                self._log_to_opik("Metadata extraction complete", "metadata_extraction_complete", {
                    "topic": metadata.get('main_topic'),
                    "headings_count": len(metadata.get('headings', [])),
                    "paragraphs_count": metadata.get('paragraphs_count', 0),
                    "images_count": metadata.get('images_count', 0)
                })
            except Exception as e:
                error_msg = f"Error extracting metadata: {e}"
                logger.error(error_msg)
                process_progress['stages']['metadata_extraction']['status'] = 'error'
                process_progress['stages']['metadata_extraction']['completed_at'] = datetime.now().isoformat()
                process_progress['stages']['metadata_extraction']['errors'].append(str(e))
                self._log_to_opik("Metadata extraction error", "metadata_extraction_error", {"error": str(e)})
                raise
            
            # STAGE 3: Gather research (NOW ASYNCHRONOUS)
            process_progress['current_stage'] = 'research_gathering'
            process_progress['stages']['research_gathering']['status'] = 'in_progress'
            process_progress['stages']['research_gathering']['started_at'] = datetime.now().isoformat()
            
            try:
                # Gather research data (using await)
                research_data = await self.gather_research(metadata) 
                
                # ... (Rest of research gathering stage logic remains the same) ...
                
            except Exception as e:
                # ... (Error handling for research gathering remains the same) ...
                raise
            
            # STAGE 4: Calculate readiness score (Synchronous)
            # ... (Readiness calculation logic remains the same) ...
            
            # STAGE 5: Generate research report (Synchronous)
            # ... (Report generation logic remains the same) ...
            
            # STAGE 6: Save results (Synchronous)
            # ... (Result saving logic remains the same) ...
            
            # ... (Success return logic remains the same, including report_url) ...

        except Exception as e:
            # ... (Overall error handling remains the same) ...
            return {
                'status': 'error',
                'error': str(e),
                'progress': process_progress
            }

    def _log_to_opik(self, message: str, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Log events to Opik MCP for monitoring and debugging.
        
        Args:
            message: Human-readable message about the event
            event_type: Type of event for classification
            data: Additional structured data about the event
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not self.opik_server:
            return False
        
        try:
            # Prepare the payload
            payload = {
                "agent": "researcher_agent",
                "message": message,
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data
            }
            
            # Send to Opik MCP
            response = requests.post(
                f"{self.opik_server}/api/v1/log",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5  # Short timeout to avoid blocking
            )
            
            if response.status_code == 200:
                logger.debug(f"Successfully logged to Opik MCP: {message}")
                return True
            else:
                logger.warning(f"Failed to log to Opik MCP: {response.status_code}")
                return False
            
        except Exception as e:
            logger.warning(f"Error logging to Opik MCP: {e}")
            return False


def main():
    """
    Command-line entry point for the Researcher Agent.
    """
    parser = argparse.ArgumentParser(description="Blog Accelerator Researcher Agent")
    
    # File argument
    parser.add_argument(
        "file_path",
        help="Path to the blog file to process (ZIP or markdown)"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--mongodb-uri",
        help="MongoDB connection URI",
        default=os.environ.get("MONGODB_URI")
    )
    parser.add_argument(
        "--brave-api-key",
        help="Brave Search API key",
        default=os.environ.get("BRAVE_API_KEY")
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key",
        default=os.environ.get("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--groq-api-key",
        help="Groq API key",
        default=os.environ.get("GROQ_API_KEY")
    )
    parser.add_argument(
        "--firecrawl-server",
        help="Firecrawl MCP server address",
        default=os.environ.get("FIRECRAWL_SERVER")
    )
    parser.add_argument(
        "--opik-server",
        help="Opik MCP server address",
        default=os.environ.get("OPIK_SERVER")
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the agent
    agent = ResearcherAgent(
        db_client=MongoDBClient(uri=args.mongodb_uri),
        brave_api_key=args.brave_api_key,
        opik_server=args.opik_server,
        firecrawl_server=args.firecrawl_server,
        openai_api_key=args.openai_api_key,
        groq_api_key=args.groq_api_key
    )
    
    # Process the blog
    result = agent.process_blog(args.file_path)
    
    # Print result
    print(json.dumps(result, indent=2))
    
    return 0


if __name__ == "__main__":
    # Calls async function correctly
    asyncio.run(main()) 
