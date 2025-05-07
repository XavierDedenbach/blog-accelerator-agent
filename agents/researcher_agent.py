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

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError as OpenAIRateLimitError, AuthenticationError as OpenAIAuthenticationError
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    # Create a mock ChatGroq class if the import fails
    class ChatGroq:
        def __init__(self, *args, **kwargs):
            raise ImportError("langchain_groq is not installed. Install it with: pip install langchain-groq")
    GROQ_AVAILABLE = False
try:
    from langchain_community.chat_models import ChatOpenRouter
    OPENROUTER_AVAILABLE = True
except ImportError:
    # Create a mock ChatOpenRouter class if the import fails
    class ChatOpenRouter:
         def __init__(self, *args, **kwargs):
            raise ImportError("langchain-community and openrouter are not installed. Install with: pip install langchain-community openrouter-python")
    OPENROUTER_AVAILABLE = False

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
        groq_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        llm_preference: Optional[List[str]] = None # Added LLM preference order
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
            openrouter_api_key: OpenRouter API key
            llm_preference: Ordered list of preferred LLM providers (e.g., ['openai', 'groq', 'openrouter'])
        """
        # Store DB client
        self.db_client = db_client
        
        # Store API keys from args or environment variables
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        
        # Store server addresses
        self.opik_server = opik_server
        self.firecrawl_server = firecrawl_server
        
        # Set LLM provider preference order
        self.llm_preference = llm_preference or ["openai", "groq", "openrouter"]
        
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
        
        # Initialize all available LLM instances
        self.openai_llm = self._initialize_openai_llm()
        self.groq_llm = self._initialize_groq_llm()
        self.openrouter_llm = self._initialize_openrouter_llm()
        
        # Store available LLM instances in a dictionary for easier access
        self.available_llms = {
            "openai": self.openai_llm,
            "groq": self.groq_llm,
            "openrouter": self.openrouter_llm
        }
        
        # Set the primary LLM based on preference and availability
        self.llm = None
        self.primary_llm_provider = None
        for provider in self.llm_preference:
            if self.available_llms.get(provider):
                self.llm = self.available_llms[provider]
                self.primary_llm_provider = provider
                logger.info(f"Primary LLM set to: {provider}")
                break
                
        if not self.llm:
             raise TopicAnalysisError("Could not initialize any LLM. Please check API keys and provider preferences.")
        
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
            self.industry_analyzer = IndustryAnalyzer(llm=self.llm, source_validator=self.source_validator)
            logger.info("Initialized IndustryAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize IndustryAnalyzer: {e}")
            self.industry_analyzer = None
            
        try:
            # Try to import and initialize SolutionAnalyzer
            from agents.research import SolutionAnalyzer
            self.solution_analyzer = SolutionAnalyzer(llm=self.llm, source_validator=self.source_validator)
            logger.info("Initialized SolutionAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize SolutionAnalyzer: {e}")
            self.solution_analyzer = None
            
        try:
            # Try to import and initialize ParadigmAnalyzer
            from agents.research import ParadigmAnalyzer
            self.paradigm_analyzer = ParadigmAnalyzer(llm=self.llm, source_validator=self.source_validator)
            logger.info("Initialized ParadigmAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize ParadigmAnalyzer: {e}")
            self.paradigm_analyzer = None
            
        try:
            # Try to import and initialize AudienceAnalyzer
            from agents.research import AudienceAnalyzer
            self.audience_analyzer = AudienceAnalyzer(llm=self.llm, source_validator=self.source_validator)
            logger.info("Initialized AudienceAnalyzer")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize AudienceAnalyzer: {e}")
            self.audience_analyzer = None
            
        try:
            # Try to import and initialize AnalogyGenerator
            from agents.research import AnalogyGenerator
            self.analogy_generator = AnalogyGenerator(
                llm=self.llm,
                source_validator=self.source_validator,
                firecrawl_client=self.firecrawl_client
            )
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
                    llm=self.llm, # Pass primary LLM
                    openai_api_key=self.openai_api_key,
                    firecrawl_client=self.firecrawl_client,
                    source_validator=self.source_validator 
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
            "openrouter_api_key": self.openrouter_api_key is not None, # Added OpenRouter key check
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
    
    def _initialize_openai_llm(self) -> Optional[BaseChatModel]:
        """Initializes OpenAI LLM if key is available."""
        if self.openai_api_key:
            try:
                logger.info("Initializing LLM with OpenAI (gpt-4o)")
                return ChatOpenAI(
                    model_name="gpt-4o", # Use latest model
                    temperature=0.5,
                    openai_api_key=self.openai_api_key
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI LLM: {e}")
        return None

    def _initialize_groq_llm(self) -> Optional[BaseChatModel]:
        """Initializes Groq LLM if key and library are available."""
        if self.groq_api_key and GROQ_AVAILABLE:
            try:
                logger.info("Initializing LLM with Groq (llama3-70b-8192)")
                return ChatGroq(
                    model_name="llama3-70b-8192", 
                    temperature=0.5, 
                    groq_api_key=self.groq_api_key
                )
            except ImportError:
                 logger.warning("Could not import langchain_groq. Skipping Groq LLM.")
            except Exception as e:
                 logger.error(f"Failed to initialize Groq LLM: {e}")
        return None

    def _initialize_openrouter_llm(self) -> Optional[BaseChatModel]:
        """Initializes OpenRouter LLM if key and library are available."""
        if self.openrouter_api_key and OPENROUTER_AVAILABLE:
            try:
                logger.info("Initializing LLM with OpenRouter (openai/gpt-4o)")
                return ChatOpenRouter(
                    model_name="openai/gpt-4o", # Specify model
                    openrouter_api_key=self.openrouter_api_key
                )
            except ImportError:
                 logger.warning("Could not import langchain_community. Skipping OpenRouter LLM.")
            except Exception as e:
                 logger.error(f"Failed to initialize OpenRouter LLM: {e}")
        return None

    def _get_fallback_order(self, start_provider: str) -> List[str]:
        """Gets the LLM provider fallback order starting from a given provider."""
        try:
            start_index = self.llm_preference.index(start_provider)
            # Return the order starting from the current provider
            return self.llm_preference[start_index:] + self.llm_preference[:start_index]
        except ValueError:
            # If start_provider is not in preference, return the default order
            logger.warning(f"Provider '{start_provider}' not in preference list. Using default order.")
            return self.llm_preference
    
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
    
    async def _execute_with_fallback(
        self, 
        task_func, 
        current_provider_name: str, 
        *args, 
        **kwargs
    ) -> Tuple[Optional[Any], Optional[str], Optional[BaseChatModel]]:
        """
        Executes a task function with LLM fallback logic, returning the result 
        and the LLM provider/instance that succeeded.
        
        Args:
            task_func: The async function to execute (e.g., component.analyze_x).
            current_provider_name: The name of the currently preferred LLM provider.
            *args: Positional arguments for the task_func.
            **kwargs: Keyword arguments for the task_func.
            
        Returns:
            A tuple containing:
            - result: The result from the task_func if successful, else None.
            - succeeded_provider_name: The name of the provider that succeeded, else None.
            - succeeded_llm_instance: The LLM instance that succeeded, else None.
        """
        fallback_order = self._get_fallback_order(current_provider_name)
        last_exception = None
        
        for provider_name in fallback_order:
            llm_instance = self.available_llms.get(provider_name)
            
            if not llm_instance:
                logger.debug(f"Skipping LLM provider '{provider_name}' as it's not available/initialized.")
                continue
                
            try:
                logger.info(f"Attempting task '{task_func.__name__}' with LLM provider: {provider_name}")
                # Pass the current llm_instance via llm_override
                kwargs['llm_override'] = llm_instance 
                result = await task_func(*args, **kwargs)
                logger.info(f"Task '{task_func.__name__}' succeeded with LLM provider: {provider_name}")
                return result, provider_name, llm_instance # Return result and successful provider info

            except (OpenAIRateLimitError, OpenAIAuthenticationError) as e: # Catch specific recoverable errors
                logger.warning(f"LLM provider '{provider_name}' failed with recoverable error: {type(e).__name__}. Trying next provider.")
                last_exception = e
                # Optionally add a small delay before retrying with the next provider
                await asyncio.sleep(1) 
            except Exception as e:
                # Catch other unexpected errors during the task execution
                logger.error(f"Unexpected error during task '{task_func.__name__}' with provider '{provider_name}': {e}", exc_info=True)
                last_exception = e
                # Decide if we should break or try next provider for general errors
                # For now, we continue to the next provider for any error during the task itself.
                # If the LLM initialization failed, it would be caught earlier.

        # If all providers failed
        logger.error(f"Task '{task_func.__name__}' failed after trying all available LLM providers in order: {fallback_order}.")
        if last_exception:
             logger.error(f"Last error encountered: {type(last_exception).__name__}: {last_exception}")
            # Instead of raising, return None to indicate failure
            # raise last_exception 
        
        return None, None, None # Indicate failure


    async def gather_research(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate the asynchronous gathering of all research components with persistent LLM fallback.
        
        Args:
            metadata: Dictionary containing topic metadata.
            
        Returns:
            Dictionary containing all gathered research data and errors.
        """
        start_time = datetime.now()
        topic = metadata.get('main_topic', 'Unknown Topic')
        logger.info(f"Starting async research orchestration for topic: {topic}")
        
        # Initialize current LLM state
        current_llm_provider_name = self.primary_llm_provider
        current_llm_instance = self.llm 
        
        # Dictionary to hold results from async tasks
        results = {}
        errors = {}
        
        # --- Define Async Tasks --- 
        
        # Citations Task (Doesn't use LLM, no fallback needed directly here)
        async def citations_task():
            task_name = "citations"
            try:
                # Note: The underlying source_validator.search_web might have its own retry/fallback logic if needed
                citations_result = await self.search_citations(topic, count=10)
                results[task_name] = citations_result
                logger.info(f"Async Citations successful: Found {len(citations_result)} citations.")
            except Exception as e:
                logger.error(f"Error in async Citations task: {e}")
                errors[task_name] = str(e)
            finally:
                 logger.info("Async Citations task finished.")
                 self._log_to_opik("Citations task finished", "task_complete", {"task": task_name, "success": task_name in results, "error": errors.get(task_name)})

        # Helper function to run an LLM task with fallback and update state
        async def run_llm_task(task_name: str, task_func, *args):
            nonlocal current_llm_provider_name, current_llm_instance # Allow modification of outer scope variables
            
            if not getattr(self, f"{task_name}_analyzer", True) and task_name != "analogy": # Check if component exists (except analogy)
                 if task_name == "analogy" and not self.analogy_generator:
                     logger.warning(f"Skipping {task_name} task (generator not initialized)")
                     errors[task_name] = "Generator not initialized"
                     return
                 elif task_name != "analogy":
                     logger.warning(f"Skipping {task_name} task (analyzer not initialized)")
                     errors[task_name] = "Analyzer not initialized"
                     return

            task_result, succeeded_provider, succeeded_instance = await self._execute_with_fallback(
                 task_func,
                 current_llm_provider_name, # Pass current provider name
                 *args # Pass positional arguments for the task
                 # llm_override is handled inside _execute_with_fallback
            )
            
            if succeeded_provider:
                 results[task_name] = task_result
                 # Update current LLM state IF it changed
                 if succeeded_provider != current_llm_provider_name:
                     logger.info(f"Switching active LLM provider from {current_llm_provider_name} to {succeeded_provider} for subsequent tasks.")
                     current_llm_provider_name = succeeded_provider
                     current_llm_instance = succeeded_instance
                 # Log success count based on typical output structure
                 count_info = ""
                 if task_name == 'industry' and isinstance(task_result, dict):
                     count_info = f"Found {len(task_result.get('challenges', []))} challenges."
                 elif task_name == 'solution' and isinstance(task_result, dict):
                     pro_count = len(task_result.get('pro_arguments', []))
                     con_count = len(task_result.get('counter_arguments', []))
                     count_info = f"Found {pro_count} pro / {con_count} con arguments."
                 elif task_name == 'paradigm' and isinstance(task_result, dict):
                     hist_count = len(task_result.get('historical_paradigms', []))
                     fut_count = len(task_result.get('future_paradigms', []))
                     count_info = f"Found {hist_count} historical / {fut_count} future paradigms."
                 elif task_name == 'audience' and isinstance(task_result, dict):
                     count_info = f"Found {len(task_result.get('audience_segments', []))} audience segments."
                 elif task_name == 'analogy' and isinstance(task_result, dict):
                     gen_count = len(task_result.get('generated_analogies', []))
                     ex_count = len(task_result.get('existing_analogies', []))
                     count_info = f"Found {gen_count} generated / {ex_count} existing analogies."
                 logger.info(f"Async {task_name.capitalize()} Analysis successful. {count_info}")
        else:
                 # Task failed even with fallback
                 error_msg = f"Task '{task_func.__name__}' failed after trying all providers."
                 logger.error(error_msg)
                 errors[task_name] = error_msg
            
            logger.info(f"Async {task_name.capitalize()} task finished.")
            self._log_to_opik(f"{task_name.capitalize()} task finished", "task_complete", {"task": task_name, "success": task_name in results, "error": errors.get(task_name)})


        # Visual Asset Task (Doesn't directly use LLM in the main call, but components might)
        async def visual_asset_task():
            task_name = "visuals"
            if not self.visual_asset_collector:
                logger.warning("Skipping visual asset collection (collector not initialized)")
                errors[task_name] = "Collector not initialized"
                return
            try:
                # Assume visual asset collector might use LLM internally via its components, 
                # but the main call doesn't need the llm_override directly.
                # If VAC needs direct LLM interaction with fallback, its methods would need adjustment.
                visual_result = await self.visual_asset_collector.collect_visuals(
                    topic=topic,
                    solution_data=results.get('solution'), # Pass results from previous tasks
                    paradigm_data=results.get('paradigm'),
                    analogy_data=results.get('analogy') 
                )
                results[task_name] = visual_result
                count = visual_result.get('stats', {}).get('total_collected', 0)
                logger.info(f"Async Visual Asset Collection successful: Collected {count} visuals.")
            except Exception as e:
                logger.error(f"Error in async Visual Asset Collection: {e}", exc_info=True)
                errors[task_name] = str(e)
            finally:
                logger.info("Async Visual Asset Collection task finished.")
                self._log_to_opik("Visual Asset task finished", "task_complete", {"task": task_name, "success": task_name in results, "error": errors.get(task_name)})

        # --- Define and Run Tasks Concurrently --- 
        
        # Define tasks using the helper
        llm_tasks_to_run = [
             ('industry', self.industry_analyzer.analyze_industry, topic),
             ('solution', self.solution_analyzer.analyze_solution, metadata.get('proposed_solution', 'Proposed Solution')),
             ('paradigm', self.paradigm_analyzer.analyze_paradigms, topic),
             ('audience', self.audience_analyzer.analyze_audience, topic),
             ('analogy', self.analogy_generator.generate_analogies, topic),
        ]
        
        # Create coroutines for LLM tasks
        llm_coroutines = [run_llm_task(*task_def) for task_def in llm_tasks_to_run if getattr(self, f"{task_def[0]}_analyzer", True) or task_def[0] == 'analogy']
        
        # Add non-LLM and visual tasks
        other_tasks = [citations_task()]
        if self.visual_asset_collector:
             other_tasks.append(visual_asset_task())

        # Combine all tasks
        all_tasks = llm_coroutines + other_tasks
             
        await asyncio.gather(*all_tasks)
        
        # --- Finalize and Return --- 
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        completion_percentage = (len(results) / len(all_tasks)) * 100
        
        logger.info(
            f"Async Research orchestration complete. Completion: {completion_percentage:.0f}%. "
            f"Errors: {len(errors)}"
        )
        
        self._log_to_opik("Research orchestration complete", "orchestration_complete", {
            "duration_seconds": duration,
            "completion_percentage": completion_percentage,
            "successful_tasks": list(results.keys()),
            "failed_tasks": list(errors.keys()),
            "errors": errors
        })
        
        # Combine results and errors for the final output
        final_result = {
            "research_data": results,
            "errors": errors,
            "metadata": metadata # Include original metadata
        }
        
        return final_result
    
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
        challenges_count = len(research_data.get('industry', {}).get('challenges', [])) # Updated key
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
        paradigm_data = research_data.get('paradigm', {}) # Updated key
        paradigms_count = paradigm_data.get('stats', {}).get('paradigms_count', 0)
        transitions_count = paradigm_data.get('stats', {}).get('transitions_count', 0)
        future_projections_count = paradigm_data.get('stats', {}).get('future_projections_count', 0)
        
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
        audience_data = research_data.get('audience', {}) # Updated key
        audience_segments_count = audience_data.get('stats', {}).get('segments_count', 0)
        # Need to update how knowledge gaps are counted if structure changed
        knowledge_gaps_count = 0
        for segment in audience_data.get('audience_segments', []):
             knowledge_gaps_count += len(segment.get('knowledge_evaluation', {}).get('likely_knowledge_gaps', []))
        
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
        visual_assets_data = research_data.get('visuals', {}) # Updated key
        solution_visuals_count = len(visual_assets_data.get('solution_visuals', []))
        paradigm_visuals_count = len(visual_assets_data.get('paradigm_visuals', []))
        
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
        analogies_data = research_data.get('analogy', {}) # Updated key
        # Default to empty lists if analogies_data is None or keys are missing
        # Assuming analogy structure provides generated analogies directly now
        generated_analogies_count = len(analogies_data.get('generated_analogies', [])) if analogies_data else 0
        existing_analogies_count = len(analogies_data.get('existing_analogies', [])) if analogies_data else 0
        
        # Use generated analogies count for scoring (adjust thresholds/logic if needed)
        challenge_analogies_target = self.readiness_thresholds.get('challenge_analogies_target', 3)
        challenge_analogies_min = self.readiness_thresholds.get('challenge_analogies_min', 1)
        challenge_analogies_points = self.readiness_thresholds.get('challenge_analogies_points', 5)
        
        if generated_analogies_count >= challenge_analogies_min:
            challenge_analogies_ratio = min(1.0, generated_analogies_count / challenge_analogies_target)
            challenge_analogies_score = challenge_analogies_ratio * challenge_analogies_points
        else:
            challenge_analogies_score = 0
        
        # Solution analogies score - Can reuse generated count or add separate logic
        solution_analogies_target = self.readiness_thresholds.get('solution_analogies_target', 3)
        solution_analogies_min = self.readiness_thresholds.get('solution_analogies_min', 1)
        solution_analogies_points = self.readiness_thresholds.get('solution_analogies_points', 5)
        
        if generated_analogies_count >= solution_analogies_min: # Reusing generated count
            solution_analogies_ratio = min(1.0, generated_analogies_count / solution_analogies_target)
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
        # Need to update keys if sequential thinking artifacts are stored differently
        has_constraints = bool(research_data.get('solution', {}).get('constraints', [])) # Example check
        has_systemic_context = bool(research_data.get('industry', {}).get('systemic_context', {})) # Example check
        has_stakeholder_perspectives = bool(research_data.get('audience', {}).get('stakeholder_mapping', {})) # Example check
        
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
        industry_data = research_data.get('industry', {}) # Updated key
        # Extract system info if available within industry data structure
        system_name = industry_data.get('system_name', 'Unknown System') 
        system_desc = industry_data.get('system_description', 'No description available')
        system_scale = industry_data.get('system_scale', 'Unknown')
        
        if system_name != 'Unknown System':
            report.append(f"**System Name:** {system_name}")
            report.append(f"**Description:** {system_desc}")
            report.append(f"**Scale:** {system_scale}")
            
            challenges = industry_data.get('challenges', []) # Use updated key
            if challenges:
                report.append("\n#### Key Challenges")
                for i, challenge_data in enumerate(challenges[:5], 1):
                    # challenge_data might now be {'challenge': ..., 'sources': ..., 'components': ...}
                    challenge = challenge_data # Keep original structure if simple list
                    if isinstance(challenge_data, dict):
                         challenge = challenge_data # Or extract from dict if structure changed
                    
                    if isinstance(challenge, dict): # Check if challenge is a dict
                        challenge_title = challenge.get('name', f'Challenge {i}')
                        challenge_desc = challenge.get('description', 'No description available')
                        report.append(f"**{i}. {challenge_title}**")
                        report.append(f"  - {challenge_desc}")
                        # Optionally add components analysis
                        components = challenge_data.get('components', {})
                        if components:
                             report.append(f"    - Risks: {len(components.get('risk_factors',[]))} factors")
                             report.append(f"    - Costs: {len(components.get('cost_factors',[]))} factors")
                    else:
                         report.append(f"**{i}. Invalid challenge format: {challenge_data}**")
            else:
                 report.append("\nNo challenges identified.")
        else:
             report.append("\nNo system/industry analysis data available.")
        
        # Solution analysis
        report.append("\n### Solution Analysis")
        solution_data = research_data.get('solution', {}) # Updated key
        proposed_solution = metadata.get('proposed_solution', {}) # Get from metadata if needed
        
        solution_name = solution_data.get('solution_title', proposed_solution.get('name', 'Proposed Solution'))
        # Assuming description is not directly in solution_data anymore, get from metadata or args if needed
        solution_description = proposed_solution.get('description', 'No description available') 
        
        report.append(f"**Solution:** {solution_name}")
        if solution_description != 'No description available':
        report.append(f"**Description:** {solution_description}")
        
        # Pro arguments
        report.append("\n#### Supporting Arguments")
        pro_args_data = solution_data.get('pro_arguments', []) # Updated key
        if pro_args_data:
            for i, arg_data in enumerate(pro_args_data[:5], 1):
                arg = arg_data.get('argument', {}) # Get inner dict
                if isinstance(arg, dict):
                    arg_text = arg.get('name', f'Argument {i}')
                    report.append(f"**{i}. {arg_text}**")
                    explanation = arg.get('description', '')
                    if explanation:
                        report.append(f"  - {explanation}")
                    # Optionally add prerequisites or metrics
                    prereqs = arg.get('prerequisites')
                    if prereqs:
                        report.append(f"    - Prerequisites: {prereqs}")
                else: # Handle if inner structure is just a string (fallback)
                    report.append(f"**{i}. Invalid argument format: {arg_data}**")
        else:
            report.append("No supporting arguments identified.")
        
        # Counter arguments
        report.append("\n#### Counter Arguments")
        counter_args_data = solution_data.get('counter_arguments', []) # Updated key
        if counter_args_data:
            for i, arg_data in enumerate(counter_args_data[:3], 1):
                arg = arg_data.get('argument', {}) # Get inner dict
                if isinstance(arg, dict):
                    arg_text = arg.get('name', f'Counter-argument {i}')
                    report.append(f"**{i}. {arg_text}**")
                    explanation = arg.get('description', '')
                    if explanation:
                        report.append(f"  - {explanation}")
                    mitigation = arg.get('mitigation_ideas', [])
                    if mitigation:
                        report.append(f"    - Mitigation Ideas: {'; '.join(mitigation)}")
                else: # Handle if inner structure is just a string (fallback)
                     report.append(f"**{i}. Invalid argument format: {arg_data}**")
        else:
            report.append("No counter arguments identified.")
            
        # Add Metrics if available
        metrics = solution_data.get('metrics', [])
        if metrics:
             report.append("\n#### Key Metrics")
             for i, metric in enumerate(metrics[:5], 1):
                  if isinstance(metric, dict):
                      report.append(f"**{i}. {metric.get('name', 'Metric')}**: {metric.get('importance_context', '')}")
                      report.append(f"     - Measurement: {metric.get('measurement_method', '')}")
                      report.append(f"     - Success: {metric.get('success_indicators', '')}")
        
        # Paradigm analysis
        report.append("\n### Historical Paradigm Analysis")
        paradigm_data = research_data.get('paradigm', {}) # Updated key
        
        historical_paradigms = paradigm_data.get('historical_paradigms', [])
        if historical_paradigms:
             report.append("\n#### Historical Paradigms")
             for i, paradigm in enumerate(historical_paradigms[:3], 1):
                 if isinstance(paradigm, dict):
                     report.append(f"**{i}. {paradigm.get('name', 'Paradigm')} ({paradigm.get('time_period', '')})**")
                     report.append(f"   - {paradigm.get('description', '')}")
                     characteristics = paradigm.get('key_characteristics', [])
                     if characteristics:
                          report.append(f"   - Characteristics: {'; '.join(characteristics)}")
        else:
             report.append("\nNo historical paradigms identified.")

        transitions = paradigm_data.get('transitions', [])
        if transitions:
             report.append("\n#### Key Transitions")
             for i, transition in enumerate(transitions[:2], 1):
                 if isinstance(transition, dict):
                     report.append(f"**{i}. {transition.get('from_paradigm', '')} -> {transition.get('to_paradigm', '')}**")
                     report.append(f"   - Triggers: {transition.get('trigger_factors', '')}")
                     report.append(f"   - Tensions: {transition.get('core_tensions', '')}")

        lessons = paradigm_data.get('lessons', [])
        if lessons:
            report.append("\n#### Key Lessons")
             for i, lesson in enumerate(lessons[:3], 1):
                 if isinstance(lesson, dict):
                     report.append(f"**{i}. {lesson.get('lesson', 'Lesson')}**: {lesson.get('explanation', '')}")
                      report.append(f"     - Relevance: {lesson.get('relevance_today', '')}")

        future_paradigms = paradigm_data.get('future_paradigms', [])
        if future_paradigms:
             report.append("\n#### Future Projections")
             for i, future in enumerate(future_paradigms[:2], 1):
                 if isinstance(future, dict):
                     report.append(f"**{i}. {future.get('name', 'Future Paradigm')}**")
                     report.append(f"   - {future.get('description', '')}")
                     report.append(f"   - Conditions: {future.get('emergence_conditions', '')}")
                     report.append(f"   - Implications: {future.get('potential_implications', '')}")
        
        # Audience analysis
        report.append("\n### Audience Analysis")
        audience_data = research_data.get('audience', {}) # Updated key
        segments = audience_data.get('audience_segments', [])
        
        if segments:
             report.append("\n#### Identified Segments")
            for i, segment in enumerate(segments[:3], 1):
                 if isinstance(segment, dict):
                     report.append(f"**{i}. {segment.get('name', 'Segment')}**")
                     report.append(f"   - Description: {segment.get('description', '')}")
                     report.append(f"   - Knowledge: {segment.get('knowledge_evaluation', {}).get('technical_depth_tolerance', segment.get('knowledge_level','Unknown'))}") # Use new or old key
                     pain_points = segment.get('needs_analysis', {}).get('core_pain_points', segment.get('pain_points',[])) # Use new or old key
                     if pain_points:
                          report.append(f"   - Pain Points: {'; '.join(pain_points[:2])}")
                     # Optionally add content strategies
                     strategies = segment.get('content_strategies', [])
                     if strategies:
                          strategy_titles = [s.get('title', 'Strategy') for s in strategies[:2]]
                          report.append(f"   - Suggested Strategies: {'; '.join(strategy_titles)}")
        else:
            report.append("\nNo specific audience segments identified.")
        
        # Analogies
        report.append("\n### Powerful Analogies")
        analogies_data = research_data.get('analogy', {}) # Updated key
            generated_analogies = analogies_data.get('generated_analogies', [])
        existing_analogies = analogies_data.get('existing_analogies', [])
            
        all_analogies = generated_analogies + existing_analogies
            
            if all_analogies:
                displayed_count = 0
                for i, analogy in enumerate(all_analogies):
                     if displayed_count >= 4: break
                     if isinstance(analogy, dict):
                     title = analogy.get('title', f'Analogy {i+1}')
                     desc = analogy.get('description', 'No description available')
                     domain = analogy.get('domain', 'Unknown Domain')
                     report.append(f"**{displayed_count+1}. {title} (Domain: {domain})**")
                         report.append(f"  - {desc}")
                     # Optionally add evaluation score if available
                     score = analogy.get("evaluation", {}).get("overall_score")
                     if score is not None:
                          report.append(f"  - Score: {float(score):.1f}/10")
                         displayed_count += 1
        else:
            report.append("No analogies identified.")
        
        # Visual assets
        visual_assets_data = research_data.get('visuals', {}) # Updated key
        if visual_assets_data:
            solution_visuals = visual_assets_data.get('solution_visuals', [])
            paradigm_visuals = visual_assets_data.get('paradigm_visuals', [])
             analogy_visuals = visual_assets_data.get('analogy_visuals', []) # Check if VAC collects these now
            
             all_visuals = solution_visuals + paradigm_visuals + analogy_visuals
             
             if all_visuals:
                report.append("\n### Visual Assets")
                report.append(f"Total collected: {visual_assets_data.get('stats',{}).get('total_collected', len(all_visuals))}")
                displayed_count = 0
                
                # Display a sample
                for i, visual in enumerate(all_visuals):
                    if displayed_count >= 5: break
                    if isinstance(visual, dict):
                        title = visual.get('title', visual.get('alt_text', f'Visual {i+1}'))
                        url = visual.get('url', 'No URL available')
                        category = visual.get('category', 'Unknown')
                        
                        report.append(f"**{displayed_count+1}. {title} (Category: {category})**")
                        report.append(f"  - URL: {url}")
                        displayed_count += 1
             else:
                  report.append("\nNo visual assets collected.")
        else:
             report.append("\nNo visual asset data available.")
        
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
                        # Enhanced schema for research statistics (Update keys)
                        "research_stats": {
                            "industry_analysis": {
                                "challenges_count": len(research_data.get('industry', {}).get('challenges', [])),
                                "sources_count": research_data.get('industry', {}).get('total_sources', 0), # Example
                            },
                            "proposed_solution": {
                                "pro_arguments_count": len(research_data.get('solution', {}).get('pro_arguments', [])),
                                "counter_arguments_count": len(research_data.get('solution', {}).get('counter_arguments', [])),
                                "metrics_count": len(research_data.get('solution', {}).get('metrics', [])),
                                "visual_assets_count": len(research_data.get('visuals', {}).get('solution_visuals', [])), # Example from visuals
                            },
                            "paradigm_analysis": { # New structure
                                "historical_paradigms_count": len(research_data.get('paradigm', {}).get('historical_paradigms', [])),
                                "transitions_count": len(research_data.get('paradigm', {}).get('transitions', [])),
                                "lessons_count": len(research_data.get('paradigm', {}).get('lessons', [])),
                                "future_projections_count": len(research_data.get('paradigm', {}).get('future_paradigms', [])),
                            },
                            "audience_analysis": {
                                "segments_count": len(research_data.get('audience', {}).get('audience_segments', [])),
                                # Add more details if needed, e.g., average pain points per segment
                            },
                             "analogy_generation": { # New structure
                                 "generated_analogies_count": len(research_data.get('analogy', {}).get('generated_analogies', [])),
                                 "existing_analogies_count": len(research_data.get('analogy', {}).get('existing_analogies', [])),
                                 "average_score": research_data.get('analogy', {}).get('stats', {}).get('average_score', 0),
                             },
                             "visual_asset_collection": { # New structure
                                 "total_collected": research_data.get('visuals', {}).get('stats', {}).get('total_collected', 0),
                             },
                            "citations_count": len(research_data.get('citations', [])), # Overall citations
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
                # Update keys for sequential thinking if structure changed
                "sequential_thinking": {
                    "constraints": research_data.get('solution', {}).get('constraints', []),
                    "systemic_context": research_data.get('industry', {}).get('systemic_context', {}),
                    "stakeholder_perspectives": research_data.get('audience', {}).get('stakeholder_mapping', {}),
                    "challenges_solutions": {
                        "industry_challenges": research_data.get('industry', {}).get('challenges', []),
                        "proposed_solutions": research_data.get('solution', {}) # Or more specific part
                    },
                    "supporting_evidence": research_data.get('citations', []),
                    "counter_arguments": research_data.get('solution', {}).get('counter_arguments', [])
                },
                "constraint_analysis": research_data.get('constraints_analysis', {}), # If generated
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
            
            # Store visual assets from research (Update key)
            visual_asset_ids = []
            visuals_data = research_data.get('visuals', {})
            all_collected_visuals = (
                visuals_data.get('solution_visuals', []) +
                visuals_data.get('paradigm_visuals', []) +
                visuals_data.get('analogy_visuals', []) +
                visuals_data.get('other_visuals', []) # Add other categories if VAC uses them
            )
            
            for asset in all_collected_visuals:
                 if isinstance(asset, dict): # Ensure it's a dict
                    asset_id = self.db_client.store_media(
                        blog_title,
                        version,
                        asset.get('format', 'unknown'),
                        "firecrawl",
                        asset.get('url'),
                        asset.get('base64'),
                         f"Visual for {asset.get('title', 'visual')}",
                         "visual"
                )
                visual_asset_ids.append(asset_id)
            
            logger.info(f"Stored {len(visual_asset_ids)} visual assets")
            
            # Create YAML tracker
            yaml_path = create_tracker_yaml(blog_title, version, research_data)
            logger.info(f"Created YAML tracker: {yaml_path}")
            
            # Update research stats in blog document (using potentially updated structure)
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
            
            research_result = {} # Initialize empty dict
            try:
                # Gather research data (using await)
                # gather_research now returns dict like {"research_data": {...}, "errors": {...}, "metadata": {...}}
                research_result = await self.gather_research(metadata) 
                research_data = research_result.get("research_data", {}) # Extract the data part
                research_errors = research_result.get("errors", {}) # Extract errors part
                
                # Update progress based on research_errors
                if research_errors:
                     process_progress['stages']['research_gathering']['status'] = 'partial_success'
                     process_progress['stages']['research_gathering']['errors'] = list(research_errors.values())
                     process_progress['result_summary']['research_errors'] = research_errors
                     logger.warning(f"Research gathering completed with errors: {research_errors}")
                else:
                    process_progress['stages']['research_gathering']['status'] = 'complete'
                
                process_progress['stages']['research_gathering']['completed_at'] = datetime.now().isoformat()
                process_progress['result_summary']['research_components'] = list(research_data.keys())
                
                self._log_to_opik("Research gathering complete", "research_gathering_complete", {
                    "components_successful": list(research_data.keys()),
                    "components_failed": list(research_errors.keys())
                })
                
            except Exception as e:
                error_msg = f"Critical error during research gathering: {e}"
                logger.error(error_msg, exc_info=True) # Log traceback for critical errors
                process_progress['stages']['research_gathering']['status'] = 'error'
                process_progress['stages']['research_gathering']['completed_at'] = datetime.now().isoformat()
                process_progress['stages']['research_gathering']['errors'].append(str(e))
                self._log_to_opik("Research gathering error", "research_gathering_error", {"error": str(e)})
                raise # Re-raise critical errors to stop processing
            
            # STAGE 4: Calculate readiness score (Synchronous)
            process_progress['current_stage'] = 'readiness_calculation'
            process_progress['stages']['readiness_calculation']['status'] = 'in_progress'
            process_progress['stages']['readiness_calculation']['started_at'] = datetime.now().isoformat()
            
            readiness_info = {}
            try:
                # Use the research_data obtained from gather_research
                readiness_info = self.calculate_readiness_score(metadata, research_data)
                readiness_score = readiness_info.get('score', 0.0)
                readiness_grade = readiness_info.get('grade', 'F')
                
                process_progress['stages']['readiness_calculation']['status'] = 'complete'
                process_progress['stages']['readiness_calculation']['completed_at'] = datetime.now().isoformat()
                process_progress['result_summary']['readiness'] = {
                    'score': readiness_score,
                    'grade': readiness_grade
                }
                self._log_to_opik("Readiness calculation complete", "readiness_calculation_complete", {
                    "score": readiness_score, "grade": readiness_grade
                })
            except Exception as e:
                error_msg = f"Error calculating readiness score: {e}"
                logger.error(error_msg)
                process_progress['stages']['readiness_calculation']['status'] = 'error'
                process_progress['stages']['readiness_calculation']['completed_at'] = datetime.now().isoformat()
                process_progress['stages']['readiness_calculation']['errors'].append(str(e))
                self._log_to_opik("Readiness calculation error", "readiness_calculation_error", {"error": str(e)})
                # Decide if this is critical - maybe allow processing to continue without score?
                readiness_score = 0.0 # Default score on error
                readiness_grade = 'N/A'
            
            # STAGE 5: Generate research report (Synchronous)
            process_progress['current_stage'] = 'report_generation'
            process_progress['stages']['report_generation']['status'] = 'in_progress'
            process_progress['stages']['report_generation']['started_at'] = datetime.now().isoformat()
            
            report_markdown = ""
            try:
                # Use the research_data obtained from gather_research
                report_markdown = self.generate_research_report(
                    blog_data,
                    metadata,
                    research_data,
                    readiness_score # Use score calculated above
                )
                
                process_progress['stages']['report_generation']['status'] = 'complete'
                process_progress['stages']['report_generation']['completed_at'] = datetime.now().isoformat()
                process_progress['result_summary']['report_length'] = len(report_markdown)
                self._log_to_opik("Report generation complete", "report_generation_complete", {"length": len(report_markdown)})
            except Exception as e:
                 error_msg = f"Error generating report: {e}"
                 logger.error(error_msg)
                 process_progress['stages']['report_generation']['status'] = 'error'
                 process_progress['stages']['report_generation']['completed_at'] = datetime.now().isoformat()
                 process_progress['stages']['report_generation']['errors'].append(str(e))
                 self._log_to_opik("Report generation error", "report_generation_error", {"error": str(e)})
                 # Continue processing, but report generation failed
            
            # STAGE 6: Save results (Synchronous)
            process_progress['current_stage'] = 'result_saving'
            process_progress['stages']['result_saving']['status'] = 'in_progress'
            process_progress['stages']['result_saving']['started_at'] = datetime.now().isoformat()
            
            save_result = {}
            report_url = None # Initialize report URL
            try:
                # Use the research_data obtained from gather_research
                save_result = self.save_research_results(
                    blog_data,
                    metadata,
                    research_data,
                    readiness_score, # Use calculated score
                    report_markdown # Use generated report
                )
                
                # Construct report URL if Opik server is configured and report ID exists
                if self.opik_server and save_result.get('report_id'):
                    report_url = f"{self.opik_server}/reports/{save_result['report_id']}" # Example URL structure
                
                process_progress['stages']['result_saving']['status'] = 'complete'
                process_progress['stages']['result_saving']['completed_at'] = datetime.now().isoformat()
                process_progress['result_summary']['save_info'] = {
                     "report_id": str(save_result.get('report_id')),
                     "research_data_id": str(save_result.get('research_data_id')),
                     "image_count": len(save_result.get('image_ids', [])),
                     "visual_asset_count": len(save_result.get('visual_asset_ids', [])),
                     "yaml_path": save_result.get('yaml_path')
                 }
                self._log_to_opik("Result saving complete", "result_saving_complete", process_progress['result_summary']['save_info'])
        except Exception as e:
                error_msg = f"Error saving results: {e}"
                logger.error(error_msg)
                process_progress['stages']['result_saving']['status'] = 'error'
                process_progress['stages']['result_saving']['completed_at'] = datetime.now().isoformat()
                process_progress['stages']['result_saving']['errors'].append(str(e))
                self._log_to_opik("Result saving error", "result_saving_error", {"error": str(e)})
                # Continue processing, but saving failed
            
            # Determine overall status
            has_errors = any(stage['status'] == 'error' for stage in process_progress['stages'].values())
            has_partial = any(stage['status'] == 'partial_success' for stage in process_progress['stages'].values())
            
            if has_errors:
                process_progress['status'] = 'error'
            elif has_partial:
                process_progress['status'] = 'partial_success'
            else:
                 process_progress['status'] = 'success'

            end_time = datetime.now()
            process_progress['completed_at'] = end_time.isoformat()
            process_progress['duration_seconds'] = (end_time - start_time).total_seconds()
            
            self._log_to_opik("Blog processing finished", "blog_processing_complete", {
                 "final_status": process_progress['status'],
                 "duration": process_progress['duration_seconds']
            })
            
            # Return the final result structure including research data and errors
            return {
                "status": process_progress['status'],
                "research_data": research_data, # Include the actual data
                "errors": research_errors, # Include the errors encountered
                "progress": process_progress, # Include detailed progress
                "report_url": report_url # Include report URL if available
            }

        except Exception as e:
            # Catch errors from synchronous stages or re-raised critical errors
            end_time = datetime.now()
            process_progress['status'] = 'error'
            process_progress['completed_at'] = end_time.isoformat()
            process_progress['duration_seconds'] = (end_time - start_time).total_seconds()
            # Add error to the relevant stage if possible, otherwise general error
            current_stage = process_progress['current_stage']
            if current_stage and current_stage in process_progress['stages']:
                 process_progress['stages'][current_stage]['status'] = 'error'
                 process_progress['stages'][current_stage]['errors'].append(str(e))
            
            logger.error(f"Overall processing failed at stage '{current_stage}': {e}", exc_info=True)
            self._log_to_opik("Blog processing failed", "blog_processing_error", {
                 "stage": current_stage, "error": str(e)
            })
            
            # Return error structure
            return {
                "status": "error",
                "error": str(e),
                "research_data": {}, # Empty data on critical failure
                "errors": {current_stage or "overall": str(e)}, # Log the error source
                "progress": process_progress
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
    # asyncio.run(main()) - This was causing issues with `await agent.process_blog`
    # Correct way to run the top-level async function:
    try:
    asyncio.run(main()) 
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)
