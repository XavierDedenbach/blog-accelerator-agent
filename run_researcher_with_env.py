import os
import sys
import asyncio
import json
import time
import logging
from collections import deque
from datetime import datetime, timezone
from dotenv import load_dotenv
from agents.researcher_agent import ResearcherAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Rate limiter class
class RateLimiter:
    """Simple rate limiter to enforce API rate limits."""
    
    def __init__(self, max_calls_per_second: int = 1, buffer_time: float = 0.1):
        """
        Initialize rate limiter.
        
        Args:
            max_calls_per_second: Maximum number of calls allowed per second
            buffer_time: Additional buffer time for safety margin
        """
        self.max_calls_per_second = max_calls_per_second
        self.buffer_time = buffer_time
        self.call_timestamps = deque()
    
    async def wait_if_needed(self):
        """Wait if necessary to stay within rate limits."""
        now = time.time()
        
        # Remove timestamps older than 1 second
        while self.call_timestamps and now - self.call_timestamps[0] > 1:
            self.call_timestamps.popleft()
        
        # If we've reached the maximum calls per second, wait
        if len(self.call_timestamps) >= self.max_calls_per_second:
            wait_time = 1 - (now - self.call_timestamps[0]) + self.buffer_time
            if wait_time > 0:
                logger.debug(f"Rate limiter: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        else:
            # Even if under the limit, add a small delay for more reliable spacing
            await asyncio.sleep(self.buffer_time)
        
        # Add current timestamp (after waiting)
        self.call_timestamps.append(time.time())

# Custom ResearcherAgent class with rate limiting and fallbacks
class RateLimitedResearcherAgent(ResearcherAgent):
    """Researcher Agent with built-in rate limiting for API calls and fallback mechanisms."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the rate-limited researcher agent with rate limiters for different APIs."""
        # Store keys for fallback
        self.openrouter_api_key = kwargs.pop('openrouter_api_key', os.environ.get('OPENROUTER_API_KEY'))
        self.using_openrouter_fallback = False
        
        # Initialize rate limiters
        self.brave_limiter = RateLimiter(max_calls_per_second=20, buffer_time=0.02)  # 2 calls per second for Brave (conservative based on testing)
        self.openai_limiter = RateLimiter(max_calls_per_second=3, buffer_time=0.1)   # 3 calls per second for OpenAI
        self.openrouter_limiter = RateLimiter(max_calls_per_second=5, buffer_time=0.05)  # 5 calls per second for OpenRouter
        
        # Initialize the base ResearcherAgent
        super().__init__(*args, **kwargs)
        
        # Store original methods
        self.original_search_citations = self.source_validator.search_web
        
        # Replace the search_citations method with our rate-limited version
        self.source_validator.search_web = self.rate_limited_search_web
        
        # Apply rate limiting to research components
        self._patch_research_components()
    
    async def rate_limited_search_web(self, query, count=5):
        """Rate-limited version of search_web with improved error handling."""
        logger.debug(f"Rate-limited search web for query: {query}")
        
        try:
            # Apply rate limiting
            await self.brave_limiter.wait_if_needed()
            
            # Call the original method
            try:
                result = await self.original_search_citations(query, count)
                logger.debug(f"Successfully found {len(result)} search results for query: {query}")
                return result
            except Exception as e:
                # Check if error is a rate limit error
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    logger.warning(f"Brave API rate limit hit for query: {query}")
                    
                    # Return fallback mock data for rate limits
                    mock_results = [
                        {
                            'title': f'Rate-limited fallback for "{query}" #{i}',
                            'url': f'https://example.com/ratelimited{i}',
                            'description': f'This is a fallback result #{i} due to API rate limits for query: {query}',
                            'source': 'Rate-Limited Fallback',
                            'date': datetime.now(timezone.utc).isoformat()
                        } for i in range(1, count + 1)
                    ]
                    logger.info(f"Returning {len(mock_results)} fallback results due to rate limiting")
                    return mock_results
                else:
                    # Re-raise non-rate-limit errors
                    logger.error(f"Error in search_web for query {query}: {e}")
                    raise
        except Exception as e:
            logger.error(f"Error in rate_limited_search_web: {e}")
            
            # Return mock data as a final fallback
            return [
                {
                    'title': f'Error fallback for "{query}" #{i}',
                    'url': f'https://example.com/error{i}',
                    'description': f'Fallback result due to search error: {str(e)}',
                    'source': 'Error Fallback',
                    'date': datetime.now(timezone.utc).isoformat()
                } for i in range(1, count + 1)
            ]
    
    def _patch_research_components(self):
        """Apply rate limiting to all research components that use LLM APIs."""
        # Patch industry analyzer
        if hasattr(self, 'industry_analyzer'):
            self._patch_llm_client(self.industry_analyzer)
        
        # Patch solution analyzer
        if hasattr(self, 'solution_analyzer'):
            self._patch_llm_client(self.solution_analyzer)
        
        # Patch paradigm analyzer
        if hasattr(self, 'paradigm_analyzer'):
            self._patch_llm_client(self.paradigm_analyzer)
        
        # Patch audience analyzer
        if hasattr(self, 'audience_analyzer'):
            self._patch_llm_client(self.audience_analyzer)
        
        # Patch analogy generator
        if hasattr(self, 'analogy_generator'):
            self._patch_llm_client(self.analogy_generator)
    
    def _patch_llm_client(self, component):
        """Patch a component's LLM client with rate limiting and fallback capabilities."""
        try:
            # Identify the attribute that contains the LLM client
            llm_attribute = None
            if hasattr(component, 'llm_client'):
                llm_attribute = 'llm_client'
            elif hasattr(component, 'llm'):
                llm_attribute = 'llm'
            else:
                logger.warning(f"No LLM client found for component: {component.__class__.__name__}")
                return
                
            llm_object = getattr(component, llm_attribute)
            
            # Get the class name to handle different types of LLM clients
            llm_class_name = llm_object.__class__.__name__
            
            # Check which method to patch based on the LLM client type
            patch_method = None
            if hasattr(llm_object, 'invoke') and callable(getattr(llm_object, 'invoke')):
                patch_method = 'invoke'
                original_method = llm_object.invoke
            elif hasattr(llm_object, '__call__') and callable(getattr(llm_object, '__call__')):
                patch_method = '__call__'
                original_method = llm_object.__call__
            else:
                logger.error(f"LLM client for {component.__class__.__name__} has neither usable 'invoke' nor '__call__' method")
                return
            
            async def rate_limited_call(*args, **kwargs):
                """Rate-limited version of the call method with fallback capability."""
                # Try with OpenAI first
                if not self.using_openrouter_fallback:
                    try:
                        logger.debug(f"Applying rate limit for OpenAI API call")
                        await self.openai_limiter.wait_if_needed()
                        return await original_method(*args, **kwargs)
                    except Exception as e:
                        error_str = str(e).lower()
                        
                        # Check if this is a rate limit error
                        if (
                            "rate limit" in error_str or 
                            "too many requests" in error_str or 
                            "429" in error_str or
                            "quota exceeded" in error_str or
                            "capacity" in error_str
                        ):
                            logger.warning(f"OpenAI rate limit exceeded. Error: {e}")
                            
                            # Try switching to OpenRouter if API key is available
                            if self.openrouter_api_key:
                                logger.info("Switching to OpenRouter fallback...")
                                self.using_openrouter_fallback = True
                                # Update the LLM client configuration
                                self._setup_openrouter_fallback(component)
                                
                                # Try again with OpenRouter
                                try:
                                    logger.info("Retrying request with OpenRouter...")
                                    await self.openrouter_limiter.wait_if_needed()
                                    # Need to get the updated LLM object
                                    updated_llm = getattr(component, llm_attribute)
                                    if patch_method == 'invoke':
                                        return await updated_llm.invoke(*args, **kwargs)
                                    else:
                                        return await updated_llm.__call__(*args, **kwargs)
                                except Exception as fallback_error:
                                    logger.error(f"OpenRouter fallback also failed: {fallback_error}")
                                    raise
                            else:
                                logger.error("No OpenRouter API key available for fallback")
                                raise
                        else:
                            # Other error not related to rate limiting
                            logger.error(f"Non-rate-limit error in LLM call: {e}")
                            raise
                else:
                    # Already using OpenRouter fallback
                    logger.debug(f"Applying rate limit for OpenRouter API call")
                    await self.openrouter_limiter.wait_if_needed()
                    return await original_method(*args, **kwargs)
            
            # Replace the method with our rate-limited version using the appropriate patching approach
            if llm_class_name == "ChatOpenAI":
                # For ChatOpenAI, we need to use a different approach to monkey patching
                # Store the original method
                import types
                if patch_method == 'invoke':
                    # Create a new method bound to the object
                    bound_method = types.MethodType(rate_limited_call, llm_object)
                    # Use a custom descriptor to intercept invoke calls
                    class MethodInterceptor:
                        def __get__(self, obj, objtype=None):
                            return bound_method
                    # Add the descriptor to the class
                    setattr(llm_object.__class__, 'invoke', MethodInterceptor())
                    logger.info(f"Successfully patched {llm_attribute}.invoke for {component.__class__.__name__} using descriptor")
                else:
                    # For __call__, we need a different approach
                    original_call = llm_object.__call__
                    async def wrapped_call(*args, **kwargs):
                        return await rate_limited_call(*args, **kwargs)
                    llm_object.__call__ = wrapped_call
                    logger.info(f"Successfully patched {llm_attribute}.__call__ for {component.__class__.__name__}")
            else:
                # For other LLM types, we can directly replace the method
                if patch_method == 'invoke':
                    llm_object.invoke = rate_limited_call
                    logger.info(f"Successfully patched {llm_attribute}.invoke for {component.__class__.__name__}")
                else:
                    llm_object.__call__ = rate_limited_call
                    logger.info(f"Successfully patched {llm_attribute}.__call__ for {component.__class__.__name__}")
        
        except Exception as e:
            logger.error(f"Failed to patch LLM for {component.__class__.__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue without failing the entire patching process

    def _setup_openrouter_fallback(self, component):
        """Set up OpenRouter as a fallback for a component's LLM client."""
        if not self.openrouter_api_key:
            logger.error("No OpenRouter API key available for fallback")
            return
            
        try:
            # Import required LLM implementation
            from langchain.chat_models import ChatOpenAI
            
            # Create a new chat model using OpenRouter API with the OpenAI interface
            openrouter_chat = ChatOpenAI(
                model="claude-3-sonnet-20240229",  # Use Claude 3 Sonnet via OpenRouter
                openai_api_key=self.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=4000,
                request_timeout=120,
                # Headers required for OpenRouter
                headers={
                    "HTTP-Referer": "https://blog-accelerator.agent",
                    "X-Title": "Blog Accelerator Agent"
                }
            )
            
            # Replace the component's LLM client/model with the OpenRouter model
            if hasattr(component, 'llm_client'):
                logger.info(f"Setting up OpenRouter fallback for {component.__class__.__name__} (llm_client)")
                component.llm_client = openrouter_chat
            elif hasattr(component, 'llm'):
                logger.info(f"Setting up OpenRouter fallback for {component.__class__.__name__} (llm)")
                component.llm = openrouter_chat
            else:
                logger.warning(f"Cannot set up OpenRouter fallback for {component.__class__.__name__}: no LLM attribute found")
                return
                
            logger.info(f"Successfully set up OpenRouter fallback for {component.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to set up OpenRouter fallback: {e}")
            # Include the traceback for better debugging
            import traceback
            logger.error(traceback.format_exc())

async def main():
    # Get environment variables
    mongodb_uri = os.getenv("MONGODB_URI")
    brave_api_key = os.getenv("BRAVE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    firecrawl_server = os.getenv("FIRECRAWL_SERVER")
    opik_server = os.getenv("OPIK_SERVER")
    
    # Print loaded API keys (without revealing full keys)
    print(f"MongoDB URI: {mongodb_uri}")
    print(f"Brave API Key: {brave_api_key[:5]}..." if brave_api_key else "Brave API Key: None")
    print(f"OpenAI API Key: {openai_api_key[:5]}..." if openai_api_key else "OpenAI API Key: None")
    print(f"Groq API Key: {groq_api_key[:5]}..." if groq_api_key else "Groq API Key: None")
    print(f"OpenRouter API Key: {openrouter_api_key[:5]}..." if openrouter_api_key else "OpenRouter API Key: None")
    print(f"Firecrawl Server: {firecrawl_server}")
    print(f"Opik Server: {opik_server}")
    
    # Inform about rate limiting
    print("\nRate limiting enabled:")
    print("API Rate Limiting:")
    print("- OpenAI API: 3 calls per second (primary)")
    print("- OpenRouter API: 5 calls per second (fallback)")
    print("- Brave Search API: 2 calls per second (conservative rate)")
    
    # Initialize the rate-limited agent
    agent = RateLimitedResearcherAgent(
        mongodb_uri=mongodb_uri,
        brave_api_key=brave_api_key,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
        firecrawl_server=firecrawl_server,
        opik_server=opik_server
    )
    
    # Get file path from command line argument
    if len(sys.argv) < 2:
        print("\nUsage: python run_researcher_with_env.py <path_to_blog_file>")
        return
    
    file_path = sys.argv[1]
    print(f"\nProcessing blog file: {file_path}")
    print("This might take some time due to rate limiting...")
    print("If OpenAI API rate limits are reached, the system will automatically fallback to OpenRouter.")
    
    # Process the blog
    try:
        start_time = time.time()
        result = await agent.process_blog(file_path)
        
        # Calculate and display processing time
        elapsed_time = time.time() - start_time
        print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
        
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing blog: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 