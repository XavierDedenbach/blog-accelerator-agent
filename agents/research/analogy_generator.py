"""
Analogy Generator component for the Blog Accelerator Agent.

This module:
1. Generates powerful analogies to explain complex concepts
2. Creates cross-domain analogies from various fields
3. Evaluates and refines analogies for accuracy and clarity
4. Provides visual representations of analogies
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import asyncio

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableConfig

# Import Groq model dynamically to avoid import errors
try:
    from langchain_groq import ChatGroq
except ImportError:
    # Create a mock ChatGroq class if the import fails
    class ChatGroq:
        def __init__(self, *args, **kwargs):
            raise ImportError("langchain_groq is not installed. Install it with: pip install langchain-groq")

from agents.utilities.source_validator import SourceValidator
from agents.utilities.firecrawl_client import FirecrawlClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalogyGenerationError(Exception):
    """Exception raised for errors in analogy generation."""
    pass


class AnalogyGenerator:
    """
    Generator for powerful analogies to explain complex concepts.
    
    Features:
    - Generation of cross-domain analogies
    - Evaluation and refinement of analogies
    - Visual representation of analogies
    - Search for existing analogies in literature
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        source_validator: Optional[SourceValidator] = None,
        firecrawl_client: Optional[FirecrawlClient] = None,
        min_analogies: int = 3
    ):
        """
        Initialize the analogy generator.
        
        Args:
            openai_api_key: OpenAI API key
            groq_api_key: Groq API key
            source_validator: SourceValidator instance
            firecrawl_client: FirecrawlClient instance
            min_analogies: Minimum number of analogies to generate
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        # Try to use OpenAI first, fall back to Groq
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.5,
                openai_api_key=self.openai_api_key
            )
        elif self.groq_api_key:
            self.llm = ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.5,
                groq_api_key=self.groq_api_key
            )
        else:
            raise AnalogyGenerationError("No API key provided for LLM")
        
        # Initialize source validator if not provided
        self.source_validator = source_validator or SourceValidator()
        
        # Initialize firecrawl client if not provided
        self.firecrawl_client = firecrawl_client or FirecrawlClient()
        
        # Set minimum analogies threshold
        self.min_analogies = min_analogies
        
        # Define domain categories for cross-domain analogies
        self.domains = [
            "Nature & Biology",
            "Technology & Computing",
            "Sports & Games",
            "Art & Music",
            "History & Politics",
            "Economics & Business",
            "Psychology & Behavior",
            "Everyday Life",
            "Physics & Chemistry",
            "Literature & Storytelling"
        ]
        
        # Load prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompts for analogy generation."""
        self.generate_analogies_prompt = PromptTemplate(
            input_variables=["concept", "domains", "min_analogies"],
            template="""You are generating powerful analogies to explain a complex concept.

Concept to explain: {concept}

Your task is to generate at least {min_analogies} creative analogies from different domains that help explain this concept clearly. Consider domains such as:
{domains}

For each analogy:
1. Provide a clear title
2. Describe the analogy in detail (3-4 sentences)
3. Explain why this analogy works well for the concept
4. Identify the core mapping between the source domain and target concept
5. Note any limitations of the analogy

Format your response as a JSON array of analogy objects with these fields:
- title: A catchy title for the analogy
- domain: The domain this analogy draws from (one of the domains listed above)
- description: Detailed description of the analogy
- explanation: Why this analogy works well
- mapping: The core mapping between source domain and target concept
- limitations: Any limitations or where the analogy breaks down
- visual_description: A brief description of how this analogy could be visualized

Only respond with the JSON array. Include at least {min_analogies} diverse analogies from different domains.
"""
        )
        
        self.evaluate_analogy_prompt = PromptTemplate(
            input_variables=["concept", "analogy"],
            template="""You are evaluating an analogy designed to explain a complex concept.

Concept to explain: {concept}

Analogy: {analogy}

Your task is to carefully evaluate this analogy using the following criteria:
1. Clarity: How clearly does the analogy convey the concept?
2. Accuracy: How accurately does the analogy represent the concept?
3. Memorability: How likely is the analogy to be remembered?
4. Relatability: How relatable is the analogy to a general audience?
5. Educational value: How well does the analogy enhance understanding?

For each criterion, provide a score from 1-10 and a brief explanation.

Format your response as a JSON object with these fields:
- clarity: {{score: number, explanation: string}}
- accuracy: {{score: number, explanation: string}}
- memorability: {{score: number, explanation: string}}
- relatability: {{score: number, explanation: string}}
- educational_value: {{score: number, explanation: string}}
- overall_score: number (average of all scores)
- improvement_suggestions: Array of suggested improvements

Only respond with the JSON object.
"""
        )
        
        self.refine_analogy_prompt = PromptTemplate(
            input_variables=["concept", "analogy", "evaluation"],
            template="""You are refining an analogy based on evaluation feedback.

Concept to explain: {concept}

Original analogy:
{analogy}

Evaluation feedback:
{evaluation}

Your task is to refine this analogy to address the feedback while preserving its strengths. Make specific improvements to:
1. Enhance clarity where needed
2. Improve accuracy of the mapping
3. Make it more memorable and engaging
4. Increase relatability to the audience
5. Strengthen its educational value

Format your response as a JSON object with these fields:
- title: A refined title for the analogy
- domain: The domain this analogy draws from (keep the same or modify)
- description: Improved description of the analogy
- explanation: Enhanced explanation of why this analogy works
- mapping: Refined mapping between source domain and target concept
- limitations: Updated limitations or where the analogy breaks down
- visual_description: Improved description of how this analogy could be visualized
- changes_made: Summary of the key changes you made to improve the analogy

Only respond with the JSON object.
"""
        )
        
        self.search_existing_analogies_prompt = PromptTemplate(
            input_variables=["concept", "results"],
            template="""You are extracting existing analogies from search results about a concept.

Concept: {concept}

Search results:
{results}

Your task is to identify any existing analogies or metaphors used to explain this concept in the search results. For each analogy you find:
1. Identify the source of the analogy
2. Extract the analogy itself
3. Explain how it maps to the concept
4. Note any credited authors or origins of the analogy

Format your response as a JSON array of existing analogy objects with these fields:
- source: Where this analogy was found (URL or reference)
- analogy: The analogy itself
- mapping: How it maps to the concept
- credited_to: Who created or is credited with this analogy (if mentioned)

Only respond with the JSON array. If no clear analogies are found, return an empty array.
"""
        )
    
    async def generate_domain_analogies(
        self,
        concept: str,
        num_analogies: int = None
    ) -> List[Dict[str, Any]]:
        """
        Generate analogies from different domains for a concept.
        
        Args:
            concept: Concept to explain with analogies
            num_analogies: Number of analogies to generate (default: self.min_analogies)
            
        Returns:
            List of analogy dictionaries
        """
        if num_analogies is None:
            num_analogies = self.min_analogies
            
        try:
            # Format domains for prompt
            domains_text = "\n".join([f"- {domain}" for domain in self.domains])
            
            # Create chain for analogy generation
            chain = LLMChain(
                llm=self.llm,
                prompt=self.generate_analogies_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                concept=concept,
                domains=domains_text,
                min_analogies=num_analogies
            )
            
            # Parse JSON response
            analogies = json.loads(response)
            
            logger.info(f"Generated {len(analogies)} analogies for concept: {concept}")
            return analogies
            
        except Exception as e:
            logger.error(f"Error generating analogies for {concept}: {e}")
            raise AnalogyGenerationError(f"Failed to generate analogies: {e}")
    
    async def evaluate_analogy(
        self, 
        concept: str,
        analogy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate an analogy based on multiple criteria.
        
        Args:
            concept: Concept the analogy explains
            analogy: Analogy dictionary
            
        Returns:
            Dictionary with evaluation scores and feedback
        """
        try:
            # Format analogy for prompt
            analogy_text = f"Title: {analogy.get('title', 'Untitled Analogy')}\n"
            analogy_text += f"Domain: {analogy.get('domain', 'Unknown Domain')}\n"
            analogy_text += f"Description: {analogy.get('description', 'No description')}\n"
            analogy_text += f"Mapping: {analogy.get('mapping', 'No mapping specified')}\n"
            
            # Create chain for analogy evaluation
            chain = LLMChain(
                llm=self.llm,
                prompt=self.evaluate_analogy_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                concept=concept,
                analogy=analogy_text
            )
            
            # Parse JSON response
            evaluation = json.loads(response)
            
            logger.info(f"Evaluated analogy: {analogy.get('title', 'Untitled')} - Score: {evaluation.get('overall_score', 0)}/10")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating analogy {analogy.get('title', 'Untitled')}: {e}")
            raise AnalogyGenerationError(f"Failed to evaluate analogy: {e}")
    
    async def refine_analogy(
        self,
        concept: str,
        analogy: Dict[str, Any],
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine an analogy based on evaluation feedback.
        
        Args:
            concept: Concept the analogy explains
            analogy: Original analogy dictionary
            evaluation: Evaluation feedback dictionary
            
        Returns:
            Dictionary with refined analogy
        """
        try:
            # Format analogy for prompt
            analogy_text = f"Title: {analogy.get('title', 'Untitled Analogy')}\n"
            analogy_text += f"Domain: {analogy.get('domain', 'Unknown Domain')}\n"
            analogy_text += f"Description: {analogy.get('description', 'No description')}\n"
            analogy_text += f"Mapping: {analogy.get('mapping', 'No mapping specified')}\n"
            if 'limitations' in analogy:
                analogy_text += f"Limitations: {analogy.get('limitations')}\n"
            
            # Format evaluation for prompt
            evaluation_text = f"Overall score: {evaluation.get('overall_score', 0)}/10\n"
            evaluation_text += "Feedback:\n"
            for criterion, data in evaluation.items():
                if criterion != 'overall_score' and criterion != 'improvement_suggestions':
                    evaluation_text += f"- {criterion.replace('_', ' ').title()}: {data.get('score', 0)}/10 - {data.get('explanation', '')}\n"
            
            if 'improvement_suggestions' in evaluation:
                evaluation_text += "Improvement suggestions:\n"
                for i, suggestion in enumerate(evaluation['improvement_suggestions'], 1):
                    evaluation_text += f"{i}. {suggestion}\n"
            
            # Create chain for analogy refinement
            chain = LLMChain(
                llm=self.llm,
                prompt=self.refine_analogy_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                concept=concept,
                analogy=analogy_text,
                evaluation=evaluation_text
            )
            
            # Parse JSON response
            refined_analogy = json.loads(response)
            
            logger.info(f"Refined analogy: {analogy.get('title', 'Untitled')} -> {refined_analogy.get('title', 'Untitled')}")
            return refined_analogy
            
        except Exception as e:
            logger.error(f"Error refining analogy {analogy.get('title', 'Untitled')}: {e}")
            raise AnalogyGenerationError(f"Failed to refine analogy: {e}")
    
    async def search_existing_analogies(
        self,
        concept: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for existing analogies in literature and online sources.
        
        Args:
            concept: Concept to search analogies for
            count: Number of search results to analyze
            
        Returns:
            List of existing analogy dictionaries
        """
        try:
            # Search for sources about analogies for this concept
            query = f"{concept} analogy metaphor explanation"
            search_results = await self.source_validator.search_web(query, count=count)
            
            if not search_results:
                logger.warning(f"No search results found for analogies about {concept}")
                return []
            
            # Format search results for prompt
            results_text = ""
            for i, result in enumerate(search_results, 1):
                results_text += f"{i}. {result.get('title', 'Untitled')}\n"
                results_text += f"URL: {result.get('url', 'No URL')}\n"
                results_text += f"Description: {result.get('description', 'No description')}\n\n"
            
            # Create chain for extracting existing analogies
            chain = LLMChain(
                llm=self.llm,
                prompt=self.search_existing_analogies_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                concept=concept,
                results=results_text
            )
            
            # Parse JSON response
            existing_analogies = json.loads(response)
            
            logger.info(f"Found {len(existing_analogies)} existing analogies for concept: {concept}")
            return existing_analogies
            
        except Exception as e:
            logger.error(f"Error searching existing analogies for {concept}: {e}")
            raise AnalogyGenerationError(f"Failed to search existing analogies: {e}")
    
    async def generate_visual_representation(
        self,
        analogy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a visual representation for an analogy.
        
        Args:
            analogy: Analogy dictionary with visual_description
            
        Returns:
            Dictionary with visual asset information
        """
        try:
            # Extract visual description
            visual_description = analogy.get("visual_description", "")
            if not visual_description:
                visual_description = f"Visual representation of {analogy.get('title', 'analogy')}"
            
            # Create search query for visual
            query = f"{analogy.get('domain', '')} {analogy.get('title', '')} visual diagram illustration"
            
            # Use FirecrawlClient to find visual assets
            visual_assets = []
            try:
                if self.firecrawl_client:
                    visual_assets = await self.firecrawl_client.search_images(query)
                
                # If no images found with Firecrawl, try Brave API if available
                if not visual_assets and self.source_validator:
                    logger.info(f"Falling back to Brave API for images for analogy: {analogy.get('title', 'Untitled')}")
                    try:
                        # Use search_web directly and await the result
                        search_results = await self.source_validator.search_web(f"{query} image", count=5)
                        
                        # Ensure search_results is iterable (e.g., a list) before processing
                        if isinstance(search_results, list):
                            # Format results as visual assets
                            visual_assets = [
                                {
                                    "url": result.get("url", ""),
                                    "title": result.get("title", ""),
                                    "description": result.get("description", ""),
                                    "source": result.get("source", "Brave Search") # Default source if missing
                                }
                                for result in search_results
                            ]
                        else:
                            logger.warning(f"Brave search fallback for {analogy.get('title', 'Untitled')} did not return a list. Received: {type(search_results)}")
                            visual_assets = [] # Keep visual_assets empty if search failed
                    except Exception as e:
                        # Log the specific error during Brave fallback
                        logger.error(f"Error searching for images with Brave API fallback: {e}", exc_info=True)
                        visual_assets = [] # Ensure visual_assets remains empty on error
            except Exception as e:
                logger.error(f"Error searching for images: {e}")
                visual_assets = []
            
            if not visual_assets:
                logger.warning(f"No visual assets found for analogy: {analogy.get('title', 'Untitled')}")
                return {"success": False, "assets": []}
            
            logger.info(f"Found {len(visual_assets)} visual assets for analogy: {analogy.get('title', 'Untitled')}")
            return {
                "success": True,
                "assets": visual_assets,
                "query": query,
                "visual_description": visual_description
            }
            
        except Exception as e:
            logger.error(f"Error generating visual for analogy {analogy.get('title', 'Untitled')}: {e}")
            return {"success": False, "error": str(e), "assets": []}
    
    async def generate_analogies(
        self,
        concept: str,
        refinement_threshold: float = 7.0
    ) -> Dict[str, Any]:
        """
        Perform complete analogy generation process for a concept.
        
        Args:
            concept: Concept to explain with analogies
            refinement_threshold: Minimum score for analogies (below this will be refined)
            
        Returns:
            Dictionary with generated analogies and metadata
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Generate initial analogies
            logger.info(f"Generating analogies for concept: {concept}")
            initial_analogies = await self.generate_domain_analogies(concept)
            
            # Search for existing analogies
            logger.info(f"Searching for existing analogies for concept: {concept}")
            existing_analogies = await self.search_existing_analogies(concept)
            
            # Evaluate and refine analogies
            evaluated_analogies = []
            refined_analogies = []
            
            for analogy in initial_analogies:
                logger.info(f"Evaluating analogy: {analogy.get('title', '')}")
                try:
                    evaluation = await self.evaluate_analogy(concept, analogy)
                    
                    # Ensure evaluation and score exist and are floats before comparison
                    overall_score = evaluation.get('overall_score')
                    if isinstance(overall_score, (int, float)):
                        logger.info(f"Evaluated analogy: {analogy.get('title', '')} - Score: {overall_score:.1f}/10")
                        evaluated_analogy = {**analogy, "evaluation": evaluation}
                        evaluated_analogies.append(evaluated_analogy)
                        
                        # Refine if below threshold
                        if overall_score < refinement_threshold:
                            logger.info(f"Refining analogy: {analogy.get('title', '')} (Score: {overall_score:.1f})")
                            refined_analogy = await self.refine_analogy(concept, analogy, evaluation)
                            refined_analogies.append(refined_analogy)
                        else:
                            # Keep the original if score is good enough
                            refined_analogies.append(evaluated_analogy) # Add the evaluated version
                    else:
                        logger.warning(f"Could not evaluate analogy '{analogy.get('title', '')}' properly, score missing or invalid. Skipping.")
                        # Optionally keep the original unevaluated analogy
                        # refined_analogies.append(analogy) 
                        
                except Exception as e:
                     logger.error(f"Error evaluating or refining analogy '{analogy.get('title', '')}': {e}")
                     # Optionally keep the original unevaluated analogy
                     # refined_analogies.append(analogy) 

            # Generate visual representations
            logger.info(f"Generating visual representations for {len(refined_analogies)} refined analogies")
            visual_representations = await asyncio.gather(*[self.generate_visual_representation(a) for a in refined_analogies])
            
            # Calculate statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = {
                "concept": concept,
                "generated_analogies": evaluated_analogies + refined_analogies,
                "existing_analogies": existing_analogies,
                "stats": {
                    "generated_count": len(evaluated_analogies) + len(refined_analogies),
                    "existing_count": len(existing_analogies),
                    "visual_assets_count": sum(len(a.get("visual", {}).get("assets", [])) for a in evaluated_analogies + refined_analogies),
                    "average_score": sum(a.get("evaluation", {}).get("overall_score", 0) for a in evaluated_analogies + refined_analogies) / max(len(evaluated_analogies + refined_analogies), 1),
                    "generation_duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(
                f"Completed analogy generation for {concept} with "
                f"{len(evaluated_analogies) + len(refined_analogies)} analogies"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analogy generation for {concept}: {e}")
            raise AnalogyGenerationError(f"Failed to complete analogy generation: {e}") 