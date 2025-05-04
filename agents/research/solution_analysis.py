"""
Solution Analysis component for the Blog Accelerator Agent.

This module:
1. Analyzes proposed solutions for challenges
2. Generates 5-10 pro arguments with supporting evidence
3. Generates 5-10 counter arguments with supporting evidence
4. Identifies key metrics for measuring solution progress
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SolutionAnalysisError(Exception):
    """Exception raised for errors in solution analysis."""
    pass


class SolutionAnalyzer:
    """
    Analyzer for proposed solutions to industry challenges.
    
    Features:
    - Generation of 5-10 pro arguments
    - Generation of 5-10 counter arguments
    - Identification of metrics for measuring progress
    - Analysis of prerequisites for arguments to be valid
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        source_validator: Optional[SourceValidator] = None,
        min_arguments: int = 5
    ):
        """
        Initialize the solution analyzer.
        
        Args:
            openai_api_key: OpenAI API key
            groq_api_key: Groq API key
            source_validator: SourceValidator instance
            min_arguments: Minimum number of arguments to generate
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        # Try to use OpenAI first, fall back to Groq
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.3,
                openai_api_key=self.openai_api_key
            )
        elif self.groq_api_key:
            self.llm = ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.3,
                groq_api_key=self.groq_api_key
            )
        else:
            raise SolutionAnalysisError("No API key provided for LLM")
        
        # Initialize source validator if not provided
        self.source_validator = source_validator or SourceValidator()
        
        # Set minimum arguments threshold
        self.min_arguments = min_arguments
        
        # Load prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompts for solution analysis."""
        self.pro_arguments_prompt = PromptTemplate(
            input_variables=["topic", "solution", "challenges", "min_arguments"],
            template="""You are analyzing a proposed solution to industry challenges, considering the context of resource-constrained teams (e.g., hardware startups).

Topic: {topic}
Proposed Solution: {solution}

The solution aims to address the following challenges:
{challenges}

**Step 1: Reflect on Context & Constraints**
Briefly consider the challenges listed and the typical constraints faced by small, resource-limited teams (e.g., funding, personnel, time-to-market pressure, hardware complexities) when implementing a solution like '{solution}'. How might these factors influence the potential benefits?

**Step 2: Generate PRO Arguments**
Based on your reflection, generate at least {min_arguments} compelling PRO arguments supporting this solution *in this specific context*.
For each argument:
1. Provide a clear name/title
2. Write a detailed description (2-3 sentences) explaining why this is a strong argument, linking back to the constraints or challenges where relevant.
3. Specify what needs to be true for this argument to be valid (prerequisites), considering the context.
4. Suggest metrics that could measure the success of this aspect of the solution, relevant to the constraints.
5. Suggest search terms to find supporting evidence for this argument, considering the nuances.

Format your response as a JSON array of argument objects with these fields:
- name: Title of the argument
- description: Detailed description linking to context/constraints
- prerequisites: Context-aware prerequisites
- metrics: Array of relevant metrics
- search_terms: Array of nuanced search terms

Only respond with the JSON array. Include at least {min_arguments} arguments.
"""
        )
        
        self.counter_arguments_prompt = PromptTemplate(
            input_variables=["topic", "solution", "challenges", "min_arguments"],
            template="""You are analyzing potential weaknesses or counter-arguments for a proposed solution, considering the context of resource-constrained teams (e.g., hardware startups).

Topic: {topic}
Proposed Solution: {solution}

The solution aims to address the following challenges:
{challenges}

**Step 1: Reflect on Context & Constraints**
Briefly consider the challenges listed and the typical constraints faced by small, resource-limited teams (e.g., funding, personnel, time-to-market pressure, hardware complexities) when implementing a solution like '{solution}'. How might these factors introduce risks or downsides?

**Step 2: Generate COUNTER Arguments**
Based on your reflection, generate at least {min_arguments} insightful COUNTER arguments or potential weaknesses of this solution *in this specific context*.
For each argument:
1. Provide a clear name/title
2. Write a detailed description (2-3 sentences) explaining the potential weakness or risk, linking back to the constraints or challenges where relevant.
3. Specify what needs to be true for this counter-argument to be significant (conditions).
4. Suggest ways this risk could potentially be mitigated, considering the constraints.
5. Suggest search terms to find evidence supporting this counter-argument or exploring its validity.

Format your response as a JSON array of argument objects with these fields:
- name: Title of the counter-argument
- description: Detailed description linking to context/constraints
- conditions: Conditions making this counter-argument significant
- mitigation_ideas: Array of potential mitigation strategies (context-aware)
- search_terms: Array of nuanced search terms

Only respond with the JSON array. Include at least {min_arguments} arguments.
"""
        )
        
        self.identify_metrics_prompt = PromptTemplate(
            input_variables=["topic", "solution", "challenges"],
            template="""You are identifying key metrics to measure the progress and success of a proposed solution, considering the context of resource-constrained teams.

Topic: {topic}
Proposed Solution: {solution}

The solution aims to address the following challenges:
{challenges}

**Step 1: Reflect on Context & Constraints**
Briefly consider the challenges, the proposed solution, and the typical constraints faced by small, resource-limited teams (e.g., funding, personnel, time-to-market pressure, hardware complexities). What aspects of success or failure would be most critical to track for such a team implementing this solution?

**Step 2: Identify Key Metrics**
Based on your reflection, identify a set of key metrics (aim for 5-10) that would effectively measure the progress and impact of the solution '{solution}' *in this specific context*.
For each metric:
1. Provide a clear name
2. Explain *why* this metric is important in the context of the challenges and constraints.
3. Describe *how* it could be measured practically by a resource-constrained team.
4. Suggest target ranges or indicators of success/failure where applicable.

Format your response as a JSON array of metric objects with these fields:
- name: Name of the metric
- importance_context: Explanation of why it's important in context
- measurement_method: Practical measurement approach for constrained teams
- success_indicators: Target ranges or indicators

Only respond with the JSON array.
"""
        )
    
    async def generate_pro_arguments(
        self,
        topic: str,
        solution: str,
        challenges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate pro arguments for the solution, reflecting on context first.

        Args:
            topic: Topic being analyzed
            solution: Proposed solution description
            challenges: List of challenge dictionaries

        Returns:
            List of pro argument dictionaries
        """
        try:
            # Format challenges for prompt
            challenges_text = "\n".join(
                [f"- {c.get('name', '')}: {c.get('description', '')}" for c in challenges]
            )

            # Create chain for pro arguments
            chain = LLMChain(
                llm=self.llm,
                prompt=self.pro_arguments_prompt
            )

            # Run the chain
            logger.info(f"Generating pro arguments for solution: {solution} with context reflection...")
            response = await chain.arun(
                topic=topic,
                solution=solution,
                challenges=challenges_text,
                min_arguments=self.min_arguments
            )

            # Parse JSON response
            arguments = json.loads(response)
            
            logger.info(f"Generated {len(arguments)} pro arguments for solution: {solution}")
            return arguments
            
        except Exception as e:
            logger.error(f"Error generating pro arguments for {solution}: {e}")
            raise SolutionAnalysisError(f"Failed to generate pro arguments: {e}")
    
    async def generate_counter_arguments(
        self,
        topic: str,
        solution: str,
        challenges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate counter arguments for the solution, reflecting on context first.

        Args:
            topic: Topic being analyzed
            solution: Proposed solution description
            challenges: List of challenge dictionaries
            
        Returns:
            List of counter argument dictionaries
        """
        try:
            # Format challenges for prompt
            challenges_text = "\n".join(
                [f"- {c.get('name', '')}: {c.get('description', '')}" for c in challenges]
            )
            
            # Create chain for counter arguments
            chain = LLMChain(
                llm=self.llm,
                prompt=self.counter_arguments_prompt
            )
            
            # Run the chain
            logger.info(f"Generating counter arguments for solution: {solution} with context reflection...")
            response = await chain.arun(
                topic=topic,
                solution=solution,
                challenges=challenges_text,
                min_arguments=self.min_arguments
            )
            
            # Parse JSON response
            arguments = json.loads(response)
            
            logger.info(f"Generated {len(arguments)} counter arguments for solution: {solution}")
            return arguments
            
        except Exception as e:
            logger.error(f"Error generating counter arguments for {solution}: {e}")
            raise SolutionAnalysisError(f"Failed to generate counter arguments: {e}")
    
    async def identify_metrics(
        self,
        topic: str,
        solution: str,
        challenges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify key metrics for the solution, reflecting on context first.

        Args:
            topic: Topic being analyzed
            solution: Proposed solution description
            challenges: List of challenge dictionaries
            
        Returns:
            List of metric dictionaries
        """
        try:
            # Format challenges for prompt
            challenges_text = "\n".join(
                [f"- {c.get('name', '')}: {c.get('description', '')}" for c in challenges]
            )
            
            # Create chain for identifying metrics
            chain = LLMChain(
                llm=self.llm,
                prompt=self.identify_metrics_prompt
            )
            
            # Run the chain
            logger.info(f"Identifying metrics for solution: {solution} with context reflection...")
            response = await chain.arun(
                topic=topic,
                solution=solution,
                challenges=challenges_text
            )
            
            # Parse JSON response
            metrics = json.loads(response)
            
            logger.info(f"Identified {len(metrics)} metrics for solution: {solution}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error identifying metrics for {solution}: {e}")
            raise SolutionAnalysisError(f"Failed to identify metrics: {e}")
    
    async def find_sources_for_argument(
        self, 
        argument: Dict[str, Any],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find sources that support a pro or counter argument.
        
        Args:
            argument: Argument dictionary
            count: Number of sources to find
            
        Returns:
            List of source dictionaries
        """
        all_sources = []
        search_terms = argument.get("search_terms", [])
        
        if not search_terms:
            # Fallback if no search terms provided
            search_terms = [argument.get("name", "")]
        
        # Use up to 3 search terms
        for term in search_terms[:3]:
            try:
                # Search for sources
                query = f"{term} {argument.get('name', '')}"
                supporting, _ = await self.source_validator.find_supporting_contradicting_sources(
                    query, count=count
                )
                
                # Add sources to list
                all_sources.extend(supporting)
                
            except Exception as e:
                logger.warning(f"Error finding sources for term '{term}': {e}")
        
        # Remove duplicates based on URL
        unique_sources = []
        seen_urls = set()
        
        for source in all_sources:
            url = source.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        
        logger.info(f"Found {len(unique_sources)} sources for argument: {argument.get('name', '')}")
        return unique_sources
    
    async def analyze_solution(
        self,
        topic: str,
        solution: str,
        challenges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform complete solution analysis for a topic and proposed solution.

        Args:
            topic: Topic being analyzed
            solution: Proposed solution description
            challenges: List of challenge dictionaries
            
        Returns:
            Dictionary with solution analysis results
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Generate pro arguments
            logger.info(f"Generating pro arguments for solution: {solution}")
            pro_arguments = await self.generate_pro_arguments(topic, solution, challenges)
            
            # Generate counter arguments
            logger.info(f"Generating counter arguments for solution: {solution}")
            counter_arguments = await self.generate_counter_arguments(topic, solution, challenges)
            
            # Identify metrics
            logger.info(f"Identifying metrics for solution: {solution}")
            metrics = await self.identify_metrics(topic, solution, challenges)
            
            # Find sources for pro arguments
            enriched_pro_arguments = []
            for argument in pro_arguments:
                # Find sources
                logger.info(f"Finding sources for pro argument: {argument.get('name', '')}")
                sources = await self.find_sources_for_argument(argument)
                
                # Add to enriched arguments
                enriched_argument = {
                    **argument,
                    "sources": sources
                }
                enriched_pro_arguments.append(enriched_argument)
            
            # Find sources for counter arguments
            enriched_counter_arguments = []
            for argument in counter_arguments:
                # Find sources
                logger.info(f"Finding sources for counter argument: {argument.get('name', '')}")
                sources = await self.find_sources_for_argument(argument)
                
                # Add to enriched arguments
                enriched_argument = {
                    **argument,
                    "sources": sources
                }
                enriched_counter_arguments.append(enriched_argument)
            
            # Calculate statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = {
                "topic": topic,
                "solution": solution,
                "pro_arguments": enriched_pro_arguments,
                "counter_arguments": enriched_counter_arguments,
                "metrics": metrics,
                "stats": {
                    "pro_arguments_count": len(enriched_pro_arguments),
                    "counter_arguments_count": len(enriched_counter_arguments),
                    "metrics_count": len(metrics),
                    "sources_count": sum(len(a.get("sources", [])) for a in enriched_pro_arguments + enriched_counter_arguments),
                    "analysis_duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(
                f"Completed solution analysis for {solution} with "
                f"{len(enriched_pro_arguments)} pro arguments, "
                f"{len(enriched_counter_arguments)} counter arguments, and "
                f"{len(metrics)} metrics"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in solution analysis for {solution}: {e}")
            raise SolutionAnalysisError(f"Failed to complete solution analysis: {e}") 