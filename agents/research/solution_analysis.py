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
import re # Import re for regex

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

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
        llm: BaseChatModel,
        source_validator: SourceValidator
    ):
        """
        Initialize the Solution Analyzer.
        
        Args:
            llm: Language model instance.
            source_validator: SourceValidator instance.
        """
        self.initial_llm = llm
        self.source_validator = source_validator
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

**Step 1: Identify Core Constraints**
Briefly consider the challenges listed and the typical constraints faced by small, resource-limited teams (e.g., funding, personnel, time-to-market pressure, hardware complexities) when implementing a solution like '{solution}'. What are the key limitations and boundaries?

**Step 2: Consider Systemic Context**
Reflect on how '{solution}' fits within the broader ecosystem and market landscape. What upstream and downstream dependencies might affect its implementation and success?

**Step 3: Map Stakeholder Perspectives**
Consider the various stakeholders (developers, users, investors, etc.) who would be affected by or involved in implementing '{solution}'. How might their different perspectives influence the value assessment?

**Step 4: Identify Potential Benefits**
Based on your reflections, generate at least {min_arguments} compelling PRO arguments supporting this solution in this specific context.
For each argument:
1. Provide a clear name/title
2. Write a detailed description (2-3 sentences) explaining why this is a strong argument, linking back to the constraints or challenges where relevant.
3. Specify what needs to be true for this argument to be valid (prerequisites), considering the context.
4. Suggest metrics that could measure the success of this aspect of the solution, relevant to the constraints.

**Step 5: Generate Supporting Evidence**
For each argument, consider what evidence would support this claim. What data points, case studies, or examples might strengthen this argument?

**Step 6: Test Counter-Arguments**
For each PRO argument, briefly identify one potential counter-argument and how you would respond to it. This will strengthen the original argument by addressing potential weaknesses.

Format your response as a JSON array of argument objects with these fields:
- name: Title of the argument
- description: Detailed description linking to context/constraints
- prerequisites: Context-aware prerequisites
- metrics: Array of relevant metrics
- supporting_evidence: Ideas for evidence that would strengthen this argument
- potential_counter: A potential counter-argument to this point
- counter_response: Brief response to the counter-argument
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

**Step 1: Identify Core Constraints**
Briefly consider the challenges listed and the typical constraints faced by small, resource-limited teams (e.g., funding, personnel, time-to-market pressure, hardware complexities) when implementing a solution like '{solution}'. What are the key limitations and boundaries?

**Step 2: Consider Systemic Context**
Reflect on how '{solution}' fits within the broader ecosystem and market landscape. What external factors might create challenges or vulnerabilities?

**Step 3: Map Stakeholder Perspectives**
Consider the various stakeholders (developers, users, investors, etc.) who would be affected by or involved in implementing '{solution}'. What concerns might they have or what resistance might they show?

**Step 4: Identify Potential Challenges**
Based on your reflections, generate at least {min_arguments} insightful COUNTER arguments or potential weaknesses of this solution in this specific context.
For each argument:
1. Provide a clear name/title
2. Write a detailed description (2-3 sentences) explaining the potential weakness or risk, linking back to the constraints or challenges where relevant.
3. Specify what needs to be true for this counter-argument to be significant (conditions).
4. Suggest ways this risk could potentially be mitigated, considering the constraints.

**Step 5: Generate Supporting Evidence**
For each counter-argument, consider what evidence would support this concern. What data points, case studies, or examples might validate this potential issue?

**Step 6: Test Pro-Arguments**
For each COUNTER argument, briefly identify one potential pro-argument that attempts to dismiss this concern and explain why the concern remains valid despite this defense.

Format your response as a JSON array of argument objects with these fields:
- name: Title of the counter-argument
- description: Detailed description linking to context/constraints
- conditions: Conditions making this counter-argument significant
- mitigation_ideas: Array of potential mitigation strategies (context-aware)
- supporting_evidence: Ideas for evidence that would strengthen this concern
- potential_defense: A potential argument that attempts to dismiss this concern
- defense_rebuttal: Brief explanation of why the concern remains valid
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

**Step 1: Identify Core Constraints**
Briefly consider the challenges, the proposed solution, and the typical constraints faced by small, resource-limited teams (e.g., funding, personnel, time-to-market pressure, hardware complexities). What are the key limitations affecting measurement?

**Step 2: Consider Systemic Context**
Reflect on how '{solution}' fits within the broader ecosystem and market landscape. What contextual factors should influence how success is measured?

**Step 3: Map Stakeholder Perspectives**
Consider the various stakeholders (developers, users, investors, etc.) who would evaluate the success of '{solution}'. What metrics would matter most to each group?

**Step 4: Identify Key Metrics**
Based on your reflections, identify a set of key metrics (aim for 5-10) that would effectively measure the progress and impact of the solution '{solution}' in this specific context.
For each metric:
1. Provide a clear name
2. Explain *why* this metric is important in the context of the challenges and constraints.
3. Describe *how* it could be measured practically by a resource-constrained team.
4. Suggest target ranges or indicators of success/failure where applicable.

**Step 5: Generate Supporting Evidence**
For each metric, briefly describe what existing benchmarks or industry standards might exist that could help contextualize the measurements.

**Step 6: Test Counter-Arguments**
For each metric, briefly address one potential criticism about why this metric might be misleading or insufficient, and how you would respond to ensure the metric remains valuable.

Format your response as a JSON array of metric objects with these fields:
- name: Name of the metric
- importance_context: Explanation of why it's important in context
- measurement_method: Practical measurement approach for constrained teams
- success_indicators: Target ranges or indicators
- benchmarks: Existing benchmarks or standards for context
- potential_criticism: A potential criticism of this metric
- criticism_response: How to address the criticism

Only respond with the JSON array.
"""
        )
    
    async def generate_pro_arguments(
        self,
        solution_title: str,
        topic: str,
        challenges: List[str],
        llm_override: Optional[BaseChatModel] = None
    ) -> List[str]:
        """Generate pro arguments for the proposed solution."""
        logger.info(f"Generating pro arguments for solution: {solution_title} with context reflection...")
        try:
            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.pro_arguments_prompt
            )
            # Pass topic and challenges to the LLM chain
            response = await chain.arun(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)
            arguments = self._parse_llm_response_to_json(response, f"pro arguments for {solution_title}")
            logger.info(f"Generated {len(arguments)} pro arguments for solution: {solution_title}")
            return arguments
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error generating pro arguments for {solution_title}: {e}")
            logger.error(f"LLM Response was: {response if 'response' in locals() else 'Response text not captured'}")
            raise SolutionAnalysisError(f"Failed to parse pro arguments JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating pro arguments for {solution_title}: {e}")
            raise SolutionAnalysisError(f"Failed to generate pro arguments: {e}")
    
    async def generate_counter_arguments(
        self,
        solution_title: str,
        topic: str,
        challenges: List[str],
        llm_override: Optional[BaseChatModel] = None
    ) -> List[str]:
        """Generate counter arguments against the proposed solution."""
        logger.info(f"Generating counter arguments for solution: {solution_title} with context reflection...")
        try:
            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.counter_arguments_prompt
            )
            # Pass topic and challenges to the LLM chain
            response = await chain.arun(solution=solution_title, topic=topic, challenges="\n".join(challenges), min_arguments=5)
            arguments = self._parse_llm_response_to_json(response, f"counter arguments for {solution_title}")
            logger.info(f"Generated {len(arguments)} counter arguments for solution: {solution_title}")
            return arguments
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error generating counter arguments for {solution_title}: {e}")
            logger.error(f"LLM Response was: {response if 'response' in locals() else 'Response text not captured'}")
            raise SolutionAnalysisError(f"Failed to parse counter arguments JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating counter arguments for {solution_title}: {e}")
            raise SolutionAnalysisError(f"Failed to generate counter arguments: {e}")
    
    async def identify_metrics(
        self,
        solution_title: str,
        topic: str,
        challenges: List[str],
        llm_override: Optional[BaseChatModel] = None
    ) -> List[Dict[str, str]]:
        """Identify key metrics for the proposed solution."""
        logger.info(f"Identifying metrics for solution: {solution_title} with context reflection...")
        try:
            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.identify_metrics_prompt
            )
            # Pass topic and challenges to the LLM chain
            # The prompt for identify_metrics expects 'topic', 'solution', and 'challenges'.
            # 'min_metrics' is not a direct input to this prompt based on its definition but is handled by the prompt's instruction to aim for 5-10 metrics.
            response = await chain.arun(solution=solution_title, topic=topic, challenges="\n".join(challenges))
            metrics = self._parse_llm_response_to_json(response, f"metrics for {solution_title}")
            logger.info(f"Identified {len(metrics)} metrics for solution: {solution_title}")
            return metrics
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error identifying metrics for {solution_title}: {e}")
            logger.error(f"LLM Response was: {response if 'response' in locals() else 'Response text not captured'}")
            raise SolutionAnalysisError(f"Failed to parse metrics JSON: {e}")
        except Exception as e:
            logger.error(f"Error identifying metrics for {solution_title}: {e}")
            raise SolutionAnalysisError(f"Failed to identify metrics: {e}")
    
    async def find_argument_sources(
        self,
        argument: str,
        solution_title: str,
        is_pro_argument: bool
    ) -> List[Dict[str, Any]]:
        """Find supporting and opposing sources for an argument."""
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
                    query, count=3
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
        solution_title: str,
        topic: str,
        challenges: List[str],
        llm_override: Optional[BaseChatModel] = None
    ) -> Dict[str, Any]:
        """
        Analyze the solution by generating pro/counter arguments and metrics.
        Also finds sources for arguments.

        Args:
            solution_title: The title of the solution to analyze.
            topic: The broader topic or domain of the solution.
            challenges: A list of challenges the solution aims to address.
            llm_override: Optional language model to override the default.

        Returns:
            A dictionary containing the analysis results, including title, args, metrics, and stats.
        """
        logger.info(f"Starting full solution analysis for: {solution_title} within topic: {topic}")
        start_time = datetime.now() # For stats
        pro_arguments_list = []
        counter_arguments_list = []
        metrics_list = []
        errors = []

        try:
            pro_arguments_list = await self.generate_pro_arguments(
                solution_title=solution_title, 
                topic=topic, 
                challenges=challenges, 
                llm_override=llm_override
            )
        except SolutionAnalysisError as e:
            logger.error(f"Error generating pro arguments in full analysis: {e}")
            errors.append(f"Pro arguments generation failed: {e}")
        
        try:
            counter_arguments_list = await self.generate_counter_arguments(
                solution_title=solution_title, 
                topic=topic, 
                challenges=challenges, 
                llm_override=llm_override
            )
        except SolutionAnalysisError as e:
            logger.error(f"Error generating counter arguments in full analysis: {e}")
            errors.append(f"Counter arguments generation failed: {e}")
        
        try:
            metrics_list = await self.identify_metrics(
                solution_title=solution_title, 
                topic=topic, 
                challenges=challenges, 
                llm_override=llm_override
            )
        except SolutionAnalysisError as e:
            logger.error(f"Error identifying metrics in full analysis: {e}")
            errors.append(f"Metrics identification failed: {e}")

        # Process pro arguments for sources
        processed_pro_arguments = []
        total_pro_sources = 0
        for arg_data in pro_arguments_list: # Use the list fetched, not from a temp results dict
            if isinstance(arg_data, dict) and "name" in arg_data and "search_terms" in arg_data:
                try:
                    sources = await self.find_argument_sources(
                        argument=arg_data, 
                        solution_title=solution_title, 
                        is_pro_argument=True
                    )
                    arg_data["sources"] = sources
                    total_pro_sources += len(sources)
                except SolutionAnalysisError as e:
                    logger.error(f"Error finding sources for pro argument '{arg_data.get('name', 'Unknown')}': {e}")
                    arg_data["sources"] = [] 
                    errors.append(f"Source finding for pro arg '{arg_data.get('name', 'Unknown')}' failed: {e}")
                processed_pro_arguments.append(arg_data)
            else:
                logger.warning(f"Skipping source finding for malformed pro argument: {arg_data}")
                processed_pro_arguments.append(arg_data) # Still add it, maybe without sources

        # Process counter arguments for sources
        processed_counter_arguments = []
        total_con_sources = 0
        for arg_data in counter_arguments_list: # Use the list fetched
            if isinstance(arg_data, dict) and "name" in arg_data and "search_terms" in arg_data:
                try:
                    sources = await self.find_argument_sources(
                        argument=arg_data, 
                        solution_title=solution_title, 
                        is_pro_argument=False
                    )
                    arg_data["sources"] = sources
                    total_con_sources += len(sources)
                except SolutionAnalysisError as e:
                    logger.error(f"Error finding sources for counter argument '{arg_data.get('name', 'Unknown')}': {e}")
                    arg_data["sources"] = []
                    errors.append(f"Source finding for counter arg '{arg_data.get('name', 'Unknown')}' failed: {e}")
                processed_counter_arguments.append(arg_data)
            else:
                logger.warning(f"Skipping source finding for malformed counter argument: {arg_data}")
                processed_counter_arguments.append(arg_data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        final_result = {
            "solution_title": solution_title,
            "topic": topic, # Include topic in results
            "challenges": challenges, # Include challenges in results
            "pro_arguments": processed_pro_arguments,
            "counter_arguments": processed_counter_arguments,
            "metrics": metrics_list,
            "errors": errors,
            "stats": {
                "pro_args_count": len(processed_pro_arguments),
                "counter_args_count": len(processed_counter_arguments),
                "metrics_count": len(metrics_list),
                "total_pro_sources": total_pro_sources,
                "total_con_sources": total_con_sources,
                "error_count": len(errors),
                "analysis_duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                "total_stages": 3
            }
        }

        if errors:
            logger.warning(f"Completed solution analysis for {solution_title} with {len(errors)} errors.")
        else:
            logger.info(f"Successfully completed solution analysis for {solution_title}.")
        
        return final_result 

    def _parse_llm_response_to_json(self, response_text: str, context: str) -> Any:
        """Helper to parse LLM JSON response, with error handling and logging."""
        # Attempt to extract JSON from markdown code block using regex
        match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", response_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1).strip()
        else:
            # If no markdown fence pattern is matched, assume the response_text is the JSON string itself (or already cleaned)
            json_str = response_text.strip()

        # ADDED: Debug log to see exactly what is being parsed
        logger.debug(f"Attempting to parse JSON (first 1000 chars): '{json_str[:1000]}'")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error {context}: {e}")
            logger.error(f"LLM Response (after attempted cleaning) was: {json_str}")
            logger.error(f"Original LLM Response was: {response_text}")
            raise SolutionAnalysisError(f"Failed to parse {context} JSON: {e}") from e 