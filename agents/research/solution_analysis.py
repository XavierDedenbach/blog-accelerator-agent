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
            response = await chain.arun(solution=solution_title, min_arguments=5)
            arguments = json.loads(response)["arguments"]
            logger.info(f"Generated {len(arguments)} pro arguments for solution: {solution_title}")
            return arguments
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error generating pro arguments for {solution_title}: {e}")
            logger.error(f"LLM Response was: {response}")
            raise SolutionAnalysisError(f"Failed to parse pro arguments JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating pro arguments for {solution_title}: {e}")
            raise SolutionAnalysisError(f"Failed to generate pro arguments: {e}")
    
    async def generate_counter_arguments(
        self,
        solution_title: str,
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
            response = await chain.arun(solution=solution_title, min_arguments=5)
            arguments = json.loads(response)["arguments"]
            logger.info(f"Generated {len(arguments)} counter arguments for solution: {solution_title}")
            return arguments
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error generating counter arguments for {solution_title}: {e}")
            logger.error(f"LLM Response was: {response}")
            raise SolutionAnalysisError(f"Failed to parse counter arguments JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating counter arguments for {solution_title}: {e}")
            raise SolutionAnalysisError(f"Failed to generate counter arguments: {e}")
    
    async def identify_metrics(
        self,
        solution_title: str,
        llm_override: Optional[BaseChatModel] = None
    ) -> List[Dict[str, str]]:
        """Identify key metrics to measure the solution's success."""
        logger.info(f"Identifying metrics for solution: {solution_title} with context reflection...")
        try:
            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.identify_metrics_prompt
            )
            response = await chain.arun(solution=solution_title, min_metrics=5)
            metrics = json.loads(response)["metrics"]
            logger.info(f"Identified {len(metrics)} metrics for solution: {solution_title}")
            return metrics
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error identifying metrics for {solution_title}: {e}")
            logger.error(f"LLM Response was: {response}")
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
        solution_title: str = "Proposed Solution",
        llm_override: Optional[BaseChatModel] = None
    ) -> Dict[str, Any]:
        """Perform full analysis for a given solution."""
        start_time = datetime.now()
        logger.info(f"Starting solution analysis for: {solution_title}")

        try:
            # Generate arguments
            pro_arguments_list = await self.generate_pro_arguments(solution_title, llm_override=llm_override)
            counter_arguments_list = await self.generate_counter_arguments(solution_title, llm_override=llm_override)
            metrics_list = await self.identify_metrics(solution_title, llm_override=llm_override)

            # Find sources for arguments
            pro_arguments = []
            total_pro_sources = 0
            for arg in pro_arguments_list:
                try:
                    logger.info(f"Finding sources for pro argument: {arg}")
                    sources = await self.find_argument_sources(arg, solution_title, is_pro_argument=True)
                    pro_arguments.append({"argument": arg, "sources": sources})
                    total_pro_sources += len(sources)
                except Exception as e:
                    logger.error(f"Error finding sources for pro argument '{arg}': {e}")
                    pro_arguments.append({"argument": arg, "sources": [], "error": str(e)})

            counter_arguments = []
            total_con_sources = 0
            for arg in counter_arguments_list:
                try:
                    logger.info(f"Finding sources for counter argument: {arg}")
                    sources = await self.find_argument_sources(arg, solution_title, is_pro_argument=False)
                    counter_arguments.append({"argument": arg, "sources": sources})
                    total_con_sources += len(sources)
                except Exception as e:
                    logger.error(f"Error finding sources for counter argument '{arg}': {e}")
                    counter_arguments.append({"argument": arg, "sources": [], "error": str(e)})

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = {
                "solution_title": solution_title,
                "pro_arguments": pro_arguments,
                "counter_arguments": counter_arguments,
                "metrics": metrics_list,
                "stats": {
                    "pro_args_count": len(pro_arguments),
                    "counter_args_count": len(counter_arguments),
                    "metrics_count": len(metrics_list),
                    "total_pro_sources": total_pro_sources,
                    "total_con_sources": total_con_sources,
                    "analysis_duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
            }

            logger.info(
                f"Completed solution analysis for {solution_title} with "
                f"{len(pro_arguments)} pro arguments, {len(counter_arguments)} counter arguments, and {len(metrics_list)} metrics"
            )
            return result

        except Exception as e:
            logger.error(f"Error in solution analysis for {solution_title}: {e}")
            raise SolutionAnalysisError(f"Failed to complete solution analysis: {e}") 