"""
Paradigm Analysis component for the Blog Accelerator Agent.

This module:
1. Analyzes historical paradigms related to a topic
2. Identifies key transitions between paradigms
3. Extracts lessons and patterns from historical examples
4. Projects future paradigm possibilities
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


class ParadigmAnalysisError(Exception):
    """Exception raised for errors in paradigm analysis."""
    pass


class ParadigmAnalyzer:
    """
    Analyzer for historical paradigms related to a topic.
    
    Features:
    - Identification of key historical paradigms
    - Analysis of transitions between paradigms
    - Extraction of lessons from historical examples
    - Projection of future paradigm possibilities
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        source_validator: Optional[SourceValidator] = None,
        min_paradigms: int = 3
    ):
        """
        Initialize the paradigm analyzer.
        
        Args:
            openai_api_key: OpenAI API key
            groq_api_key: Groq API key
            source_validator: SourceValidator instance
            min_paradigms: Minimum number of paradigms to identify
        """
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        # Try to use OpenAI first, fall back to Groq
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.2,
                openai_api_key=self.openai_api_key
            )
        elif self.groq_api_key:
            self.llm = ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.2,
                groq_api_key=self.groq_api_key
            )
        else:
            raise ParadigmAnalysisError("No API key provided for LLM")
        
        # Initialize source validator if not provided
        self.source_validator = source_validator or SourceValidator()
        
        # Set minimum paradigms threshold
        self.min_paradigms = min_paradigms
        
        # Load prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompts for paradigm analysis."""
        self.identify_paradigms_prompt = PromptTemplate(
            input_variables=["topic", "min_paradigms"],
            template="""You are analyzing the historical paradigms related to the topic: {topic}.
            
Your task is to identify at least {min_paradigms} significant historical paradigms related to this topic. 
Focus on paradigms that are:
1. Distinct approaches or worldviews
2. Historically significant
3. Well-documented with authoritative sources
4. Showing clear progression or evolution over time
5. Relevant to understanding the current and future state of the topic

For each paradigm:
1. Provide a clear name
2. Write a detailed description (2-3 sentences)
3. Specify the approximate time period when this paradigm was dominant
4. Explain key characteristics that defined this paradigm
5. Suggest search terms to find authoritative sources about this paradigm

Format your response as a JSON array of paradigm objects with these fields:
- name: Name of the paradigm
- description: Detailed description
- time_period: Approximate time period (e.g., "1950s-1970s", "Late 19th century")
- key_characteristics: Array of key characteristics
- search_terms: Array of search terms to find sources

Only respond with the JSON array. Include at least {min_paradigms} paradigms in chronological order.
"""
        )
        
        self.analyze_transitions_prompt = PromptTemplate(
            input_variables=["paradigms"],
            template="""You are analyzing the transitions between historical paradigms.

Below are the paradigms in chronological order:
{paradigms}

Your task is to analyze the transitions between these paradigms. For each transition:
1. Identify what factors led to the shift from one paradigm to the next
2. Explain key innovations or discoveries that enabled the transition
3. Identify any resistance to change and how it was overcome
4. Note any parallel or competing paradigms during the transition period

Format your response as a JSON array of transition objects with these fields:
- from_paradigm: Name of the earlier paradigm
- to_paradigm: Name of the later paradigm
- factors: Array of factors that led to the shift
- key_innovations: Array of innovations or discoveries that enabled the transition
- resistance: Description of resistance to change and how it was overcome
- competing_paradigms: Array of any competing paradigms during the transition

Only respond with the JSON array. Include an analysis for each transition between consecutive paradigms.
"""
        )
        
        self.extract_lessons_prompt = PromptTemplate(
            input_variables=["paradigms", "transitions"],
            template="""You are extracting lessons and patterns from historical paradigm shifts.

Below are the paradigms in chronological order:
{paradigms}

And here are the transitions between them:
{transitions}

Your task is to extract key lessons and patterns from these historical examples that might be relevant to understanding future paradigm shifts. Focus on:
1. Common patterns in how paradigms evolve
2. Factors that consistently lead to paradigm shifts
3. Indicators that might predict when a paradigm is nearing its end
4. Lessons for innovators trying to introduce new paradigms

Format your response as a JSON object with these fields:
- evolution_patterns: Array of patterns in how paradigms evolve
- shift_factors: Array of factors that consistently lead to paradigm shifts
- end_indicators: Array of indicators that might predict when a paradigm is nearing its end
- innovation_lessons: Array of lessons for innovators

Only respond with the JSON object.
"""
        )
        
        self.project_future_prompt = PromptTemplate(
            input_variables=["topic", "paradigms", "transitions", "lessons"],
            template="""You are projecting future paradigm possibilities for: {topic}.

Based on the historical paradigms:
{paradigms}

The transitions between them:
{transitions}

And the lessons extracted:
{lessons}

Your task is to project possible future paradigms that might emerge next. For each future paradigm possibility:
1. Provide a clear name
2. Write a detailed description of what this paradigm might entail
3. Explain the conditions that would need to be true for this paradigm to emerge
4. Identify early indicators that might suggest this paradigm is emerging
5. Estimate a rough timeline for when this paradigm might become dominant

Format your response as a JSON array of future paradigm objects with these fields:
- name: Name of the potential future paradigm
- description: Detailed description
- emergence_conditions: Conditions that would need to be true
- early_indicators: Early indicators that might suggest emergence
- estimated_timeline: Rough timeline (e.g., "Next 5-10 years", "2030s-2040s")

Only respond with the JSON array. Include at least 3 different possible future paradigms.
"""
        )
    
    async def identify_historical_paradigms(self, topic: str) -> List[Dict[str, Any]]:
        """
        Identify historical paradigms related to a topic.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            List of paradigm dictionaries
        """
        try:
            # Create chain for paradigm identification
            chain = LLMChain(
                llm=self.llm,
                prompt=self.identify_paradigms_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                topic=topic,
                min_paradigms=self.min_paradigms
            )
            
            # Parse JSON response
            paradigms = json.loads(response)
            
            logger.info(f"Identified {len(paradigms)} historical paradigms for topic: {topic}")
            return paradigms
            
        except Exception as e:
            logger.error(f"Error identifying paradigms for {topic}: {e}")
            raise ParadigmAnalysisError(f"Failed to identify paradigms: {e}")
    
    async def analyze_paradigm_transitions(
        self, 
        paradigms: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze transitions between paradigms.
        
        Args:
            paradigms: List of paradigm dictionaries in chronological order
            
        Returns:
            List of transition dictionaries
        """
        try:
            # Format paradigms for prompt
            paradigms_text = ""
            for i, paradigm in enumerate(paradigms, 1):
                paradigms_text += f"{i}. {paradigm.get('name', 'Unknown Paradigm')} ({paradigm.get('time_period', 'Unknown Period')}): "
                paradigms_text += f"{paradigm.get('description', 'No description')}\n"
                paradigms_text += f"   Key characteristics: {', '.join(paradigm.get('key_characteristics', []))}\n\n"
            
            # Create chain for transition analysis
            chain = LLMChain(
                llm=self.llm,
                prompt=self.analyze_transitions_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                paradigms=paradigms_text
            )
            
            # Parse JSON response
            transitions = json.loads(response)
            
            logger.info(f"Analyzed {len(transitions)} paradigm transitions")
            return transitions
            
        except Exception as e:
            logger.error(f"Error analyzing paradigm transitions: {e}")
            raise ParadigmAnalysisError(f"Failed to analyze paradigm transitions: {e}")
    
    async def extract_historical_lessons(
        self,
        paradigms: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract lessons and patterns from historical paradigm shifts.
        
        Args:
            paradigms: List of paradigm dictionaries
            transitions: List of transition dictionaries
            
        Returns:
            Dictionary with extracted lessons and patterns
        """
        try:
            # Format paradigms for prompt
            paradigms_text = ""
            for i, paradigm in enumerate(paradigms, 1):
                paradigms_text += f"{i}. {paradigm.get('name', 'Unknown Paradigm')} ({paradigm.get('time_period', 'Unknown Period')}): "
                paradigms_text += f"{paradigm.get('description', 'No description')}\n"
            
            # Format transitions for prompt
            transitions_text = ""
            for i, transition in enumerate(transitions, 1):
                transitions_text += f"{i}. From {transition.get('from_paradigm', 'Unknown')} to {transition.get('to_paradigm', 'Unknown')}:\n"
                transitions_text += f"   Factors: {', '.join(transition.get('factors', []))}\n"
                transitions_text += f"   Key innovations: {', '.join(transition.get('key_innovations', []))}\n"
                transitions_text += f"   Resistance: {transition.get('resistance', 'Unknown')}\n\n"
            
            # Create chain for lesson extraction
            chain = LLMChain(
                llm=self.llm,
                prompt=self.extract_lessons_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                paradigms=paradigms_text,
                transitions=transitions_text
            )
            
            # Parse JSON response
            lessons = json.loads(response)
            
            logger.info(f"Extracted lessons from historical paradigm shifts")
            return lessons
            
        except Exception as e:
            logger.error(f"Error extracting historical lessons: {e}")
            raise ParadigmAnalysisError(f"Failed to extract historical lessons: {e}")
    
    async def project_future_paradigms(
        self,
        topic: str,
        paradigms: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]],
        lessons: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Project possible future paradigms.
        
        Args:
            topic: Topic to analyze
            paradigms: List of paradigm dictionaries
            transitions: List of transition dictionaries
            lessons: Dictionary with extracted lessons
            
        Returns:
            List of future paradigm dictionaries
        """
        try:
            # Format paradigms for prompt
            paradigms_text = ""
            for i, paradigm in enumerate(paradigms, 1):
                paradigms_text += f"{i}. {paradigm.get('name', 'Unknown Paradigm')} ({paradigm.get('time_period', 'Unknown Period')}): "
                paradigms_text += f"{paradigm.get('description', 'No description')}\n"
            
            # Format transitions for prompt
            transitions_text = ""
            for i, transition in enumerate(transitions, 1):
                transitions_text += f"{i}. From {transition.get('from_paradigm', 'Unknown')} to {transition.get('to_paradigm', 'Unknown')}:\n"
                transitions_text += f"   Factors: {', '.join(transition.get('factors', []))}\n"
                transitions_text += f"   Key innovations: {', '.join(transition.get('key_innovations', []))}\n"
            
            # Format lessons for prompt
            lessons_text = ""
            lessons_text += f"Evolution patterns: {', '.join(lessons.get('evolution_patterns', []))}\n"
            lessons_text += f"Shift factors: {', '.join(lessons.get('shift_factors', []))}\n"
            lessons_text += f"End indicators: {', '.join(lessons.get('end_indicators', []))}\n"
            
            # Create chain for future projection
            chain = LLMChain(
                llm=self.llm,
                prompt=self.project_future_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                topic=topic,
                paradigms=paradigms_text,
                transitions=transitions_text,
                lessons=lessons_text
            )
            
            # Parse JSON response
            future_paradigms = json.loads(response)
            
            logger.info(f"Projected {len(future_paradigms)} possible future paradigms for {topic}")
            return future_paradigms
            
        except Exception as e:
            logger.error(f"Error projecting future paradigms for {topic}: {e}")
            raise ParadigmAnalysisError(f"Failed to project future paradigms: {e}")
    
    async def find_sources_for_paradigm(
        self, 
        paradigm: Dict[str, Any],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find authoritative sources for a paradigm.
        
        Args:
            paradigm: Paradigm dictionary with search_terms
            count: Number of sources to find per search term
            
        Returns:
            List of source dictionaries
        """
        all_sources = []
        search_terms = paradigm.get("search_terms", [])
        
        if not search_terms:
            # Fallback if no search terms provided
            search_terms = [paradigm.get("name", "")]
        
        # Use up to 3 search terms
        for term in search_terms[:3]:
            try:
                # Search for sources
                query = f"{term} {paradigm.get('name', '')} history"
                supporting, _ = self.source_validator.find_supporting_contradicting_sources(
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
        
        logger.info(f"Found {len(unique_sources)} sources for paradigm: {paradigm.get('name', '')}")
        return unique_sources
    
    async def analyze_paradigms(self, topic: str) -> Dict[str, Any]:
        """
        Perform complete paradigm analysis for a topic.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Dictionary with paradigm analysis results
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Identify historical paradigms
            logger.info(f"Identifying historical paradigms for topic: {topic}")
            paradigms = await self.identify_historical_paradigms(topic)
            
            # Ensure we have at least min_paradigms
            if len(paradigms) < self.min_paradigms:
                logger.warning(
                    f"Only identified {len(paradigms)} paradigms, " 
                    f"which is less than minimum {self.min_paradigms}"
                )
            
            # Find sources for each paradigm
            enriched_paradigms = []
            for paradigm in paradigms:
                # Find sources
                logger.info(f"Finding sources for paradigm: {paradigm.get('name', '')}")
                sources = await self.find_sources_for_paradigm(paradigm)
                
                # Add to enriched paradigms
                enriched_paradigm = {
                    **paradigm,
                    "sources": sources
                }
                enriched_paradigms.append(enriched_paradigm)
            
            # Analyze transitions between paradigms
            logger.info(f"Analyzing transitions between paradigms")
            transitions = await self.analyze_paradigm_transitions(paradigms)
            
            # Extract lessons from historical shifts
            logger.info(f"Extracting lessons from historical paradigm shifts")
            lessons = await self.extract_historical_lessons(paradigms, transitions)
            
            # Project future paradigms
            logger.info(f"Projecting future paradigm possibilities for {topic}")
            future_paradigms = await self.project_future_paradigms(topic, paradigms, transitions, lessons)
            
            # Calculate statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = {
                "topic": topic,
                "historical_paradigms": enriched_paradigms,
                "transitions": transitions,
                "lessons": lessons,
                "future_paradigms": future_paradigms,
                "stats": {
                    "paradigms_count": len(enriched_paradigms),
                    "transitions_count": len(transitions),
                    "future_projections_count": len(future_paradigms),
                    "sources_count": sum(len(p.get("sources", [])) for p in enriched_paradigms),
                    "analysis_duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(
                f"Completed paradigm analysis for {topic} with "
                f"{len(enriched_paradigms)} historical paradigms and "
                f"{len(future_paradigms)} future projections"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in paradigm analysis for {topic}: {e}")
            raise ParadigmAnalysisError(f"Failed to complete paradigm analysis: {e}") 