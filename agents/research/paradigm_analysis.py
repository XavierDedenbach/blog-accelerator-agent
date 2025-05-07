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

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage

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
        llm: BaseChatModel,
        source_validator: SourceValidator
    ):
        """
        Initialize the paradigm analyzer.
        
        Args:
            llm: Language model instance.
            source_validator: SourceValidator instance.
        """
        self.initial_llm = llm
        self.source_validator = source_validator
        self.min_paradigms = 3  # ADDED: Default minimum paradigms for identification
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompts for paradigm analysis."""
        self.identify_paradigms_prompt = PromptTemplate(
            input_variables=["topic", "min_paradigms"],
            template="""You are analyzing the historical paradigms related to the topic: {topic}.

            **Step 1: Identify Core Constraints**
            Begin by considering the fundamental technological, economic, social, or scientific constraints that have shaped the evolution of '{topic}' over time. What hard limitations or boundaries have defined different eras?

            **Step 2: Consider Systemic Context**
            Reflect on how '{topic}' fits within broader historical, social, and technological systems. How have larger contextual forces shaped its evolution?

            **Step 3: Map Stakeholder Perspectives**
            Consider the various stakeholders throughout history who have influenced or been affected by '{topic}'. How have their perspectives and needs shaped different paradigms?

            **Step 4: Identify Historical Paradigms**
            Based on your reflections, identify at least {min_paradigms} significant historical paradigms related to '{topic}'. Ensure they are:
            1. Distinct and well-defined
            2. Historically significant and well-documented
            3. Show clear progression or shifts related to the core constraints identified
            4. Relevant to understanding the current state

            For each paradigm:
            1. Provide a clear name
            2. Write a detailed description (2-3 sentences), linking it back to the core constraints where possible.
            3. Specify the approximate time period
            4. Explain key characteristics defining this paradigm, reflecting the context of earlier steps.

            **Step 5: Generate Supporting Evidence**
            For each paradigm, briefly note what types of historical evidence best document or demonstrate its existence and influence.

            **Step 6: Test Counter-Arguments**
            For each paradigm, briefly identify one potential critique or alternative interpretation of this historical framing, and how you would respond to it.

            Format your response as a JSON array of paradigm objects with these fields:
            - name: Name of the paradigm
            - description: Detailed description linking to constraints
            - time_period: Approximate time period
            - key_characteristics: Array of characteristics
            - supporting_evidence: Types of evidence that document this paradigm
            - potential_critique: A potential critique of this paradigm framing
            - critique_response: Brief response to the critique
            - search_terms: Array of nuanced search terms

            Only respond with the JSON array. Include at least {min_paradigms} paradigms in chronological order.
            """
        )
        
        self.analyze_transitions_prompt = PromptTemplate(
            input_variables=["paradigms"],
            template="""You are analyzing the transitions between historical paradigms, focusing on the underlying causes.

            Paradigms (in chronological order):
            {paradigms}

            **Step 1: Identify Core Constraints**
            Consider what fundamental constraints (technological, economic, social, etc.) were overcome or changed to enable each transition. What key limitations were pushed beyond?

            **Step 2: Consider Systemic Context**
            Reflect on the broader ecosystem factors that influenced these transitions. What external pressures or opportunities created conditions for change?

            **Step 3: Map Stakeholder Perspectives**
            Consider the various stakeholders who would have participated in or resisted these transitions. How did different perspectives influence the adoption of new paradigms?

            **Step 4: Identify Transition Dynamics**
            Based on your reflections, analyze the transition *between each consecutive pair* of paradigms listed above.
            For each transition:
            1. Identify the 'From' and 'To' paradigms.
            2. Describe the key factors and events that triggered or facilitated this shift, linking back to constraints and context.
            3. Explain the primary tensions or conflicts between the outgoing and incoming paradigms.
            4. Estimate the approximate timeframe of the transition period.

            **Step 5: Generate Supporting Evidence**
            For each transition, briefly identify key historical evidence (events, innovations, publications, etc.) that mark or document this transition period.

            **Step 6: Test Counter-Arguments**
            For each transition, briefly address one alternative explanation for why the shift occurred, and why your analysis provides a more comprehensive understanding.

            Format your response as a JSON array of transition objects with these fields:
            - from_paradigm: Name of the earlier paradigm
            - to_paradigm: Name of the later paradigm
            - trigger_factors: Description of key factors/events causing the shift
            - core_tensions: Explanation of conflicts between the paradigms
            - transition_period: Approximate timeframe (e.g., "Late 1980s")
            - key_evidence: Historical markers of this transition
            - alternative_explanation: An alternative view of why this transition occurred
            - explanation_defense: Why your analysis is more comprehensive

            Only respond with the JSON array.
            """
        )
        
        self.extract_lessons_prompt = PromptTemplate(
            input_variables=["paradigms", "transitions"],
            template="""You are extracting actionable lessons from historical paradigm shifts related to a topic.

            Historical Paradigms:
            {paradigms}

            Paradigm Transitions:
            {transitions}

            **Step 1: Identify Core Constraints**
            Consider what fundamental constraints (technological, economic, social, etc.) have consistently shaped the evolution of this field across multiple paradigm shifts.

            **Step 2: Consider Systemic Context**
            Reflect on how broader system dynamics have influenced successful or failed transitions. What patterns emerge in how ecosystems evolve?

            **Step 3: Map Stakeholder Perspectives**
            Consider how different stakeholder groups have repeatedly responded to paradigm shifts. Who typically drives change, who resists it, and why?

            **Step 4: Identify Key Lessons**
            Based on your reflections, extract 5-7 key lessons learned from these historical shifts that are relevant for understanding the topic today and navigating future changes.
            For each lesson:
            1. State the lesson clearly and concisely.
            2. Provide a brief explanation (2-3 sentences) grounding the lesson in specific examples from the provided paradigms/transitions.
            3. Suggest how this lesson might apply to current or future challenges/opportunities related to the topic.

            **Step 5: Generate Supporting Evidence**
            For each lesson, briefly identify specific historical examples that best demonstrate or validate this lesson.

            **Step 6: Test Counter-Arguments**
            For each lesson, briefly address one potential argument that might challenge the validity or applicability of this lesson, and how you would respond to strengthen your point.

            Format your response as a JSON array of lesson objects with these fields:
            - lesson: Clear statement of the lesson
            - explanation: Grounded explanation with historical examples
            - relevance_today: How it applies now or in the future
            - historical_examples: Specific examples supporting this lesson
            - potential_challenge: A potential challenge to this lesson
            - challenge_response: Your response to the challenge

            Only respond with the JSON array.
            """
        )
        
        self.project_future_paradigms_prompt = PromptTemplate(
            input_variables=["topic", "paradigms", "transitions", "lessons"],
            template="""You are projecting potential future paradigms related to the topic: {topic}, based on historical context and current trends.

            Historical Paradigms:
            {paradigms}

            Paradigm Transitions:
            {transitions}

            Historical Lessons:
            {lessons}

            **Step 1: Identify Core Constraints**
            Consider what fundamental constraints currently define the boundaries of '{topic}'. What technological, economic, or social limitations might be overcome next?

            **Step 2: Consider Systemic Context**
            Reflect on the broader ecosystem trends affecting '{topic}'. What external forces are creating pressure for change?

            **Step 3: Map Stakeholder Perspectives**
            Consider the various stakeholder groups with interests in '{topic}'. What emerging needs or expectations might drive paradigm evolution?

            **Step 4: Project Future Paradigms**
            Based on your reflections, project 2-4 plausible future paradigms for '{topic}'. These should represent distinct potential futures based on the constraints and forces identified.
            For each projected paradigm:
            1. Provide a speculative but descriptive name.
            2. Describe the core assumptions and characteristics of this potential future paradigm, linking it to the constraints and forces identified.
            3. Explain what key developments or events would need to occur for this paradigm to emerge.
            4. Discuss potential implications or consequences if this paradigm became dominant.

            **Step 5: Generate Supporting Evidence**
            For each projected paradigm, briefly identify early signals or emerging trends that provide initial evidence this paradigm might be developing.

            **Step 6: Test Counter-Arguments**
            For each projection, briefly address one potential criticism about why this paradigm might not emerge as expected, and provide a reasoned response.

            Format your response as a JSON array of future paradigm objects with these fields:
            - name: Speculative name of the future paradigm
            - description: Core assumptions/characteristics linked to constraints/forces
            - emergence_conditions: Key developments needed for it to arise
            - potential_implications: Consequences if dominant
            - early_signals: Current trends or signals indicating movement in this direction
            - potential_criticism: A potential criticism of this projection
            - criticism_response: Your reasoned response to the criticism
            - search_terms: Terms to explore emerging signals

            Only respond with the JSON array.
            """
        )
    
    async def identify_historical_paradigms(
        self,
        topic: str,
        llm_override: Optional[BaseChatModel] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify historical paradigms related to a topic, reflecting on core drivers first.
        
        Args:
            topic: Topic to analyze
            llm_override: Optional language model override
            
        Returns:
            List of paradigm dictionaries
        """
        logger.info(f"Identifying historical paradigms for topic: {topic} with core driver reflection...")
        try:
            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.identify_paradigms_prompt
            )
            response_text = await chain.arun(topic=topic, min_paradigms=3)
            paradigms = json.loads(response_text)
            logger.info(f"Identified {len(paradigms)} historical paradigms for topic: {topic}")
            return paradigms
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error identifying paradigms for {topic}: {e}")
            logger.error(f"LLM Response was: {response_text if 'response_text' in locals() else 'Response text not captured'}")
            raise ParadigmAnalysisError(f"Failed to parse paradigms JSON: {e}")
        except Exception as e:
            logger.error(f"Error identifying historical paradigms for {topic}: {e}")
            raise ParadigmAnalysisError(f"Failed to identify historical paradigms: {e}")
    
    async def analyze_paradigm_transitions(
        self,
        paradigms: List[Dict[str, Any]],
        llm_override: Optional[BaseChatModel] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze transitions between paradigms, reflecting on dynamics first.
        
        Args:
            paradigms: List of paradigm dictionaries
            llm_override: Optional language model override
            
        Returns:
            List of transition dictionaries
        """
        if len(paradigms) < 2:
            logger.info("Not enough paradigms to analyze transitions.")
            return []
        
        logger.info(f"Analyzing transitions between {len(paradigms)} paradigms with dynamics reflection...")
        try:
            paradigms_text = "\n".join([
                f"{i+1}. {p.get('name', '')}: {p.get('description', '')} (Drivers: {p.get('drivers', '')})"
                for i, p in enumerate(paradigms)
            ])
            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.analyze_transitions_prompt
            )
            response_text = await chain.arun(paradigms=paradigms_text)
            transitions = json.loads(response_text)
            logger.info(f"Analyzed {len(transitions)} paradigm transitions.")
            return transitions
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error analyzing transitions: {e}")
            logger.error(f"LLM Response was: {response_text if 'response_text' in locals() else 'Response text not captured'}")
            raise ParadigmAnalysisError(f"Failed to parse transitions JSON: {e}")
        except Exception as e:
            logger.error(f"Error analyzing paradigm transitions: {e}")
            raise ParadigmAnalysisError(f"Failed to analyze paradigm transitions: {e}")
    
    async def extract_historical_lessons(
        self,
        paradigms: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]],
        llm_override: Optional[BaseChatModel] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract lessons from historical shifts, reflecting on patterns first.
        
        Args:
            paradigms: List of paradigm dictionaries
            transitions: List of transition dictionaries
            llm_override: Optional language model override
            
        Returns:
            List of lesson dictionaries
        """
        logger.info(f"Extracting historical lessons with pattern reflection...")
        try:
            paradigms_text = "\n".join([
                f"{i+1}. {p.get('name', '')}: {p.get('description', '')}"
                for i, p in enumerate(paradigms)
            ])
            transitions_text = "\n".join([
                f"{i+1}. From {t.get('from_paradigm', '')} to {t.get('to_paradigm', '')}: {t.get('analysis', '')}"
                for i, t in enumerate(transitions)
            ])
            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.extract_lessons_prompt
            )
            response_text = await chain.arun(
                paradigms=paradigms_text,
                transitions=transitions_text,
                min_lessons=3
            )
            lessons = json.loads(response_text)
            logger.info(f"Extracted {len(lessons)} historical lessons.")
            return lessons
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error extracting lessons: {e}")
            logger.error(f"LLM Response was: {response_text if 'response_text' in locals() else 'Response text not captured'}")
            raise ParadigmAnalysisError(f"Failed to parse lessons JSON: {e}")
        except Exception as e:
            logger.error(f"Error extracting historical lessons: {e}")
            raise ParadigmAnalysisError(f"Failed to extract historical lessons: {e}")
    
    async def project_future_paradigms(
        self,
        topic: str,
        historical_paradigms: List[Dict[str, Any]],
        historical_transitions: List[Dict[str, Any]],
        lessons: List[Dict[str, Any]],
        llm_override: Optional[BaseChatModel] = None
    ) -> List[Dict[str, Any]]:
        """
        Project future paradigms, reflecting on current drivers first.
        
        Args:
            topic: Topic being analyzed
            historical_paradigms: List of historical paradigm dictionaries
            historical_transitions: List of historical transition dictionaries
            lessons: List of lesson dictionaries
            llm_override: Optional language model override
            
        Returns:
            List of future paradigm projection dictionaries
        """
        logger.info(f"Projecting future paradigms for {topic} with driver/disruptor reflection...")
        try:
            paradigms_text = "\n".join([
                f"{i+1}. {p.get('name', '')}: {p.get('description', '')}"
                for i, p in enumerate(historical_paradigms)
            ])
            lessons_text = "\n".join([f"- {l['lesson']}" for l in lessons])
            
            transitions_text = "\n".join([
                f"{i+1}. From {t.get('from_paradigm', 'N/A')} to {t.get('to_paradigm', 'N/A')}: "
                f"Factors: {t.get('trigger_factors', 'N/A')}, Tensions: {t.get('core_tensions', 'N/A')}"
                for i, t in enumerate(historical_transitions)
            ]) if historical_transitions else "No historical transitions were analyzed or provided for this projection."

            current_llm = llm_override or self.initial_llm
            chain = LLMChain(
                llm=current_llm,
                prompt=self.project_future_paradigms_prompt
            )
            response_text = await chain.arun(
                topic=topic,
                paradigms=paradigms_text,
                transitions=transitions_text,
                lessons=lessons_text,
            )
            future_paradigms = json.loads(response_text)
            logger.info(f"Projected {len(future_paradigms)} future paradigms for {topic}.")
            return future_paradigms
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parsing Error projecting future paradigms: {e}")
            logger.error(f"LLM Response was: {response_text if 'response_text' in locals() else 'Response text not captured'}")
            raise ParadigmAnalysisError(f"Failed to parse future paradigms JSON: {e}")
        except Exception as e:
            logger.error(f"Error projecting future paradigms: {e}")
            raise ParadigmAnalysisError(f"Failed to project future paradigms: {e}")
    
    async def find_sources_for_paradigm(
        self, 
        paradigm: Dict[str, Any],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find authoritative sources for a paradigm.
        
        Args:
            paradigm: Paradigm dictionary 
            count: Number of sources to find
            
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
        
        logger.info(f"Found {len(unique_sources)} sources for paradigm: {paradigm.get('name', '')}")
        return unique_sources
    
    async def analyze_paradigms(
        self,
        topic: str,
        llm_override: Optional[BaseChatModel] = None
    ) -> Dict[str, Any]:
        """
        Perform complete paradigm analysis for a topic.
        
        Args:
            topic: Topic to analyze
            llm_override: Optional language model override
            
        Returns:
            Dictionary with paradigm analysis results
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Identify historical paradigms
            logger.info(f"Identifying historical paradigms for topic: {topic}")
            paradigms = await self.identify_historical_paradigms(topic, llm_override=llm_override)
            
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
            transitions = await self.analyze_paradigm_transitions(paradigms, llm_override=llm_override)
            
            # Extract lessons from historical shifts
            logger.info(f"Extracting lessons from historical paradigm shifts")
            lessons = await self.extract_historical_lessons(paradigms, transitions, llm_override=llm_override)
            
            # Project future paradigms
            logger.info(f"Projecting future paradigm possibilities for {topic}")
            future_paradigms = await self.project_future_paradigms(topic, enriched_paradigms, transitions, lessons, llm_override=llm_override)
            
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