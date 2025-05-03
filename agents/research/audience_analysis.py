"""
Audience Analysis component for the Blog Accelerator Agent.

This module:
1. Identifies target audience segments for a topic
2. Analyzes audience needs, pain points, and motivations
3. Evaluates existing knowledge and expertise levels
4. Recommends content strategies based on audience characteristics
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


class AudienceAnalysisError(Exception):
    """Exception raised for errors in audience analysis."""
    pass


class AudienceAnalyzer:
    """
    Analyzer for target audience segments related to a topic.
    
    Features:
    - Identification of distinct audience segments
    - Analysis of audience needs and pain points
    - Evaluation of audience knowledge levels
    - Recommendation of content strategies
    """
    
    # Default target audience segments for the Blog Accelerator Agent
    DEFAULT_AUDIENCE_SEGMENTS = [
        {
            "name": "STEM Students",
            "description": "College students in later years of STEM programs, seeking practical applications of their theoretical knowledge.",
            "motivations": ["Connecting theory to real-world applications", "Exploring career paths", "Staying current with industry trends"],
            "pain_points": ["Too much theory, not enough practice", "Difficulty seeing real-world relevance", "Information overload"],
            "knowledge_level": "intermediate"
        },
        {
            "name": "Technical Professionals",
            "description": "Working professionals across various technical disciplines, looking to stay current with innovations and trends.",
            "motivations": ["Continuous learning", "Career advancement", "Problem-solving improvements"],
            "pain_points": ["Limited time for research", "Rapid technological change", "Difficulty evaluating competing solutions"],
            "knowledge_level": "advanced"
        },
        {
            "name": "Engineers",
            "description": "Engineering practitioners who need to understand both technical details and systemic impacts of new technologies.",
            "motivations": ["Finding efficient solutions", "Technical depth", "Cross-disciplinary applications"],
            "pain_points": ["Balancing depth vs. breadth", "Connecting specialized knowledge to broader context", "Evaluating trade-offs"],
            "knowledge_level": "advanced"
        },
        {
            "name": "Founders",
            "description": "Startup founders and entrepreneurs looking for insights on technological trends to inform business decisions.",
            "motivations": ["Identifying opportunities", "Understanding competitive landscape", "Risk assessment"],
            "pain_points": ["Information asymmetry", "Technical due diligence challenges", "Resource constraints"],
            "knowledge_level": "intermediate"
        },
        {
            "name": "Business Professionals",
            "description": "Business leaders and managers who need to understand technical topics for strategic decision-making.",
            "motivations": ["Strategic planning", "Investment decisions", "Cross-functional communication"],
            "pain_points": ["Technical knowledge gaps", "Translating technical concepts to business impact", "Keeping pace with innovation"],
            "knowledge_level": "beginner"
        }
    ]
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        source_validator: Optional[SourceValidator] = None,
        min_segments: int = 3,
        use_default_segments: bool = False
    ):
        """
        Initialize the audience analyzer.
        
        Args:
            openai_api_key: OpenAI API key
            groq_api_key: Groq API key
            source_validator: SourceValidator instance
            min_segments: Minimum number of audience segments to identify
            use_default_segments: Whether to use default audience segments
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
            raise AudienceAnalysisError("No API key provided for LLM")
        
        # Initialize source validator if not provided
        self.source_validator = source_validator or SourceValidator()
        
        # Set minimum segments threshold
        self.min_segments = min_segments
        
        # Set whether to use default segments
        self.use_default_segments = use_default_segments
        
        # Load prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompts for audience analysis."""
        self.identify_segments_prompt = PromptTemplate(
            input_variables=["topic", "min_segments"],
            template="""You are analyzing the target audience for content about: {topic}.
            
Your task is to identify at least {min_segments} distinct audience segments who would be interested in this topic. 
Focus on segments that are:
1. Specific and well-defined
2. Likely to have genuine interest in the topic
3. Diverse in their backgrounds, needs, and motivations
4. Relevant to the topic's domain
5. Actionable for content creation purposes

For each audience segment:
1. Provide a clear name/title for the segment
2. Write a detailed description (2-3 sentences)
3. Describe their primary motivations for engaging with this topic
4. Identify their key pain points related to this topic
5. Estimate their existing knowledge/expertise level (beginner, intermediate, advanced, expert)
6. Suggest search terms to find data about this audience segment

Format your response as a JSON array of segment objects with these fields:
- name: Name of the audience segment
- description: Detailed description
- motivations: Array of primary motivations
- pain_points: Array of key pain points
- knowledge_level: One of: "beginner", "intermediate", "advanced", or "expert"
- search_terms: Array of search terms to find data about this segment

Only respond with the JSON array. Include at least {min_segments} distinct audience segments.

Important: Consider how this topic might specifically appeal to: STEM students in college, technical professionals, engineers, founders, and business people.
"""
        )
        
        self.analyze_needs_prompt = PromptTemplate(
            input_variables=["segment", "topic"],
            template="""You are deeply analyzing the needs and goals of a specific audience segment interested in: {topic}.

Audience Segment: {segment}

Your task is to analyze this segment's needs, goals, and preferences in depth. Focus on:
1. What specific problems they need solved related to this topic
2. What goals they want to achieve by learning about this topic
3. What content formats and styles would appeal to this segment
4. What level of technical detail is appropriate for this segment
5. What would make content about this topic valuable to them

Format your response as a JSON object with these fields:
- problems_to_solve: Array of specific problems this segment needs solved
- goals: Array of goals they want to achieve
- preferred_formats: Array of content formats that would appeal to them (e.g., tutorials, case studies)
- technical_detail: One of: "minimal", "moderate", "substantial", or "comprehensive"
- value_creators: Array of elements that would make content valuable to this segment

Only respond with the JSON object.
"""
        )
        
        self.evaluate_knowledge_prompt = PromptTemplate(
            input_variables=["segment", "topic"],
            template="""You are evaluating the existing knowledge and expertise level of a specific audience segment interested in: {topic}.

Audience Segment: {segment}

Your task is to evaluate what this segment likely already knows about the topic, and where their knowledge gaps are. Focus on:
1. Fundamental concepts they likely already understand
2. Technical terminology they are probably familiar with
3. Specific knowledge gaps they likely have
4. Common misconceptions they might hold
5. Areas where they would benefit from deeper explanation

Format your response as a JSON object with these fields:
- existing_knowledge: Array of concepts they likely already understand
- familiar_terminology: Array of technical terms they probably know
- knowledge_gaps: Array of areas where they likely lack knowledge
- misconceptions: Array of common misconceptions they might hold
- areas_for_explanation: Array of topics that would benefit from deeper explanation

Only respond with the JSON object.
"""
        )
        
        self.recommend_strategies_prompt = PromptTemplate(
            input_variables=["segment", "topic", "needs", "knowledge"],
            template="""You are recommending content strategies for a specific audience segment interested in: {topic}.

Audience Segment: {segment}

Their needs and preferences:
{needs}

Their knowledge level:
{knowledge}

Your task is to recommend specific content strategies that would effectively engage this audience segment. Focus on:
1. Key topics to prioritize for this segment
2. Effective framing and positioning approaches
3. Content formats and structures that would work well
4. Specific examples, stories, or case studies that would resonate
5. Tone and style recommendations

Format your response as a JSON object with these fields:
- priority_topics: Array of key topics to prioritize
- framing_approaches: Array of effective framing/positioning approaches
- recommended_formats: Array of content formats and structures
- examples_to_include: Array of specific examples or case studies to consider
- tone_recommendations: Array of tone and style recommendations

Only respond with the JSON object.
"""
        )
    
    async def identify_audience_segments(self, topic: str) -> List[Dict[str, Any]]:
        """
        Identify distinct audience segments for a topic.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            List of audience segment dictionaries
        """
        # If using default segments, return them with search terms for the topic
        if self.use_default_segments:
            segments = self.DEFAULT_AUDIENCE_SEGMENTS.copy()
            # Add search terms for each segment based on the topic
            for segment in segments:
                segment["search_terms"] = [
                    f"{segment['name']} {topic}",
                    f"{topic} for {segment['name']}",
                    f"{segment['name']} needs {topic}"
                ]
            logger.info(f"Using default audience segments for topic: {topic}")
            return segments
            
        try:
            # Create chain for segment identification
            chain = LLMChain(
                llm=self.llm,
                prompt=self.identify_segments_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                topic=topic,
                min_segments=self.min_segments
            )
            
            # Parse JSON response
            segments = json.loads(response)
            
            logger.info(f"Identified {len(segments)} audience segments for topic: {topic}")
            return segments
            
        except Exception as e:
            logger.error(f"Error identifying audience segments for {topic}: {e}")
            raise AudienceAnalysisError(f"Failed to identify audience segments: {e}")
    
    async def analyze_segment_needs(
        self, 
        segment: Dict[str, Any],
        topic: str
    ) -> Dict[str, Any]:
        """
        Analyze the needs and goals of an audience segment.
        
        Args:
            segment: Audience segment dictionary
            topic: Main topic
            
        Returns:
            Dictionary with needs analysis
        """
        try:
            # Format segment for prompt
            segment_text = f"{segment.get('name', 'Unknown Segment')}: {segment.get('description', 'No description')}\n"
            segment_text += f"Motivations: {', '.join(segment.get('motivations', []))}\n"
            segment_text += f"Pain points: {', '.join(segment.get('pain_points', []))}\n"
            segment_text += f"Knowledge level: {segment.get('knowledge_level', 'Unknown')}"
            
            # Create chain for needs analysis
            chain = LLMChain(
                llm=self.llm,
                prompt=self.analyze_needs_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                segment=segment_text,
                topic=topic
            )
            
            # Parse JSON response
            needs = json.loads(response)
            
            logger.info(f"Analyzed needs for segment: {segment.get('name', '')}")
            return needs
            
        except Exception as e:
            logger.error(f"Error analyzing needs for segment {segment.get('name', '')}: {e}")
            raise AudienceAnalysisError(f"Failed to analyze segment needs: {e}")
    
    async def evaluate_segment_knowledge(
        self, 
        segment: Dict[str, Any],
        topic: str
    ) -> Dict[str, Any]:
        """
        Evaluate the existing knowledge of an audience segment.
        
        Args:
            segment: Audience segment dictionary
            topic: Main topic
            
        Returns:
            Dictionary with knowledge evaluation
        """
        try:
            # Format segment for prompt
            segment_text = f"{segment.get('name', 'Unknown Segment')}: {segment.get('description', 'No description')}\n"
            segment_text += f"Knowledge level: {segment.get('knowledge_level', 'Unknown')}"
            
            # Create chain for knowledge evaluation
            chain = LLMChain(
                llm=self.llm,
                prompt=self.evaluate_knowledge_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                segment=segment_text,
                topic=topic
            )
            
            # Parse JSON response
            knowledge = json.loads(response)
            
            logger.info(f"Evaluated knowledge for segment: {segment.get('name', '')}")
            return knowledge
            
        except Exception as e:
            logger.error(f"Error evaluating knowledge for segment {segment.get('name', '')}: {e}")
            raise AudienceAnalysisError(f"Failed to evaluate segment knowledge: {e}")
    
    async def recommend_content_strategies(
        self,
        segment: Dict[str, Any],
        topic: str,
        needs: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recommend content strategies for an audience segment.
        
        Args:
            segment: Audience segment dictionary
            topic: Main topic
            needs: Needs analysis dictionary
            knowledge: Knowledge evaluation dictionary
            
        Returns:
            Dictionary with content strategy recommendations
        """
        try:
            # Format segment for prompt
            segment_text = f"{segment.get('name', 'Unknown Segment')}: {segment.get('description', 'No description')}\n"
            segment_text += f"Knowledge level: {segment.get('knowledge_level', 'Unknown')}"
            
            # Format needs for prompt
            needs_text = f"Problems to solve: {', '.join(needs.get('problems_to_solve', []))}\n"
            needs_text += f"Goals: {', '.join(needs.get('goals', []))}\n"
            needs_text += f"Preferred formats: {', '.join(needs.get('preferred_formats', []))}\n"
            needs_text += f"Technical detail: {needs.get('technical_detail', 'Unknown')}"
            
            # Format knowledge for prompt
            knowledge_text = f"Existing knowledge: {', '.join(knowledge.get('existing_knowledge', []))}\n"
            knowledge_text += f"Knowledge gaps: {', '.join(knowledge.get('knowledge_gaps', []))}\n"
            knowledge_text += f"Misconceptions: {', '.join(knowledge.get('misconceptions', []))}"
            
            # Create chain for strategy recommendations
            chain = LLMChain(
                llm=self.llm,
                prompt=self.recommend_strategies_prompt
            )
            
            # Run the chain
            response = await chain.arun(
                segment=segment_text,
                topic=topic,
                needs=needs_text,
                knowledge=knowledge_text
            )
            
            # Parse JSON response
            strategies = json.loads(response)
            
            logger.info(f"Recommended content strategies for segment: {segment.get('name', '')}")
            return strategies
            
        except Exception as e:
            logger.error(f"Error recommending strategies for segment {segment.get('name', '')}: {e}")
            raise AudienceAnalysisError(f"Failed to recommend content strategies: {e}")
    
    async def find_sources_for_segment(
        self, 
        segment: Dict[str, Any],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find supporting sources for an audience segment.
        
        Args:
            segment: Audience segment dictionary with search_terms
            count: Number of sources to find per search term
            
        Returns:
            List of source dictionaries
        """
        all_sources = []
        search_terms = segment.get("search_terms", [])
        
        if not search_terms:
            # Fallback if no search terms provided
            search_terms = [segment.get("name", "")]
        
        # Use up to 3 search terms
        for term in search_terms[:3]:
            try:
                # Search for sources
                query = f"{term} audience data {segment.get('name', '')}"
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
        
        logger.info(f"Found {len(unique_sources)} sources for audience segment: {segment.get('name', '')}")
        return unique_sources
    
    async def analyze_audience(self, topic: str) -> Dict[str, Any]:
        """
        Perform complete audience analysis for a topic.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Dictionary with audience analysis results
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Identify audience segments
            logger.info(f"Identifying audience segments for topic: {topic}")
            segments = await self.identify_audience_segments(topic)
            
            # Ensure we have at least min_segments
            if len(segments) < self.min_segments:
                logger.warning(
                    f"Only identified {len(segments)} audience segments, " 
                    f"which is less than minimum {self.min_segments}"
                )
            
            # Process each segment
            enriched_segments = []
            for segment in segments:
                # Find sources
                logger.info(f"Finding sources for segment: {segment.get('name', '')}")
                sources = await self.find_sources_for_segment(segment)
                
                # Analyze needs
                logger.info(f"Analyzing needs for segment: {segment.get('name', '')}")
                needs = await self.analyze_segment_needs(segment, topic)
                
                # Evaluate knowledge
                logger.info(f"Evaluating knowledge for segment: {segment.get('name', '')}")
                knowledge = await self.evaluate_segment_knowledge(segment, topic)
                
                # Recommend strategies
                logger.info(f"Recommending strategies for segment: {segment.get('name', '')}")
                strategies = await self.recommend_content_strategies(segment, topic, needs, knowledge)
                
                # Add to enriched segments
                enriched_segment = {
                    **segment,
                    "sources": sources,
                    "needs_analysis": needs,
                    "knowledge_evaluation": knowledge,
                    "content_strategies": strategies
                }
                enriched_segments.append(enriched_segment)
            
            # Calculate statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = {
                "topic": topic,
                "audience_segments": enriched_segments,
                "stats": {
                    "segments_count": len(enriched_segments),
                    "sources_count": sum(len(s.get("sources", [])) for s in enriched_segments),
                    "analysis_duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(
                f"Completed audience analysis for {topic} with "
                f"{len(enriched_segments)} audience segments"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in audience analysis for {topic}: {e}")
            raise AudienceAnalysisError(f"Failed to complete audience analysis: {e}") 