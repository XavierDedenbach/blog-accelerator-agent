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
            input_variables=["topic", "min_segments", "existing_segments"],
            template="""You are identifying distinct target audience segments for content related to the topic: {topic}. Consider the default segments provided, but refine or add based on the specific topic.

            Default Segments (for context):
            {existing_segments}

            Topic: {topic}

            **Step 1: Reflect on Topic Relevance & Potential Audiences**
            Before defining segments, briefly consider who would be most interested in or impacted by '{topic}'. Think about different roles, industries, levels of expertise, motivations (e.g., learning, problem-solving, strategic insight), and potential pain points related to this specific topic. How does the topic intersect with the default segments? Are there gaps?

            **Step 2: Define Target Audience Segments**
            Based on your reflection and the topic '{topic}', define at least {min_segments} distinct and relevant audience segments. You can adapt the default segments or create new ones specific to the topic. Ensure segments are:
            1. Clearly distinct from each other.
            2. Directly relevant to the specific topic '{topic}'.
            3. Described with sufficient detail to understand their perspective.

            For each segment:
            1. Provide a clear name (e.g., "Hardware Engineers implementing {topic}", "VCs evaluating {topic} startups").
            2. Write a detailed description (2-3 sentences) explaining who they are and their relationship to the topic.
            3. List their likely motivations for engaging with content on this topic.
            4. List potential pain points or challenges they face related to this topic.
            5. Estimate their general knowledge level regarding this specific topic (e.g., beginner, intermediate, advanced, expert).
            6. Suggest search terms to find demographic data or discussions involving this audience segment related to the topic.

            Format your response as a JSON array of segment objects with these fields:
            - name: Specific name of the segment
            - description: Detailed description related to the topic
            - motivations: Array of topic-specific motivations
            - pain_points: Array of topic-specific pain points
            - knowledge_level: Estimated topic-specific knowledge level
            - search_terms: Array of nuanced search terms

            Only respond with the JSON array. Include at least {min_segments} segments.
            """
        )
        
        self.analyze_needs_prompt = PromptTemplate(
            input_variables=["segment_name", "segment_description", "topic"],
            template="""You are analyzing the specific needs and pain points of a target audience segment regarding a specific topic.

            Audience Segment: {segment_name}
            Description: {segment_description}
            Topic: {topic}

            **Step 1: Reflect on Segment's Perspective**
            Consider the '{segment_name}' based on their description. From their perspective, what are the most critical aspects of '{topic}'? What questions would they likely have? What problems related to '{topic}' are they trying to solve? What are their biggest frustrations or challenges when trying to understand or apply '{topic}'?

            **Step 2: Detail Needs and Pain Points**
            Based on your reflection, provide a detailed analysis of this segment's needs and pain points specifically related to '{topic}'.
            Structure the analysis into:
            - Information Needs: What specific information or knowledge are they lacking or seeking about '{topic}'? (List 3-5 specific needs)
            - Key Questions: What are the top 3-5 questions this segment likely has about '{topic}'?
            - Core Pain Points: What are the primary frustrations or difficulties they experience concerning '{topic}'? (List 3-5 specific pain points)
            - Desired Outcomes: What do they hope to achieve by learning about or engaging with '{topic}'?

            Format your response as a JSON object with these keys:
            - information_needs: Array of specific information needs (strings)
            - key_questions: Array of likely questions (strings)
            - core_pain_points: Array of specific pain points (strings)
            - desired_outcomes: Array of desired outcomes (strings)

            Only respond with the JSON object.
            """
        )
        
        self.evaluate_knowledge_prompt = PromptTemplate(
            input_variables=["segment_name", "segment_description", "topic"],
            template="""You are evaluating the existing knowledge level and potential misconceptions of a target audience segment regarding a specific topic.

            Audience Segment: {segment_name}
            Description: {segment_description}
            Topic: {topic}

            **Step 1: Reflect on Likely Background and Exposure**
            Consider the '{segment_name}' based on their description. What is their likely educational or professional background? How much exposure have they likely had to '{topic}' or related concepts? What common assumptions or simplifications might they hold about '{topic}'? Are there potential areas of confusion or common misconceptions for this type of audience?

            **Step 2: Evaluate Knowledge Level and Misconceptions**
            Based on your reflection, provide an evaluation of this segment's likely knowledge state regarding '{topic}'.
            Structure the evaluation into:
            - Assumed Knowledge: What foundational concepts or background related to '{topic}' can likely be assumed?
            - Likely Knowledge Gaps: What specific areas of '{topic}' are they likely unfamiliar with? (List 3-5 specific gaps)
            - Potential Misconceptions: What common misunderstandings or incorrect assumptions might they have about '{topic}'? (List 2-4 potential misconceptions)
            - Technical Depth Tolerance: How much technical detail are they likely comfortable with regarding '{topic}'? (e.g., High-level overview, moderate detail, deep technical specifics)

            Format your response as a JSON object with these keys:
            - assumed_knowledge: Array of assumed concepts (strings)
            - likely_knowledge_gaps: Array of specific knowledge gaps (strings)
            - potential_misconceptions: Array of potential misconceptions (strings)
            - technical_depth_tolerance: String describing tolerance level

            Only respond with the JSON object.
            """
        )
        
        self.recommend_strategies_prompt = PromptTemplate(
            input_variables=["segment_name", "topic", "needs_analysis", "knowledge_evaluation"],
            template="""You are recommending content strategies tailored to a specific audience segment for a given topic, based on their needs and knowledge level.

            Audience Segment: {segment_name}
            Topic: {topic}

            Needs Analysis Summary:
            {needs_analysis}

            Knowledge Evaluation Summary:
            {knowledge_evaluation}

            **Step 1: Reflect on Bridging the Gap**
            Consider the segment's needs, pain points, knowledge gaps, and potential misconceptions. How can content effectively bridge the gap between their current state and their desired outcomes related to '{topic}'? What approaches would resonate best given their background and technical depth tolerance? What formats or angles would be most engaging?

            **Step 2: Recommend Content Strategies**
            Based on your reflection, recommend 3-5 specific content strategies tailored for the '{segment_name}' regarding '{topic}'.
            For each strategy:
            1. Provide a clear title (e.g., "Use Case Deep Dive", "Misconception Buster Q&A", "Comparative Analysis Framework").
            2. Describe the strategy (2-3 sentences), explaining the angle, format, or approach.
            3. Explain *why* this strategy is suitable for this specific segment, referencing their needs, knowledge, or pain points.
            4. Suggest key elements or content points to include within this strategy.

            Format your response as a JSON array of strategy objects with these fields:
            - title: Title of the content strategy
            - description: Description of the strategy's angle/format
            - suitability_rationale: Explanation of why it fits this segment
            - key_elements: Array of suggested content points/elements

            Only respond with the JSON array.
            """
        )
    
    async def identify_audience_segments(self, topic: str) -> List[Dict[str, Any]]:
        """
        Identify target audience segments for a topic, reflecting on relevance first.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            List of audience segment dictionaries
        """
        try:
            # Format existing segments for prompt
            existing_segments_text = json.dumps(self.DEFAULT_AUDIENCE_SEGMENTS, indent=2)
            
            # Create chain for segment identification
            chain = LLMChain(
                llm=self.llm,
                prompt=self.identify_segments_prompt
            )
            
            # Run the chain
            logger.info(f"Identifying audience segments for topic: {topic} with relevance reflection...")
            response = await chain.arun(
                topic=topic,
                min_segments=self.min_segments,
                existing_segments=existing_segments_text
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
        Analyze the needs and pain points of a segment, reflecting on perspective first.
        
        Args:
            segment: Audience segment dictionary
            topic: Topic being analyzed
            
        Returns:
            Dictionary with needs analysis (needs, questions, pain points, outcomes)
        """
        try:
            # Create chain for needs analysis
            chain = LLMChain(
                llm=self.llm,
                prompt=self.analyze_needs_prompt
            )
            
            # Run the chain
            segment_name = segment.get('name', 'Unknown Segment')
            logger.info(f"Analyzing needs for segment: {segment_name} with perspective reflection...")
            response = await chain.arun(
                segment_name=segment_name,
                segment_description=segment.get('description', ''),
                topic=topic
            )
            
            # Parse JSON response
            needs = json.loads(response)
            
            logger.info(f"Analyzed needs for segment: {segment_name}")
            return needs
            
        except Exception as e:
            segment_name = segment.get('name', 'Unknown Segment')
            logger.error(f"Error analyzing needs for segment {segment_name}: {e}")
            raise AudienceAnalysisError(f"Failed to analyze segment needs: {e}")
    
    async def evaluate_segment_knowledge(
        self,
        segment: Dict[str, Any],
        topic: str
    ) -> Dict[str, Any]:
        """
        Evaluate the knowledge level of a segment, reflecting on background first.
        
        Args:
            segment: Audience segment dictionary
            topic: Topic being analyzed
            
        Returns:
            Dictionary with knowledge evaluation (assumed, gaps, misconceptions, depth)
        """
        try:
            # Create chain for knowledge evaluation
            chain = LLMChain(
                llm=self.llm,
                prompt=self.evaluate_knowledge_prompt
            )
            
            # Run the chain
            segment_name = segment.get('name', 'Unknown Segment')
            logger.info(f"Evaluating knowledge for segment: {segment_name} with background reflection...")
            response = await chain.arun(
                segment_name=segment_name,
                segment_description=segment.get('description', ''),
                topic=topic
            )
            
            # Parse JSON response
            knowledge = json.loads(response)
            
            logger.info(f"Evaluated knowledge for segment: {segment_name}")
            return knowledge
            
        except Exception as e:
            segment_name = segment.get('name', 'Unknown Segment')
            logger.error(f"Error evaluating knowledge for segment {segment_name}: {e}")
            raise AudienceAnalysisError(f"Failed to evaluate segment knowledge: {e}")
    
    async def recommend_content_strategies(
        self,
        segment: Dict[str, Any],
        topic: str,
        needs: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Recommend content strategies for a segment, reflecting on bridging gaps first.
        
        Args:
            segment: Audience segment dictionary
            topic: Topic being analyzed
            needs: Needs analysis dictionary
            knowledge: Knowledge evaluation dictionary
            
        Returns:
            List of recommended content strategy dictionaries
        """
        try:
            # Format needs and knowledge for prompt
            needs_text = json.dumps(needs, indent=2)
            knowledge_text = json.dumps(knowledge, indent=2)
            
            # Create chain for strategy recommendation
            chain = LLMChain(
                llm=self.llm,
                prompt=self.recommend_strategies_prompt
            )
            
            # Run the chain
            segment_name = segment.get('name', 'Unknown Segment')
            logger.info(f"Recommending strategies for segment: {segment_name} with gap-bridging reflection...")
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = await chain.arun(
                        segment_name=segment_name,
                        topic=topic,
                        needs_analysis=needs_text,
                        knowledge_evaluation=knowledge_text
                    )
                    
                    # Parse JSON response
                    strategies = json.loads(response)
                    
                    logger.info(f"Recommended {len(strategies)} strategies for segment: {segment_name}")
                    return strategies
                    
                except Exception as e:
                    error_msg = str(e)
                    retry_count += 1
                    
                    # Check if it's a rate limit error
                    if any(err in error_msg.lower() for err in ["429", "rate limit", "quota", "capacity", "tokens per min"]):
                        if retry_count < max_retries:
                            wait_time = 5 * retry_count  # Exponential backoff
                            logger.warning(f"Rate limit error, retrying in {wait_time} seconds: {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Maximum retries reached for rate limit error: {e}")
                            raise
                    else:
                        # Not a rate limit error, re-raise
                        raise
            
            # Should not reach here, but just in case
            raise ValueError("Exceeded maximum retries")
            
        except Exception as e:
            segment_name = segment.get('name', 'Unknown Segment')
            logger.error(f"Error recommending strategies for segment {segment_name}: {e}")
            
            # Return fallback strategies if we can't get the API to work
            if "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower():
                logger.warning(f"Using fallback strategies due to API rate limits")
                return self._fallback_strategies(segment_name, topic)
            
            raise AudienceAnalysisError(f"Failed to recommend content strategies: {e}")
    
    def _fallback_strategies(self, segment_name: str, topic: str) -> List[Dict[str, Any]]:
        """
        Provide fallback content strategies when API calls fail.
        
        Args:
            segment_name: Name of the audience segment
            topic: Topic being analyzed
            
        Returns:
            List of basic content strategies
        """
        # Create some basic strategies that would work for most segments
        return [
            {
                "title": "Comprehensive Guide",
                "description": f"A detailed guide explaining {topic} from the ground up, tailored for {segment_name}",
                "suitability_rationale": f"Provides complete information for {segment_name} regardless of prior knowledge level",
                "key_elements": [
                    "Step-by-step explanations",
                    "Visual diagrams",
                    "Real-world examples",
                    "Common pitfalls to avoid"
                ]
            },
            {
                "title": "Problem-Solution Framework",
                "description": f"Content that frames {topic} in terms of common problems faced by {segment_name} and their solutions",
                "suitability_rationale": "Directly addresses pain points and motivates engagement through practical problem-solving",
                "key_elements": [
                    "Problem statements relevant to the audience",
                    "Step-by-step solutions",
                    "Case studies and success stories",
                    "Implementation guidance"
                ]
            },
            {
                "title": "Comparison Analysis",
                "description": f"Compare and contrast {topic} with alternatives or previous approaches familiar to {segment_name}",
                "suitability_rationale": "Builds on existing knowledge and helps with decision-making",
                "key_elements": [
                    "Side-by-side comparisons",
                    "Pros and cons analysis",
                    "Use case considerations",
                    "Decision framework"
                ]
            }
        ]
    
    async def find_sources_for_segment(
        self, 
        segment: Dict[str, Any],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find sources relevant to an audience segment.
        
        Args:
            segment: Audience segment to find sources for
            count: Number of sources to find
            
        Returns:
            List of sources
        """
        all_sources = []
        
        # Use search terms if available
        search_terms = segment.get("search_terms", [])
        
        if not search_terms:
            # Fallback if no search terms provided
            search_terms = [segment.get("name", "")]
        
        # Use up to 3 search terms
        for term in search_terms[:3]:
            try:
                # Search for sources
                query = f"{term} audience data {segment.get('name', '')}"
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
            # Logger message moved to identify_audience_segments
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
                segment_name = segment.get('name', 'Unknown Segment')
                # Find sources
                logger.info(f"Finding sources for segment: {segment_name}")
                sources = await self.find_sources_for_segment(segment)
                
                # Analyze needs
                # Logger message moved to analyze_segment_needs
                needs = await self.analyze_segment_needs(segment, topic)
                
                # Evaluate knowledge
                # Logger message moved to evaluate_segment_knowledge
                knowledge = await self.evaluate_segment_knowledge(segment, topic)
                
                # Recommend strategies
                # Logger message moved to recommend_content_strategies
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