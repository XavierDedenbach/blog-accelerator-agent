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

            **Step 1: Identify Core Constraints**
            Consider the inherent limitations in audience targeting for '{topic}', such as knowledge prerequisites, access barriers, and relevance boundaries. Who is fundamentally excluded or included by the nature of the topic?

            **Step 2: Consider Systemic Context**
            Reflect on how '{topic}' fits within broader professional, social, or market ecosystems. What adjacent fields or interest areas create natural audience connections?

            **Step 3: Map Stakeholder Perspectives**
            Consider the various roles, positions, and viewpoints related to '{topic}'. Who has direct interest, indirect interest, decision-making power, or implementation responsibility?

            **Step 4: Identify Target Segments**
            Based on your reflections, define at least {min_segments} distinct and relevant audience segments for '{topic}'. You can adapt the default segments or create new ones specific to the topic. Ensure segments are:
            1. Clearly distinct from each other.
            2. Directly relevant to the specific topic '{topic}'.
            3. Described with sufficient detail to understand their perspective.

            For each segment:
            1. Provide a clear name (e.g., "Hardware Engineers implementing {topic}", "VCs evaluating {topic} startups").
            2. Write a detailed description (2-3 sentences) explaining who they are and their relationship to the topic.
            3. List their likely motivations for engaging with content on this topic.
            4. List potential pain points or challenges they face related to this topic.
            5. Estimate their general knowledge level regarding this specific topic (e.g., beginner, intermediate, advanced, expert).

            **Step 5: Generate Supporting Evidence**
            For each segment, briefly identify where you might find data or information to validate the size and characteristics of this audience (e.g., industry reports, professional associations, social media groups).

            **Step 6: Test Counter-Arguments**
            For each segment, briefly address one reason why this segment might be considered too broad, too narrow, or otherwise problematic, and explain why it remains a valid and useful audience segment despite this concern.

            Format your response as a JSON array of segment objects with these fields:
            - name: Specific name of the segment
            - description: Detailed description related to the topic
            - motivations: Array of topic-specific motivations
            - pain_points: Array of topic-specific pain points
            - knowledge_level: Estimated topic-specific knowledge level
            - validation_sources: Where to find data about this audience
            - potential_critique: A potential criticism of this segment definition
            - critique_response: Why the segment remains valid despite the critique
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

            **Step 1: Identify Core Constraints**
            Consider the fundamental limitations this audience segment faces in relation to '{topic}', such as time, resources, knowledge, or access constraints. What restricts their engagement or success?

            **Step 2: Consider Systemic Context**
            Reflect on how this segment operates within broader systems related to '{topic}'. What external pressures, requirements, or dependencies affect their needs?

            **Step 3: Map Stakeholder Perspectives**
            Consider how this segment interacts with other stakeholders in relation to '{topic}'. How do these relationships shape their specific needs and challenges?

            **Step 4: Identify Specific Needs**
            Based on your reflections, provide a detailed analysis of this segment's needs and pain points specifically related to '{topic}'.
            Structure the analysis into:
            - Information Needs: What specific information or knowledge are they lacking or seeking about '{topic}'? (List 3-5 specific needs)
            - Key Questions: What are the top 3-5 questions this segment likely has about '{topic}'?
            - Core Pain Points: What are the primary frustrations or difficulties they experience concerning '{topic}'? (List 3-5 specific pain points)
            - Desired Outcomes: What do they hope to achieve by learning about or engaging with '{topic}'?

            **Step 5: Generate Supporting Evidence**
            Briefly describe how these needs could be validated or further researched (e.g., surveys, interviews, search trend analysis, forum discussions).

            **Step 6: Test Counter-Arguments**
            For each major need or pain point identified, briefly consider one potential objection or alternative perspective, and explain why your analysis remains valid despite this challenge.

            Format your response as a JSON object with these keys:
            - information_needs: Array of specific information needs (strings)
            - key_questions: Array of likely questions (strings)
            - core_pain_points: Array of specific pain points (strings)
            - desired_outcomes: Array of desired outcomes (strings)
            - validation_methods: Ways to validate these needs
            - potential_objections: Array of possible objections to your analysis
            - objection_responses: Array of responses to those objections

            Only respond with the JSON object.
            """
        )
        
        self.evaluate_knowledge_prompt = PromptTemplate(
            input_variables=["segment_name", "segment_description", "topic"],
            template="""You are evaluating the existing knowledge level and potential misconceptions of a target audience segment regarding a specific topic.

            Audience Segment: {segment_name}
            Description: {segment_description}
            Topic: {topic}

            **Step 1: Identify Core Constraints**
            Consider what fundamental limitations affect this segment's knowledge acquisition about '{topic}', such as educational background, professional exposure, or access to information sources.

            **Step 2: Consider Systemic Context**
            Reflect on how broader systems (educational, professional, media) have shaped this segment's understanding of '{topic}'. What systemic factors influence their knowledge base?

            **Step 3: Map Stakeholder Perspectives**
            Consider how this segment's relationship with other stakeholders affects their knowledge and perspective on '{topic}'. How do interactions with others shape their understanding?

            **Step 4: Evaluate Knowledge State**
            Based on your reflections, provide an evaluation of this segment's likely knowledge state regarding '{topic}'.
            Structure the evaluation into:
            - Assumed Knowledge: What foundational concepts or background related to '{topic}' can likely be assumed?
            - Likely Knowledge Gaps: What specific areas of '{topic}' are they likely unfamiliar with? (List 3-5 specific gaps)
            - Potential Misconceptions: What common misunderstandings or incorrect assumptions might they have about '{topic}'? (List 2-4 potential misconceptions)
            - Technical Depth Tolerance: How much technical detail are they likely comfortable with regarding '{topic}'? (e.g., High-level overview, moderate detail, deep technical specifics)

            **Step 5: Generate Supporting Evidence**
            Briefly describe what evidence would help validate your assessment of this segment's knowledge level (e.g., quiz performance, common questions in forums, typical errors in implementation).

            **Step 6: Test Counter-Arguments**
            For each major knowledge gap or misconception identified, briefly consider one reason why your assessment might be incorrect, and explain why your analysis remains valid despite this possibility.

            Format your response as a JSON object with these keys:
            - assumed_knowledge: Array of assumed concepts (strings)
            - likely_knowledge_gaps: Array of specific knowledge gaps (strings)
            - potential_misconceptions: Array of potential misconceptions (strings)
            - technical_depth_tolerance: String describing tolerance level
            - validation_evidence: Ways to validate this knowledge assessment
            - assessment_challenges: Array of challenges to your knowledge assessment
            - challenge_responses: Array of responses to those challenges

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

            **Step 1: Identify Core Constraints**
            Consider what fundamental limitations might affect content strategy for this audience, such as attention span, access channels, or format preferences. What boundaries must be respected?

            **Step 2: Consider Systemic Context**
            Reflect on how this segment typically consumes content within their broader ecosystem. What contextual factors influence effective content delivery?

            **Step 3: Map Stakeholder Perspectives**
            Consider how this segment's interactions with other stakeholders should influence content strategy. How might content facilitate these relationships?

            **Step 4: Identify Content Strategies**
            Based on your reflections, recommend 3-5 specific content strategies tailored for the '{segment_name}' regarding '{topic}'.
            For each strategy:
            1. Provide a clear title (e.g., "Use Case Deep Dive", "Misconception Buster Q&A", "Comparative Analysis Framework").
            2. Describe the strategy (2-3 sentences), explaining the angle, format, or approach.
            3. Explain *why* this strategy is suitable for this specific segment, referencing their needs, knowledge, or pain points.
            4. Suggest key elements or content points to include within this strategy.

            **Step 5: Generate Supporting Evidence**
            For each strategy, briefly describe what evidence or examples suggest this approach would be effective (e.g., similar successful content, engagement patterns, learning theory).

            **Step 6: Test Counter-Arguments**
            For each strategy, briefly address one potential criticism or limitation of this approach, and explain how the strategy could be adjusted to address this concern.

            Format your response as a JSON array of strategy objects with these fields:
            - title: Title of the content strategy
            - description: Description of the strategy's angle/format
            - suitability_rationale: Explanation of why it fits this segment
            - key_elements: Array of suggested content points/elements
            - supporting_evidence: Evidence suggesting effectiveness
            - potential_limitation: A potential criticism or limitation
            - adaptation_response: How to adjust for this limitation

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