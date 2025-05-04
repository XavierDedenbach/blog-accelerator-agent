"""
Industry Analysis component for the Blog Accelerator Agent.

This module:
1. Analyzes industry or system affected by a topic
2. Identifies 10+ critical challenges
3. Finds authoritative sources for each challenge
4. Provides detailed analysis of challenge components (risk, inefficiency, etc.)
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


class IndustryAnalysisError(Exception):
    """Exception raised for errors in industry analysis."""
    pass


class IndustryAnalyzer:
    """
    Analyzer for industry or system affected by a topic.
    
    Features:
    - Identification of 10+ critical challenges
    - Supporting sources for each challenge
    - Analysis of core challenge components
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        source_validator: Optional[SourceValidator] = None,
        min_challenges: int = 10
    ):
        """
        Initialize the industry analyzer.
        
        Args:
            openai_api_key: OpenAI API key
            groq_api_key: Groq API key
            source_validator: SourceValidator instance
            min_challenges: Minimum number of challenges to identify
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
            raise IndustryAnalysisError("No API key provided for LLM")
        
        # Initialize source validator if not provided
        self.source_validator = source_validator or SourceValidator()
        
        # Set minimum challenges threshold
        self.min_challenges = min_challenges
        
        # Load prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompts for industry analysis."""
        self.identify_challenges_prompt = PromptTemplate(
            input_variables=["topic", "min_challenges"],
            template="""You are analyzing the industry or system affected by the topic: {topic}.
            Assume the context often involves small, resource-constrained teams (like hardware startups) facing these challenges.

            **Step 1: Reflect on Core Constraints**
            Before identifying challenges, briefly consider the inherent difficulties and constraints related to '{topic}'. 
            Focus specifically on how these challenges affect different user roles/personas mentioned in the topic.
            Key considerations:
            - Who are the specific USER TYPES or ROLES mentioned in the topic (e.g., "policy makers", "developers", "engineers", "healthcare providers")?
            - What SPECIFIC CONSTRAINTS do these user types face (budget limits, technical knowledge, decision authority)?
            - How are these user types UNIQUELY AFFECTED by limitations in:
              - Resource availability (funding, personnel)
              - Domain expertise and knowledge gaps
              - Technical capabilities or access
              - Regulatory or compliance requirements
              - Decision-making authority and processes
              - Physical vs. digital constraints (if applicable)
              - Time constraints and urgency factors

            **Step 2: Identify Critical Challenges**
            Based on your reflection and the topic '{topic}', identify at least {min_challenges} critical challenges facing the relevant industry or system. Focus on challenges that are:
            1. Significant and impactful for the SPECIFIC USER TYPES identified in Step 1
            2. Measurable or observable
            3. Relevant to the topic
            4. Diverse in nature (technical, economic, operational, regulatory, etc.)
            5. Specific rather than general

            For each challenge:
            1. Provide a clear name
            2. Write a detailed description (2-3 sentences), explaining *why* it's a challenge SPECIFICALLY FOR THE USER TYPES identified
            3. Explain why this is a *critical* challenge for these user types
            4. Suggest search terms to find authoritative sources about this challenge, considering the nuances discussed.

            Format your response as a JSON array of challenge objects with these fields:
            - name: Name of the challenge
            - description: Detailed description linked to user types and their constraints
            - criticality: Explanation of why it's critical for specific user types
            - search_terms: Array of nuanced search terms

            Only respond with the JSON array.
            """
        )
        
        self.analyze_challenge_component_prompt = PromptTemplate(
            input_variables=["challenge", "description", "sources"],
            template="""You are analyzing the core components of a specific industry challenge, considering the context of resource-constrained teams.

            Challenge Name: {challenge}
            Description: {description}

            Below are sources with information about this challenge:
            {sources}

            **Step 1: Reflect on Constraints**
            Briefly reconsider the core constraints (e.g., small teams, limited funding, hardware complexities) and how they specifically impact the challenge '{challenge}'.

            **Step 2: Analyze Components**
            Based on the sources provided AND your reflection on constraints, provide a comprehensive analysis of the core components that make this challenge:
            1. Risky: What specific risks does this challenge pose, amplified by the constraints?
            2. Slow: What aspects cause delays or inefficiencies, especially for small teams?
            3. Expensive: What cost factors are involved, and why are they particularly burdensome?
            4. Inefficient: What specific inefficiencies exist, and how do they stem from or interact with the constraints?

            For each component, cite specific evidence from the sources provided where possible, but also incorporate insights derived from the constraints.

            Format your response as a JSON object with these keys:
            - risk_factors: Array of specific risk factors (with source citations and constraint links)
            - slowdown_factors: Array of factors causing delays (with source citations and constraint links)
            - cost_factors: Array of cost factors (with source citations and constraint links)
            - inefficiency_factors: Array of specific inefficiencies (with source citations and constraint links)

            Only respond with the JSON object.
            """
        )
    
    async def identify_challenges(self, topic: str) -> List[Dict[str, Any]]:
        """
        Identify critical challenges facing an industry or system, reflecting on constraints first.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            List of challenge dictionaries
        """
        try:
            # Create chain for challenge identification
            chain = LLMChain(
                llm=self.llm,
                prompt=self.identify_challenges_prompt
            )
            
            # Run the chain
            logger.info(f"Identifying challenges for topic: {topic} with constraint reflection...")
            response = await chain.arun(
                topic=topic,
                min_challenges=self.min_challenges
            )
            
            # Parse JSON response
            challenges = json.loads(response)
            
            logger.info(f"Identified {len(challenges)} challenges for topic: {topic}")
            return challenges
            
        except Exception as e:
            logger.error(f"Error identifying challenges for {topic}: {e}")
            raise IndustryAnalysisError(f"Failed to identify challenges: {e}")
    
    async def find_sources_for_challenge(
        self, 
        challenge: Dict[str, Any],
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find authoritative sources for a challenge.
        
        Args:
            challenge: Challenge dictionary
            count: Number of sources to find
            
        Returns:
            List of source dictionaries
        """
        all_sources = []
        search_terms = challenge.get("search_terms", [])
        
        if not search_terms:
            # Fallback if no search terms provided
            search_terms = [challenge.get("name", "")]
        
        # Use up to 3 search terms
        for term in search_terms[:3]:
            try:
                # Search for sources
                query = f"{term} {challenge.get('name', '')}"
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
        
        logger.info(f"Found {len(unique_sources)} sources for challenge: {challenge.get('name', '')}")
        return unique_sources
    
    async def analyze_challenge_components(
        self,
        challenge: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the components of a challenge, reflecting on constraints first.
        
        Args:
            challenge: Challenge dictionary
            sources: List of source dictionaries
            
        Returns:
            Dictionary with analysis of risk, slowdown, cost, and inefficiency factors
        """
        try:
            # Format sources for prompt
            sources_text = ""
            for i, source in enumerate(sources, 1):
                sources_text += f"Source {i}: {source.get('title', 'Untitled')}\n"
                sources_text += f"URL: {source.get('url', 'No URL')}\n"
                # Limit description length if needed
                desc = source.get('description', 'No description')
                sources_text += f"Description: {desc[:300]}{'...' if len(desc) > 300 else ''}\n\n"
            
            # Create chain for component analysis
            chain = LLMChain(
                llm=self.llm,
                prompt=self.analyze_challenge_component_prompt
            )
            
            # Run the chain
            logger.info(f"Analyzing components for challenge: {challenge.get('name', '')} with constraint reflection...")
            response = await chain.arun(
                challenge=challenge.get("name", ""),
                description=challenge.get("description", ""),
                sources=sources_text
            )
            
            # Parse JSON response
            components = json.loads(response)
            
            logger.info(f"Analyzed components for challenge: {challenge.get('name', '')}")
            return components
            
        except Exception as e:
            logger.error(f"Error analyzing components for {challenge.get('name', '')}: {e}")
            raise IndustryAnalysisError(f"Failed to analyze challenge components: {e}")
    
    async def analyze_industry(self, topic: str) -> Dict[str, Any]:
        """
        Perform complete industry analysis for a topic.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            Dictionary with industry analysis results
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Identify challenges
            logger.info(f"Identifying challenges for topic: {topic}")
            challenges = await self.identify_challenges(topic)
            
            # Ensure we have at least min_challenges
            if len(challenges) < self.min_challenges:
                logger.warning(
                    f"Only identified {len(challenges)} challenges, " 
                    f"which is less than minimum {self.min_challenges}"
                )
            
            # Process each challenge
            enriched_challenges = []
            for challenge in challenges:
                # Find sources
                logger.info(f"Finding sources for challenge: {challenge.get('name', '')}")
                sources = await self.find_sources_for_challenge(challenge)
                
                # Analyze components
                logger.info(f"Analyzing components for challenge: {challenge.get('name', '')}")
                components = await self.analyze_challenge_components(challenge, sources)
                
                # Add to enriched challenges
                enriched_challenge = {
                    **challenge,
                    "sources": sources,
                    "components": components
                }
                enriched_challenges.append(enriched_challenge)
            
            # Calculate statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create result
            result = {
                "topic": topic,
                "challenges": enriched_challenges,
                "stats": {
                    "challenges_count": len(enriched_challenges),
                    "sources_count": sum(len(c.get("sources", [])) for c in enriched_challenges),
                    "analysis_duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(
                f"Completed industry analysis for {topic} with "
                f"{len(enriched_challenges)} challenges and "
                f"{result['stats']['sources_count']} sources"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in industry analysis for {topic}: {e}")
            raise IndustryAnalysisError(f"Failed to complete industry analysis: {e}") 