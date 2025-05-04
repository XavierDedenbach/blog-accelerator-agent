"""
Visual Asset Collector component for the Blog Accelerator Agent.

This component:
1. Collects visual assets related to the topic and provided research
2. Categorizes visuals by type (diagrams, charts, photos, illustrations)
3. Filters visuals by relevance and quality
4. Provides metadata and captions for each visual
5. Integrates with Firecrawl for image search and retrieval
"""

import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from agents.utilities.firecrawl_client import FirecrawlClient
from agents.utilities.source_validator import SourceValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualAssetCollectionError(Exception):
    """Exception raised for errors in visual asset collection."""
    pass


class VisualAssetCollector:
    """
    Component for collecting, categorizing, and filtering visual assets.
    
    Features:
    - Search and collection of solution visuals (50-100)
    - Search and collection of paradigm visuals (10-20)
    - Categorization by visual type
    - Filtering for relevance and quality
    - Generation of captions and metadata
    """
    
    def __init__(
        self,
        openai_api_key: str,
        firecrawl_client: Optional[FirecrawlClient] = None,
        source_validator: Optional[SourceValidator] = None,
        model_name: str = "gpt-4o"
    ):
        """
        Initialize the Visual Asset Collector component.
        
        Args:
            openai_api_key: OpenAI API key for LLM
            firecrawl_client: Firecrawl client for image search and retrieval
            source_validator: Source validator for validating image sources
            model_name: LLM model to use
        """
        self.openai_api_key = openai_api_key
        
        # Initialize or use provided firecrawl client
        self.firecrawl_client = firecrawl_client or FirecrawlClient()
        
        # Initialize or use provided source validator
        self.source_validator = source_validator or SourceValidator()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=openai_api_key,
            temperature=0.2
        )
        
        # Initialize prompt templates
        self._init_prompts()
        
        # Define visual asset categories
        self.categories = {
            "diagrams": {
                "description": "Visual representations showing relationships, processes, or structures",
                "search_terms": ["diagram", "flowchart", "architecture", "structure"]
            },
            "charts": {
                "description": "Data visualizations showing trends, comparisons, or distributions",
                "search_terms": ["chart", "graph", "data visualization", "dashboard"]
            },
            "illustrations": {
                "description": "Artistic representations of concepts, ideas, or scenarios",
                "search_terms": ["illustration", "concept art", "visual representation"]
            },
            "photos": {
                "description": "Photographic images showing real-world examples or implementations",
                "search_terms": ["photo", "image", "screenshot", "implementation"]
            },
            "infographics": {
                "description": "Information-rich visuals combining data and graphics",
                "search_terms": ["infographic", "data story", "visual explanation"]
            }
        }
        
        # Define visual asset collections
        self.collections = {
            "solutions": {
                "description": "Visuals related to solution approaches, implementations, or examples",
                "target_count": 50
            },
            "paradigms": {
                "description": "Visuals related to historical paradigms, trends, or future projections",
                "target_count": 15
            }
        }
    
    def _init_prompts(self):
        """Initialize prompt templates for visual asset collection."""
        
        # Prompt for generating visual search queries
        self.generate_queries_prompt = PromptTemplate(
            input_variables=["topic", "collection_type", "category", "research_data"],
            template="""
            You are a visual research specialist for a Blog Accelerator Agent.
            
            Your task is to generate effective search queries for finding high-quality visual assets.
            
            # TOPIC
            {topic}
            
            # COLLECTION TYPE
            {collection_type} ({collections_dict})
            
            # VISUAL CATEGORY
            {category} ({categories_dict})
            
            # RELEVANT RESEARCH DATA
            {research_data}
            
            # SEQUENTIAL THINKING PROCESS
            
            Step 1: Identify Core Constraints
            - What specific aspects of the topic need visual representation?
            - What is the technical depth required?
            - What key concepts from the research must be visually communicated?
            
            Step 2: Consider Systemic Context
            - How do these visuals fit into the broader narrative?
            - What industry or domain context is important to capture?
            - What level of specificity vs. generality is appropriate?
            
            Step 3: Map Stakeholder Perspectives
            - Who will be viewing these visuals?
            - What visual conventions would they be familiar with?
            - What would make the visuals most valuable to them?
            
            Step 4: Generate Specific Search Queries
            - Create 5-8 precise search queries for finding relevant {category} visuals
            - Each query should be:
              * Specific enough to return relevant results
              * Include appropriate technical terminology from the research
              * Target high-quality sources likely to have professional visuals
              * Include the visual type (diagram, chart, etc.) in the query
              * For solution visuals, include implementation or practical aspects
              * For paradigm visuals, include historical or trend elements
            
            Step 5: Prioritize for Diversity and Comprehensiveness
            - Ensure queries cover different aspects of the topic
            - Include both general and specific approaches
            - Incorporate alternative terminology/synonyms
            - Consider both current and forward-looking terminology
            
            Step 6: Verify Query Effectiveness
            - Review and refine queries for precision and relevance
            - Ensure queries will likely return diverse results
            - Eliminate queries that are too generic or likely to return irrelevant results
            
            # OUTPUT INSTRUCTIONS
            
            Return a list of 5-8 search queries for finding {category} visuals related to {collection_type} for the topic.
            
            Format your response as a JSON array of strings, like this:
            ["query 1", "query 2", "query 3", "query 4", "query 5"]
            """
        )
        
        # Prompt for filtering and categorizing visuals
        self.filter_categorize_prompt = PromptTemplate(
            input_variables=["topic", "collection_type", "category", "visual_assets", "research_data"],
            template="""
            You are a visual content curator for a Blog Accelerator Agent.
            
            Your task is to filter and categorize visual assets based on relevance, quality, and alignment with research.
            
            # TOPIC
            {topic}
            
            # COLLECTION TYPE
            {collection_type} ({collections_dict})
            
            # VISUAL CATEGORY
            {category} ({categories_dict})
            
            # VISUAL ASSETS
            {visual_assets}
            
            # RELEVANT RESEARCH DATA
            {research_data}
            
            # SEQUENTIAL THINKING PROCESS
            
            Step 1: Identify Core Constraints
            - What makes a visual genuinely useful for this topic?
            - What quality standards must be maintained?
            - What technical accuracy is required?
            
            Step 2: Consider Systemic Context
            - How do these visuals support the overall narrative?
            - What industry or domain standards apply?
            - How should these visuals complement the written content?
            
            Step 3: Map Stakeholder Perspectives
            - How would the target audience interpret these visuals?
            - What would make the visuals most accessible?
            - What visual conventions are expected by the audience?
            
            Step 4: Evaluate Each Visual Asset
            - Assess relevance to the specific topic and research
            - Evaluate technical accuracy and alignment with research findings
            - Judge visual quality (resolution, clarity, professionalism)
            - Consider credibility of the source
            - Rate how well it communicates key concepts
            
            Step 5: Categorize and Prioritize
            - Group visuals by subthemes within the category
            - Prioritize based on relevance, quality, and uniqueness
            - Identify any gaps in visual coverage of key concepts
            
            Step 6: Test Alternative Interpretations
            - Challenge initial categorizations
            - Consider if visuals could be misleading or misinterpreted
            - Identify any potential biases in the visual selection
            
            # OUTPUT INSTRUCTIONS
            
            For each visual asset, provide an assessment with the following:
            1. keep_asset (boolean): Whether to keep this visual (true/false)
            2. relevance_score (1-10): How relevant the visual is to the topic and research
            3. quality_score (1-10): The visual quality, clarity, and professionalism
            4. subcategory: A specific subcategory within the main category
            5. caption: A descriptive caption explaining the visual's content and relevance
            6. key_points: 2-3 key points this visual effectively communicates
            7. rejection_reason (if not keeping): Reason for rejection, if applicable
            
            Format your response as a JSON array of objects with these fields.
            """
        )
        
        # Prompt for generating captions and metadata
        self.generate_metadata_prompt = PromptTemplate(
            input_variables=["topic", "collection_type", "category", "visual_asset", "research_data"],
            template="""
            You are a visual content specialist for a Blog Accelerator Agent.
            
            Your task is to generate detailed metadata and captions for a visual asset.
            
            # TOPIC
            {topic}
            
            # COLLECTION TYPE
            {collection_type} ({collections_dict})
            
            # VISUAL CATEGORY
            {category} ({categories_dict})
            
            # VISUAL ASSET
            {visual_asset}
            
            # RELEVANT RESEARCH DATA
            {research_data}
            
            # SEQUENTIAL THINKING PROCESS
            
            Step 1: Identify Core Constraints
            - What key information must be conveyed about this visual?
            - What terminology should be used to describe it accurately?
            - What level of technical detail is appropriate?
            
            Step 2: Consider Systemic Context
            - How does this visual fit into the broader topic narrative?
            - What industry or domain context is important to mention?
            - How does it relate to the specific research findings?
            
            Step 3: Map Stakeholder Perspectives
            - How would different audience members interpret this visual?
            - What contextual information would make it most valuable?
            - What questions might viewers have about it?
            
            Step 4: Generate Comprehensive Metadata
            - Create a concise but informative caption
            - Identify key concepts illustrated
            - Note technical details (if relevant)
            - Highlight connection to research findings
            - Specify appropriate attribution info
            
            Step 5: Develop Supporting Context
            - Identify how this visual supports specific points in the research
            - Note any limitations or caveats about the visual's accuracy
            - Suggest optimal placement/usage within content
            
            Step 6: Test for Clarity and Accuracy
            - Verify all terminology is accurate
            - Ensure caption is clear without seeing the image
            - Confirm all descriptions are factually correct based on available info
            
            # OUTPUT INSTRUCTIONS
            
            Generate comprehensive metadata for this visual asset with the following:
            1. caption: A clear, concise caption (15-20 words)
            2. extended_description: A more detailed description (50-75 words)
            3. key_concepts: 3-5 key concepts illustrated by this visual
            4. technical_details: Any relevant technical specifications or methodologies shown
            5. research_connections: How this visual relates to specific research findings
            6. suggested_placement: Where this would be most effective in content
            7. audience_benefit: How this visual specifically helps the target audience
            
            Format your response as a JSON object with these fields.
            """
        )
    
    async def collect_solution_visuals(
        self,
        topic: str,
        solution_data: Dict[str, Any],
        count: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Collect visuals related to solution approaches for the topic.
        
        Args:
            topic: The research topic
            solution_data: Solution analysis data from the Solution Analyzer
            count: Target number of solution visuals to collect
            
        Returns:
            List of visual assets with metadata
        """
        try:
            start_time = time.time()
            logger.info(f"Starting collection of solution visuals for topic: {topic}")
            
            # Extract key data from solution analysis
            pro_arguments = solution_data.get("pro_arguments", [])
            counter_arguments = solution_data.get("counter_arguments", [])
            metrics = solution_data.get("metrics", [])
            solution_text = solution_data.get("solution", "")
            
            # Create a summary of solution data for the LLM
            solution_summary = {
                "solution": solution_text,
                "pro_arguments_summary": [arg.get("name", "") for arg in pro_arguments],
                "counter_arguments_summary": [arg.get("name", "") for arg in counter_arguments],
                "metrics_summary": [metric.get("name", "") for metric in metrics]
            }
            
            # Collect visuals for each category with appropriate distribution
            all_visuals = []
            category_distribution = {
                "diagrams": 0.3,  # 30%
                "charts": 0.25,   # 25%
                "illustrations": 0.25,  # 25%
                "photos": 0.1,    # 10%
                "infographics": 0.1  # 10%
            }
            
            for category, percentage in category_distribution.items():
                category_count = int(count * percentage)
                logger.info(f"Collecting {category_count} {category} for solutions")
                
                category_visuals = await self._collect_category_visuals(
                    topic=topic,
                    collection_type="solutions",
                    category=category,
                    research_data=solution_summary,
                    target_count=category_count
                )
                
                all_visuals.extend(category_visuals)
            
            # Calculate stats
            stats = {
                "total_visuals_collected": len(all_visuals),
                "collection_duration_seconds": round(time.time() - start_time, 2),
                "visuals_by_category": {category: sum(1 for v in all_visuals if v.get("category") == category) 
                                       for category in self.categories.keys()},
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Completed collection of solution visuals: {stats['total_visuals_collected']} collected")
            
            return {
                "topic": topic,
                "collection_type": "solutions",
                "visuals": all_visuals,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error collecting solution visuals: {e}")
            raise VisualAssetCollectionError(f"Failed to collect solution visuals: {e}")
    
    async def collect_paradigm_visuals(
        self,
        topic: str,
        paradigm_data: Dict[str, Any],
        count: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Collect visuals related to historical paradigms and future projections.
        
        Args:
            topic: The research topic
            paradigm_data: Paradigm analysis data from the Paradigm Analyzer
            count: Target number of paradigm visuals to collect
            
        Returns:
            List of visual assets with metadata
        """
        try:
            start_time = time.time()
            logger.info(f"Starting collection of paradigm visuals for topic: {topic}")
            
            # Extract key data from paradigm analysis
            historical_paradigms = paradigm_data.get("historical_paradigms", [])
            transitions = paradigm_data.get("transitions", [])
            future_paradigms = paradigm_data.get("future_paradigms", [])
            
            # Create a summary of paradigm data for the LLM
            paradigm_summary = {
                "historical_paradigms_summary": [p.get("name", "") for p in historical_paradigms],
                "transitions_summary": [f"From {t.get('from_paradigm', '')} to {t.get('to_paradigm', '')}" 
                                      for t in transitions],
                "future_paradigms_summary": [p.get("name", "") for p in future_paradigms]
            }
            
            # Collect visuals for each category with appropriate distribution
            all_visuals = []
            category_distribution = {
                "diagrams": 0.35,   # 35%
                "charts": 0.3,      # 30%
                "illustrations": 0.25,  # 25%
                "photos": 0.05,     # 5%
                "infographics": 0.05   # 5%
            }
            
            for category, percentage in category_distribution.items():
                category_count = max(1, int(count * percentage))
                logger.info(f"Collecting {category_count} {category} for paradigms")
                
                category_visuals = await self._collect_category_visuals(
                    topic=topic,
                    collection_type="paradigms",
                    category=category,
                    research_data=paradigm_summary,
                    target_count=category_count
                )
                
                all_visuals.extend(category_visuals)
            
            # Calculate stats
            stats = {
                "total_visuals_collected": len(all_visuals),
                "collection_duration_seconds": round(time.time() - start_time, 2),
                "visuals_by_category": {category: sum(1 for v in all_visuals if v.get("category") == category) 
                                       for category in self.categories.keys()},
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Completed collection of paradigm visuals: {stats['total_visuals_collected']} collected")
            
            return {
                "topic": topic,
                "collection_type": "paradigms",
                "visuals": all_visuals,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error collecting paradigm visuals: {e}")
            raise VisualAssetCollectionError(f"Failed to collect paradigm visuals: {e}")
    
    async def _collect_category_visuals(
        self,
        topic: str,
        collection_type: str,
        category: str,
        research_data: Dict[str, Any],
        target_count: int
    ) -> List[Dict[str, Any]]:
        """
        Collect visuals for a specific category.
        
        Args:
            topic: The research topic
            collection_type: Type of collection (solutions or paradigms)
            category: Visual category (diagrams, charts, etc.)
            research_data: Research data summary
            target_count: Target number of visuals to collect
            
        Returns:
            List of visual assets with metadata
        """
        # Generate search queries
        search_queries = await self._generate_search_queries(
            topic=topic,
            collection_type=collection_type,
            category=category,
            research_data=research_data
        )
        
        # Calculate visuals needed per query
        visuals_per_query = max(3, target_count // len(search_queries))
        
        # Collect and process visuals
        collected_visuals = []
        
        for query in search_queries:
            # Break if we've collected enough visuals
            if len(collected_visuals) >= target_count:
                break
                
            # Search for images
            search_results = self.firecrawl_client.search_images(
                query=query,
                count=visuals_per_query * 2,  # Get extra to allow for filtering
                image_type=self._get_image_type_for_category(category),
                min_width=800,
                min_height=600
            )
            
            # Download and process images
            for result in search_results:
                # Break if we've collected enough visuals
                if len(collected_visuals) >= target_count:
                    break
                    
                # Download the image
                downloaded = self.firecrawl_client.download_image(result.get("url"))
                
                if downloaded:
                    # Add basic metadata
                    visual_asset = {
                        "url": result.get("url"),
                        "title": result.get("title", ""),
                        "source_url": result.get("source_url", ""),
                        "width": downloaded.get("width"),
                        "height": downloaded.get("height"),
                        "format": downloaded.get("format"),
                        "file_path": downloaded.get("file_path"),
                        "file_size": downloaded.get("file_size"),
                        "collection_type": collection_type,
                        "category": category,
                        "search_query": query,
                        "collected_at": datetime.now().isoformat()
                    }
                    
                    # Add to collection
                    collected_visuals.append(visual_asset)
        
        # Filter and categorize the collected visuals
        filtered_visuals = await self._filter_and_categorize_visuals(
            topic=topic,
            collection_type=collection_type,
            category=category,
            visual_assets=collected_visuals,
            research_data=research_data
        )
        
        # Generate detailed metadata for each visual
        enriched_visuals = []
        
        for visual in filtered_visuals:
            if visual.get("keep_asset", True):
                # Generate metadata
                metadata = await self._generate_visual_metadata(
                    topic=topic,
                    collection_type=collection_type,
                    category=category,
                    visual_asset=visual,
                    research_data=research_data
                )
                
                # Merge visual with metadata
                enriched_visual = {**visual, **metadata}
                enriched_visuals.append(enriched_visual)
        
        return enriched_visuals[:target_count]
    
    async def _generate_search_queries(
        self,
        topic: str,
        collection_type: str,
        category: str,
        research_data: Dict[str, Any]
    ) -> List[str]:
        """
        Generate search queries for finding visual assets.
        
        Args:
            topic: The research topic
            collection_type: Type of collection (solutions or paradigms)
            category: Visual category (diagrams, charts, etc.)
            research_data: Research data summary
            
        Returns:
            List of search queries
        """
        try:
            # Create LLMChain for query generation
            query_chain = LLMChain(
                llm=self.llm,
                prompt=self.generate_queries_prompt
            )
            
            # Run the chain
            response = await query_chain.arun(
                topic=topic,
                collection_type=collection_type,
                category=category,
                categories_dict=self.categories[category]["description"],
                collections_dict=self.collections[collection_type]["description"],
                research_data=json.dumps(research_data, indent=2)
            )
            
            # Parse the JSON response
            try:
                queries = json.loads(response)
                return queries
            except json.JSONDecodeError:
                # Fallback: try to extract queries manually if JSON parsing fails
                lines = response.strip().split('\n')
                queries = []
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        queries.append(line.strip('"'))
                    elif line.startswith("'") and line.endswith("'"):
                        queries.append(line.strip("'"))
                
                if not queries:
                    # Last resort: generate default queries
                    search_terms = self.categories[category]["search_terms"]
                    queries = [f"{topic} {term}" for term in search_terms]
                
                return queries[:8]  # Limit to 8 queries
                
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            # Fallback to basic queries
            search_terms = self.categories[category]["search_terms"]
            return [f"{topic} {term}" for term in search_terms]
    
    async def _filter_and_categorize_visuals(
        self,
        topic: str,
        collection_type: str,
        category: str,
        visual_assets: List[Dict[str, Any]],
        research_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter and categorize visual assets.
        
        Args:
            topic: The research topic
            collection_type: Type of collection (solutions or paradigms)
            category: Visual category (diagrams, charts, etc.)
            visual_assets: List of visual assets to filter and categorize
            research_data: Research data summary
            
        Returns:
            Filtered and categorized visual assets
        """
        try:
            # If no visuals, return empty list
            if not visual_assets:
                return []
                
            # Create LLMChain for filtering
            filter_chain = LLMChain(
                llm=self.llm,
                prompt=self.filter_categorize_prompt
            )
            
            # Run the chain
            response = await filter_chain.arun(
                topic=topic,
                collection_type=collection_type,
                category=category,
                categories_dict=self.categories[category]["description"],
                collections_dict=self.collections[collection_type]["description"],
                visual_assets=json.dumps([{
                    "url": asset.get("url"),
                    "title": asset.get("title"),
                    "source_url": asset.get("source_url"),
                    "width": asset.get("width"),
                    "height": asset.get("height")
                } for asset in visual_assets], indent=2),
                research_data=json.dumps(research_data, indent=2)
            )
            
            # Parse the JSON response
            try:
                filtered_assets = json.loads(response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse filter response as JSON: {response}")
                # Fallback: keep all assets with basic metadata
                filtered_assets = [{
                    "keep_asset": True,
                    "relevance_score": 5,
                    "quality_score": 5,
                    "subcategory": "general",
                    "caption": asset.get("title", ""),
                    "key_points": []
                } for asset in visual_assets]
            
            # Merge filtering results with original assets
            result = []
            
            for i, assessment in enumerate(filtered_assets):
                if i < len(visual_assets):
                    # Merge with original asset data
                    merged = {**visual_assets[i], **assessment}
                    result.append(merged)
            
            return result
                
        except Exception as e:
            logger.error(f"Error filtering visual assets: {e}")
            # Fallback: return original assets with default metadata
            return [{
                **asset,
                "keep_asset": True,
                "relevance_score": 5,
                "quality_score": 5,
                "subcategory": "general",
                "caption": asset.get("title", ""),
                "key_points": []
            } for asset in visual_assets]
    
    async def _generate_visual_metadata(
        self,
        topic: str,
        collection_type: str,
        category: str,
        visual_asset: Dict[str, Any],
        research_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate detailed metadata for a visual asset.
        
        Args:
            topic: The research topic
            collection_type: Type of collection (solutions or paradigms)
            category: Visual category (diagrams, charts, etc.)
            visual_asset: The visual asset to generate metadata for
            research_data: Research data summary
            
        Returns:
            Visual asset metadata
        """
        try:
            # Create LLMChain for metadata generation
            metadata_chain = LLMChain(
                llm=self.llm,
                prompt=self.generate_metadata_prompt
            )
            
            # Run the chain
            response = await metadata_chain.arun(
                topic=topic,
                collection_type=collection_type,
                category=category,
                categories_dict=self.categories[category]["description"],
                collections_dict=self.collections[collection_type]["description"],
                visual_asset=json.dumps({
                    "url": visual_asset.get("url"),
                    "title": visual_asset.get("title"),
                    "source_url": visual_asset.get("source_url"),
                    "width": visual_asset.get("width"),
                    "height": visual_asset.get("height"),
                    "caption": visual_asset.get("caption", ""),
                    "subcategory": visual_asset.get("subcategory", "")
                }, indent=2),
                research_data=json.dumps(research_data, indent=2)
            )
            
            # Parse the JSON response
            try:
                metadata = json.loads(response)
                return metadata
            except json.JSONDecodeError:
                logger.error(f"Failed to parse metadata response as JSON: {response}")
                # Fallback: return basic metadata
                return {
                    "caption": visual_asset.get("caption", visual_asset.get("title", "")),
                    "extended_description": f"Visual asset related to {topic}",
                    "key_concepts": ["Topic relevance"],
                    "technical_details": "",
                    "research_connections": "",
                    "suggested_placement": "Body content",
                    "audience_benefit": "Visual illustration of key concepts"
                }
                
        except Exception as e:
            logger.error(f"Error generating visual metadata: {e}")
            # Fallback: return basic metadata
            return {
                "caption": visual_asset.get("caption", visual_asset.get("title", "")),
                "extended_description": f"Visual asset related to {topic}",
                "key_concepts": ["Topic relevance"],
                "technical_details": "",
                "research_connections": "",
                "suggested_placement": "Body content",
                "audience_benefit": "Visual illustration of key concepts"
            }
    
    def _get_image_type_for_category(self, category: str) -> Optional[str]:
        """
        Get the appropriate image type parameter for Firecrawl search.
        
        Args:
            category: Visual category (diagrams, charts, etc.)
            
        Returns:
            Image type parameter for Firecrawl
        """
        if category == "diagrams":
            return "diagram"
        elif category == "charts":
            return "chart"
        elif category == "illustrations":
            return "illustration"
        elif category == "photos":
            return "photo"
        elif category == "infographics":
            return "infographic"
        else:
            return None
    
    async def analyze_visual_assets(
        self,
        topic: str,
        solution_data: Dict[str, Any],
        paradigm_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a topic by collecting and organizing visual assets.
        
        Args:
            topic: The research topic
            solution_data: Solution analysis data
            paradigm_data: Paradigm analysis data
            
        Returns:
            Complete visual asset collection with metadata
        """
        try:
            start_time = time.time()
            logger.info(f"Starting visual asset collection for topic: {topic}")
            
            # Collect solution visuals
            solution_visuals_result = await self.collect_solution_visuals(
                topic=topic,
                solution_data=solution_data
            )
            
            # Collect paradigm visuals
            paradigm_visuals_result = await self.collect_paradigm_visuals(
                topic=topic,
                paradigm_data=paradigm_data
            )
            
            # Combine results
            solution_visuals = solution_visuals_result.get("visuals", [])
            paradigm_visuals = paradigm_visuals_result.get("visuals", [])
            
            # Calculate aggregate stats
            stats = {
                "total_solution_visuals": len(solution_visuals),
                "total_paradigm_visuals": len(paradigm_visuals),
                "total_visuals": len(solution_visuals) + len(paradigm_visuals),
                "visuals_by_category": {
                    category: sum(1 for v in solution_visuals + paradigm_visuals if v.get("category") == category)
                    for category in self.categories.keys()
                },
                "analysis_duration_seconds": round(time.time() - start_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # Build final result
            result = {
                "topic": topic,
                "solution_visuals": solution_visuals,
                "paradigm_visuals": paradigm_visuals,
                "stats": stats
            }
            
            logger.info(f"Completed visual asset collection: {stats['total_visuals']} total visuals collected")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in visual asset analysis: {e}")
            raise VisualAssetCollectionError(f"Failed to analyze visual assets: {e}") 