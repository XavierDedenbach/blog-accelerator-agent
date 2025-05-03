"""
Researcher agent responsible for processing topic analysis and gathering research data.

This agent:
1. Parses topic metadata from new markdown
2. Parses embedded image references
3. Calls Brave MCP for citations
4. Stores outputs to review_files and media in MongoDB
5. Assigns readiness score
6. Implements enhanced research capabilities:
   - 10+ industry/system challenges
   - 5-10 supporting/opposing arguments for solutions
   - Historical paradigm analysis
   - Visual asset collection (50-100 assets)
   - Analogy generation
"""

import os
import re
import json
import logging
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import requests

# Import utility modules
from agents.utilities.db import MongoDBClient
from agents.utilities.yaml_guard import create_tracker_yaml
from agents.utilities.file_ops import (
    read_markdown_file, parse_image_references, 
    detect_version_from_filename, FileOpsError,
    collect_blog_assets, process_blog_upload
)
from agents.utilities.source_validator import SourceValidator
from agents.utilities.firecrawl_client import FirecrawlClient

# Import research components
from agents.research import (
    IndustryAnalyzer,
    SolutionAnalyzer,
    ParadigmAnalyzer,
    AudienceAnalyzer,
    AnalogyGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TopicAnalysisError(Exception):
    """Exception raised for errors in topic analysis."""
    pass


class CitationError(Exception):
    """Exception raised for errors in citation gathering."""
    pass


class ResearcherAgent:
    """
    Agent responsible for researching topics, gathering citations,
    and preparing research documentation.
    """
    
    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        firecrawl_server: Optional[str] = None,
        opik_server: Optional[str] = None
    ):
        """
        Initialize the researcher agent.
        
        Args:
            mongodb_uri: MongoDB connection URI
            brave_api_key: API key for Brave Search API
            openai_api_key: API key for OpenAI
            groq_api_key: API key for Groq
            firecrawl_server: URL for Firecrawl MCP server
            opik_server: URL for Opik MCP server
        """
        # Initialize database connection
        self.db_client = MongoDBClient(uri=mongodb_uri)
        
        # Get API keys from environment if not provided
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.brave_api_key:
            logger.warning("Brave API key not provided. Citation gathering will be limited.")
            
        # Get Opik server from environment if not provided
        self.opik_server = opik_server or os.environ.get("OPIK_SERVER")
        self.firecrawl_server = firecrawl_server or os.environ.get("FIRECRAWL_SERVER")
        
        # Initialize source validator and firecrawl client
        self.source_validator = SourceValidator(brave_api_key=self.brave_api_key)
        self.firecrawl_client = FirecrawlClient(
            server_url=self.firecrawl_server,
            brave_api_key=self.brave_api_key
        )
        
        # Initialize all research components
        self.industry_analyzer = IndustryAnalyzer(
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            source_validator=self.source_validator
        )
        
        self.solution_analyzer = SolutionAnalyzer(
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            source_validator=self.source_validator
        )
        
        self.paradigm_analyzer = ParadigmAnalyzer(
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            source_validator=self.source_validator
        )
        
        self.audience_analyzer = AudienceAnalyzer(
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            source_validator=self.source_validator
        )
        
        self.analogy_generator = AnalogyGenerator(
            openai_api_key=self.openai_api_key,
            groq_api_key=self.groq_api_key,
            source_validator=self.source_validator,
            firecrawl_client=self.firecrawl_client
        )
        
        # Initialize readiness score thresholds
        self.readiness_thresholds = {
            "min_citations": 3,
            "min_images": 1,
            "min_headings": 3,
            "min_paragraphs": 5,
            "min_challenges": 10,
            "min_pro_arguments": 5,
            "min_counter_arguments": 5,
            "min_visual_assets": 50,
            "min_analogies": 3
        }
        
        # Define letter grade thresholds
        self.grade_thresholds = {
            "A": 90,
            "B": 80,
            "C": 70,
            "D": 60,
            "F": 0
        }
    
    def process_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a markdown file to extract metadata and content.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dict containing extracted metadata and content
        """
        logger.info(f"Processing markdown file: {file_path}")
        
        try:
            # Extract blog title and version from filename
            blog_title, version = detect_version_from_filename(file_path)
            logger.info(f"Detected blog title: {blog_title}, version: {version}")
            
            # Collect all blog assets
            blog_data = collect_blog_assets(file_path)
            
            # Extract metadata from the content
            metadata = self.extract_metadata(blog_data['content'])
            
            # Add title and version to metadata
            metadata['blog_title'] = blog_title
            metadata['version'] = version
            
            # Add to the return data
            result = {
                **blog_data,
                'metadata': metadata
            }
            
            return result
        except (FileOpsError, Exception) as e:
            logger.error(f"Error processing markdown file: {e}")
            raise TopicAnalysisError(f"Failed to process markdown file: {e}")
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            Dict containing extracted metadata
        """
        metadata = {
            'headings': [],
            'topics': [],
            'main_topic': None,
            'summary': None,
            'audience': None,
            'paragraphs_count': 0,
            'images_count': 0,
            'has_code_blocks': False,
            'has_tables': False,
            'has_lists': False,
            'reading_time_minutes': 0
        }
        
        # Extract headings (all levels)
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        metadata['headings'] = [h[1].strip() for h in headings]
        
        # Try to identify main topic from first h1 or title
        h1_headings = [h[1].strip() for h in headings if len(h[0]) == 1]
        if h1_headings:
            metadata['main_topic'] = h1_headings[0]
        
        # Count paragraphs (text blocks separated by newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        metadata['paragraphs_count'] = len(paragraphs)
        
        # Count images
        image_refs = parse_image_references(content)
        metadata['images_count'] = len(image_refs)
        
        # Check for code blocks
        metadata['has_code_blocks'] = bool(re.search(r'```\w*\n[\s\S]*?\n```', content))
        
        # Check for tables
        metadata['has_tables'] = bool(re.search(r'\|.*\|.*\n\|[\s-]*\|', content))
        
        # Check for lists
        metadata['has_lists'] = bool(re.search(r'^\s*[-*+]\s+', content, re.MULTILINE))
        
        # Estimate reading time (average adult reads ~250 words per minute)
        word_count = len(content.split())
        metadata['reading_time_minutes'] = round(word_count / 250)
        
        # Try to extract summary from first paragraph after title
        if len(paragraphs) > 1:
            # Skip headings and empty paragraphs
            for p in paragraphs:
                p = p.strip()
                if p and not p.startswith('#') and len(p) > 50:
                    metadata['summary'] = p[:200] + ('...' if len(p) > 200 else '')
                    break
        
        return metadata
    
    def search_citations(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Search for citations using Brave Search API.
        
        Args:
            query: Search query
            count: Number of results to return
            
        Returns:
            List of citation results
        """
        if not self.brave_api_key:
            logger.warning("Brave API key not available. Using mock citations.")
            # Return mock data
            return [{
                'title': f'Mock citation for "{query}"',
                'url': 'https://example.com/mock',
                'description': 'This is a mock citation result.',
                'source': 'Mock Source',
                'date': datetime.now(timezone.utc).isoformat()
            }]
        
        # Use Brave Search API
        try:
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': self.brave_api_key
            }
            
            params = {
                'q': query,
                'count': count,
                'search_lang': 'en'
            }
            
            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                logger.error(f"Error searching Brave API: {response.status_code} - {response.text}")
                raise CitationError(f"Brave Search API error: {response.status_code}")
            
            results = response.json().get('web', {}).get('results', [])
            
            citations = []
            for result in results:
                citation = {
                    'title': result.get('title'),
                    'url': result.get('url'),
                    'description': result.get('description'),
                    'source': result.get('extra_snippets', {}).get('source', 'Unknown Source'),
                    'date': datetime.now(timezone.utc).isoformat()
                }
                citations.append(citation)
                
            return citations
        
        except Exception as e:
            logger.error(f"Error in citation search: {e}")
            raise CitationError(f"Failed to get citations: {e}")
    
    async def gather_research(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather research data for a topic.
        
        Args:
            metadata: Metadata extracted from content
            
        Returns:
            Research data including citations and analysis
        """
        main_topic = metadata.get('main_topic', 'Unknown Topic')
        logger.info(f"Gathering comprehensive research for topic: {main_topic}")
        
        research_data = {
            'citations': [],
            'system_affected': {},
            'challenges': [],
            'solution': {},
            'paradigms': {},
            'audience': {},
            'analogies': []
        }
        
        # Gather initial citations
        try:
            # Generate search queries based on metadata
            queries = []
            if main_topic:
                queries.append(main_topic)
                queries.append(f"{main_topic} industry")
                queries.append(f"{main_topic} solution")
                queries.append(f"{main_topic} history")
            
            # If no main topic, use headings
            if not queries and metadata.get('headings'):
                for heading in metadata.get('headings')[:2]:
                    queries.append(heading)
            
            # Gather citations for each query
            for query in queries:
                try:
                    citations = self.search_citations(query)
                    research_data['citations'].extend(citations)
                except CitationError as e:
                    logger.warning(f"Citation error for query '{query}': {e}")
        except Exception as e:
            logger.error(f"Error gathering initial citations: {e}")
        
        # Run all research components in parallel
        try:
            # Create tasks for all research components
            industry_task = asyncio.create_task(
                self.industry_analyzer.analyze_industry(main_topic)
            )
            
            solution_task = asyncio.create_task(
                self.solution_analyzer.analyze_solution(
                    main_topic,
                    f"Innovative approach to {main_topic}",
                    []  # Will be populated with challenges once industry analysis is done
                )
            )
            
            paradigm_task = asyncio.create_task(
                self.paradigm_analyzer.analyze_paradigms(main_topic)
            )
            
            audience_task = asyncio.create_task(
                self.audience_analyzer.analyze_audience(main_topic)
            )
            
            analogy_task = asyncio.create_task(
                self.analogy_generator.generate_analogies(main_topic)
            )
            
            # Wait for industry analysis to complete first
            industry_result = await industry_task
            
            # Update challenges for solution analysis with results from industry analysis
            if 'challenges' in industry_result:
                # Cancel the existing solution task
                solution_task.cancel()
                
                # Create a new solution task with the challenges
                solution_task = asyncio.create_task(
                    self.solution_analyzer.analyze_solution(
                        main_topic,
                        f"Innovative approach to {main_topic}",
                        industry_result.get('challenges', [])
                    )
                )
            
            # Wait for all tasks to complete
            solution_result = await solution_task
            paradigm_result = await paradigm_task
            audience_result = await audience_task
            analogy_result = await analogy_task
            
            # Update research data with results
            research_data['system_affected'] = industry_result
            research_data['challenges'] = industry_result.get('challenges', [])
            research_data['solution'] = solution_result
            research_data['paradigms'] = paradigm_result
            research_data['audience'] = audience_result
            research_data['analogies'] = analogy_result
            
            # Add additional citations from research components
            all_sources = []
            for challenge in industry_result.get('challenges', []):
                all_sources.extend(challenge.get('sources', []))
                
            for argument in solution_result.get('pro_arguments', []) + solution_result.get('counter_arguments', []):
                all_sources.extend(argument.get('sources', []))
                
            for paradigm in paradigm_result.get('historical_paradigms', []):
                all_sources.extend(paradigm.get('sources', []))
                
            for segment in audience_result.get('audience_segments', []):
                all_sources.extend(segment.get('sources', []))
            
            # Convert sources to citation format and add to research data
            for source in all_sources:
                citation = {
                    'title': source.get('title'),
                    'url': source.get('url'),
                    'description': source.get('description'),
                    'source': source.get('source', 'Unknown Source'),
                    'date': datetime.now(timezone.utc).isoformat()
                }
                if citation not in research_data['citations']:
                    research_data['citations'].append(citation)
            
        except Exception as e:
            logger.error(f"Error gathering research data: {e}")
        
        return research_data
    
    def calculate_readiness_score(
        self, 
        metadata: Dict[str, Any], 
        research_data: Dict[str, Any]
    ) -> int:
        """
        Calculate readiness score based on metadata and research data.
        
        Args:
            metadata: Metadata extracted from content
            research_data: Research data gathered
            
        Returns:
            Readiness score (0-100)
        """
        score = 50  # Base score
        
        # Add points for citations
        citation_count = len(research_data.get('citations', []))
        if citation_count >= self.readiness_thresholds['min_citations']:
            score += 10
        
        # Add points for images
        image_count = metadata.get('images_count', 0)
        if image_count >= self.readiness_thresholds['min_images']:
            score += 5
        
        # Add points for headings
        heading_count = len(metadata.get('headings', []))
        if heading_count >= self.readiness_thresholds['min_headings']:
            score += 5
        
        # Add points for paragraphs
        paragraph_count = metadata.get('paragraphs_count', 0)
        if paragraph_count >= self.readiness_thresholds['min_paragraphs']:
            score += 5
        
        # Add points for additional content features
        if metadata.get('has_code_blocks'):
            score += 3
        
        if metadata.get('has_tables'):
            score += 3
        
        if metadata.get('has_lists'):
            score += 3
        
        # Add points for research components
        industry_challenges = research_data.get('challenges', [])
        if len(industry_challenges) >= self.readiness_thresholds['min_challenges']:
            score += 10
        
        solution_pro_args = research_data.get('solution', {}).get('pro_arguments', [])
        if len(solution_pro_args) >= self.readiness_thresholds['min_pro_arguments']:
            score += 5
        
        solution_counter_args = research_data.get('solution', {}).get('counter_arguments', [])
        if len(solution_counter_args) >= self.readiness_thresholds['min_counter_arguments']:
            score += 5
        
        # Count visual assets
        visual_assets_count = 0
        for analogy in research_data.get('analogies', {}).get('generated_analogies', []):
            visual_assets_count += len(analogy.get('visual', {}).get('assets', []))
        
        if visual_assets_count >= self.readiness_thresholds['min_visual_assets']:
            score += 10
        
        # Add points for analogies
        analogy_count = len(research_data.get('analogies', {}).get('generated_analogies', []))
        if analogy_count >= self.readiness_thresholds['min_analogies']:
            score += 5
        
        # Cap the score at 100
        return min(score, 100)
    
    def generate_research_report(
        self, 
        blog_data: Dict[str, Any],
        metadata: Dict[str, Any],
        research_data: Dict[str, Any],
        readiness_score: int
    ) -> str:
        """
        Generate a markdown research report.
        
        Args:
            blog_data: Blog data
            metadata: Metadata extracted from content
            research_data: Research data gathered
            readiness_score: Calculated readiness score
            
        Returns:
            Markdown research report
        """
        blog_title = metadata.get('blog_title', 'Unknown Title')
        main_topic = metadata.get('main_topic', 'Unknown Topic')
        
        report = f"""# Research Report: {main_topic}

## Blog Information
- **Title:** {blog_title}
- **Version:** {metadata.get('version', 1)}
- **Readiness Score:** {readiness_score}/100

## Topic Analysis
- **Main Topic:** {main_topic}
- **Summary:** {metadata.get('summary', 'No summary available')}
- **Reading Time:** {metadata.get('reading_time_minutes', 0)} minutes

## Content Structure
- **Headings:** {len(metadata.get('headings', []))}
- **Paragraphs:** {metadata.get('paragraphs_count', 0)}
- **Images:** {metadata.get('images_count', 0)}
- **Has Code Blocks:** {'Yes' if metadata.get('has_code_blocks') else 'No'}
- **Has Tables:** {'Yes' if metadata.get('has_tables') else 'No'}
- **Has Lists:** {'Yes' if metadata.get('has_lists') else 'No'}

## Research Data
"""
        
        # Add industry/system challenges
        report += """
### Industry/System Challenges
"""
        challenges = research_data.get('challenges', [])
        for i, challenge in enumerate(challenges[:5], 1):  # Show top 5 challenges
            report += f"""
#### {i}. {challenge.get('name', 'Unknown Challenge')}
{challenge.get('description', 'No description available')}

**Key components:**
- Risk factors: {', '.join(challenge.get('components', {}).get('risk_factors', ['None identified'])[:3])}
- Inefficiencies: {', '.join(challenge.get('components', {}).get('inefficiency_factors', ['None identified'])[:3])}
"""
        
        if len(challenges) > 5:
            report += f"\n*Plus {len(challenges) - 5} more challenges identified*\n"
        
        # Add solution analysis
        report += """
### Solution Analysis
"""
        pro_arguments = research_data.get('solution', {}).get('pro_arguments', [])
        counter_arguments = research_data.get('solution', {}).get('counter_arguments', [])
        
        report += """
#### Supporting Arguments
"""
        for i, arg in enumerate(pro_arguments[:3], 1):  # Show top 3 pro arguments
            report += f"""
**{i}. {arg.get('name', 'Unknown Argument')}**
{arg.get('description', 'No description available')}
"""
        
        if len(pro_arguments) > 3:
            report += f"\n*Plus {len(pro_arguments) - 3} more supporting arguments identified*\n"
        
        report += """
#### Counter Arguments
"""
        for i, arg in enumerate(counter_arguments[:3], 1):  # Show top 3 counter arguments
            report += f"""
**{i}. {arg.get('name', 'Unknown Argument')}**
{arg.get('description', 'No description available')}
"""
        
        if len(counter_arguments) > 3:
            report += f"\n*Plus {len(counter_arguments) - 3} more counter arguments identified*\n"
        
        # Add paradigm analysis
        report += """
### Historical Paradigm Analysis
"""
        historical_paradigms = research_data.get('paradigms', {}).get('historical_paradigms', [])
        future_paradigms = research_data.get('paradigms', {}).get('future_paradigms', [])
        
        for i, paradigm in enumerate(historical_paradigms[:3], 1):  # Show top 3 historical paradigms
            report += f"""
**{i}. {paradigm.get('name', 'Unknown Paradigm')} ({paradigm.get('time_period', 'Unknown period')})**
{paradigm.get('description', 'No description available')}
"""
        
        if historical_paradigms:
            report += """
#### Future Paradigm Possibilities
"""
            for i, paradigm in enumerate(future_paradigms[:2], 1):  # Show top 2 future paradigms
                report += f"""
**{i}. {paradigm.get('name', 'Unknown Paradigm')}**
{paradigm.get('description', 'No description available')}
Estimated timeline: {paradigm.get('estimated_timeline', 'Unknown')}
"""
        
        # Add audience analysis
        report += """
### Audience Analysis
"""
        audience_segments = research_data.get('audience', {}).get('audience_segments', [])
        
        for i, segment in enumerate(audience_segments[:3], 1):  # Show top 3 audience segments
            report += f"""
**{i}. {segment.get('name', 'Unknown Segment')}**
{segment.get('description', 'No description available')}
- Knowledge level: {segment.get('knowledge_level', 'Unknown')}
- Key pain points: {', '.join(segment.get('pain_points', ['None identified'])[:3])}
"""
        
        # Add analogies
        report += """
### Powerful Analogies
"""
        analogies = research_data.get('analogies', {}).get('generated_analogies', [])
        
        for i, analogy in enumerate(analogies[:3], 1):  # Show top 3 analogies
            report += f"""
**{i}. {analogy.get('title', 'Unknown Analogy')}** (from {analogy.get('domain', 'Unknown domain')})
{analogy.get('description', 'No description available')}
"""
        
        # Add citations
        report += """
## Citations

The following sources were found during research:
"""
        
        # Add citations
        citations = research_data.get('citations', [])
        for i, citation in enumerate(citations[:10], 1):  # Show top 10 citations
            report += f"""
### {i}. {citation.get('title', 'Unknown Title')}
- **URL:** {citation.get('url', 'No URL')}
- **Source:** {citation.get('source', 'Unknown Source')}
- **Description:** {citation.get('description', 'No description available')}
"""
        
        if len(citations) > 10:
            report += f"\n*Plus {len(citations) - 10} more citations gathered*\n"
        
        # Add readiness assessment
        report += """
## Readiness Assessment

"""
        if readiness_score >= 80:
            report += "This blog post shows **excellent readiness** for review. It has sufficient citations, structure, and comprehensive research components to proceed to the review stage."
        elif readiness_score >= 60:
            report += "This blog post shows **good readiness** for review. Some improvements might enhance the overall quality, but it can proceed to the review stage."
        elif readiness_score >= 40:
            report += "This blog post shows **moderate readiness** for review. Consider expanding the research components before proceeding to the review stage."
        else:
            report += "This blog post shows **low readiness** for review. Significant improvements to content and research are recommended before proceeding to the review stage."
        
        return report
    
    def save_research_results(
        self, 
        blog_data: Dict[str, Any],
        metadata: Dict[str, Any],
        research_data: Dict[str, Any],
        readiness_score: int,
        report_markdown: str
    ) -> Dict[str, Any]:
        """
        Save research results to MongoDB.
        
        Args:
            blog_data: Blog data
            metadata: Metadata extracted from content
            research_data: Research data gathered
            readiness_score: Calculated readiness score
            report_markdown: Generated research report
            
        Returns:
            Dict with status and stored data information
        """
        blog_title = metadata.get('blog_title')
        version = metadata.get('version', 1)
        
        try:
            # Store the blog content
            blog_doc = {
                "title": blog_title,
                "current_version": version,
                "asset_folder": blog_data.get('asset_folder', ''),
                "versions": [
                    {
                        "version": version,
                        "file_path": "",  # Will be updated later
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "review_status": {
                            "factual_review": {"complete": False, "result_file": None},
                            "style_review": {"complete": False, "result_file": None},
                            "grammar_review": {"complete": False, "result_file": None},
                            "final_release": {"complete": False}
                        },
                        "readiness_score": readiness_score
                    }
                ]
            }
            
            # Store blog in MongoDB
            blog_id = self.db_client.db.blogs.update_one(
                {"title": blog_title},
                {"$set": {"current_version": version}, 
                 "$push": {"versions": blog_doc["versions"][0]}},
                upsert=True
            ).upserted_id
            
            if blog_id:
                logger.info(f"Created new blog entry with ID: {blog_id}")
            else:
                logger.info(f"Updated existing blog: {blog_title}")
            
            # Store the report
            report_filename = f"{blog_title}_research_report_v{version}.md"
            report_id = self.db_client.store_review_result(
                blog_title,
                version,
                "research",
                report_markdown,
                report_filename
            )
            logger.info(f"Stored research report with ID: {report_id}")
            
            # Store the research data
            research_data_id = self.db_client.db.research_data.update_one(
                {"blog_title": blog_title, "version": version},
                {"$set": {
                    "blog_title": blog_title,
                    "version": version,
                    "data": research_data,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }},
                upsert=True
            ).upserted_id
            
            if research_data_id:
                logger.info(f"Stored research data with ID: {research_data_id}")
            else:
                logger.info(f"Updated existing research data for {blog_title} v{version}")
            
            # Store the images
            image_ids = []
            for img_ref, img_data in blog_data.get('images', {}).items():
                image_id = self.db_client.store_media(
                    blog_title,
                    version,
                    img_data.get('format', 'unknown'),
                    "local",
                    None,  # URL is None for local images
                    img_data.get('base64'),
                    img_ref  # Use image reference as alt text
                )
                image_ids.append(image_id)
            
            logger.info(f"Stored {len(image_ids)} blog images")
            
            # Store visual assets from analogies
            visual_asset_ids = []
            for analogy in research_data.get('analogies', {}).get('generated_analogies', []):
                visual = analogy.get('visual', {})
                assets = visual.get('assets', [])
                
                for asset in assets:
                    asset_id = self.db_client.store_media(
                        blog_title,
                        version,
                        asset.get('format', 'unknown'),
                        "firecrawl",
                        asset.get('url'),
                        asset.get('base64'),
                        f"Visual for {analogy.get('title', 'analogy')}"
                    )
                    visual_asset_ids.append(asset_id)
            
            logger.info(f"Stored {len(visual_asset_ids)} visual assets")
            
            # Create YAML tracker
            yaml_path = create_tracker_yaml(blog_title, version)
            logger.info(f"Created YAML tracker: {yaml_path}")
            
            return {
                "status": "success",
                "blog_title": blog_title,
                "version": version,
                "readiness_score": readiness_score,
                "report_id": report_id,
                "image_ids": image_ids,
                "visual_asset_ids": visual_asset_ids,
                "yaml_path": yaml_path
            }
            
        except Exception as e:
            logger.error(f"Error saving research results: {e}")
            raise
    
    async def process_blog(self, file_path: str) -> Dict[str, Any]:
        """
        Process a blog from a file path.
        
        Args:
            file_path: Path to the blog file (markdown or ZIP)
            
        Returns:
            Dict with processing results
        """
        try:
            # Check if the file is a ZIP or markdown
            if file_path.lower().endswith('.zip'):
                # Process ZIP file
                blog_data = process_blog_upload(file_path)
            else:
                # Process markdown file directly
                blog_data = self.process_markdown_file(file_path)
            
            # Extract metadata
            metadata = self.extract_metadata(blog_data['content'])
            metadata['blog_title'] = blog_data.get('blog_title')
            metadata['version'] = blog_data.get('version')
            
            # Gather research data
            research_data = await self.gather_research(metadata)
            
            # Calculate readiness score
            readiness_score = self.calculate_readiness_score(metadata, research_data)
            
            # Generate research report
            report = self.generate_research_report(
                blog_data, metadata, research_data, readiness_score
            )
            
            # Save results to MongoDB
            result = self.save_research_results(
                blog_data, metadata, research_data, readiness_score, report
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing blog: {e}")
            raise


def main():
    """Main function to run the researcher agent from the command line."""
    parser = argparse.ArgumentParser(description='Blog Accelerator Researcher Agent')
    parser.add_argument('file_path', help='Path to markdown file or ZIP file containing blog content')
    parser.add_argument('--mongodb-uri', help='MongoDB connection URI')
    parser.add_argument('--brave-api-key', help='Brave Search API key')
    parser.add_argument('--openai-api-key', help='OpenAI API key')
    parser.add_argument('--groq-api-key', help='Groq API key')
    parser.add_argument('--firecrawl-server', help='Firecrawl MCP server URL')
    parser.add_argument('--opik-server', help='Opik MCP server URL')
    
    args = parser.parse_args()
    
    try:
        # Initialize the agent
        agent = ResearcherAgent(
            mongodb_uri=args.mongodb_uri,
            brave_api_key=args.brave_api_key,
            openai_api_key=args.openai_api_key,
            groq_api_key=args.groq_api_key,
            firecrawl_server=args.firecrawl_server,
            opik_server=args.opik_server
        )
        
        # Process the blog
        result = asyncio.run(agent.process_blog(args.file_path))
        
        # Output result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
