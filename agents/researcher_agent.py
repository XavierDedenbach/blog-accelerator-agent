"""
Researcher agent responsible for processing topic analysis and gathering research data.

This agent:
1. Parses topic metadata from new markdown
2. Parses embedded image references
3. Calls Brave MCP for citations
4. Stores outputs to review_files and media in MongoDB
5. Assigns readiness score
"""

import os
import re
import json
import logging
import argparse
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
        opik_server: Optional[str] = None
    ):
        """
        Initialize the researcher agent.
        
        Args:
            mongodb_uri: MongoDB connection URI
            brave_api_key: API key for Brave Search API
            opik_server: URL for Opik MCP server
        """
        # Initialize database connection
        self.db_client = MongoDBClient(uri=mongodb_uri)
        
        # Get API keys from environment if not provided
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        if not self.brave_api_key:
            logger.warning("Brave API key not provided. Citation gathering will be limited.")
            
        # Get Opik server from environment if not provided
        self.opik_server = opik_server or os.environ.get("OPIK_SERVER")
        
        # Initialize readiness score thresholds
        self.readiness_thresholds = {
            "min_citations": 3,
            "min_images": 1,
            "min_headings": 3,
            "min_paragraphs": 5
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
    
    def gather_research(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather research data for a topic.
        
        Args:
            metadata: Metadata extracted from content
            
        Returns:
            Research data including citations and analysis
        """
        research_data = {
            'citations': [],
            'system_affected': {},
            'current_paradigm': {},
            'proposed_solution': {},
            'audience_analysis': {}
        }
        
        # Generate search queries based on metadata
        queries = []
        main_topic = metadata.get('main_topic')
        
        if main_topic:
            queries.append(main_topic)
            queries.append(f"{main_topic} solution")
            queries.append(f"{main_topic} industry")
            queries.append(f"{main_topic} current state")
        
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
        
        # Basic analysis for each category
        if main_topic:
            # System affected analysis
            research_data['system_affected'] = {
                'name': main_topic,
                'description': f"System or industry affected by {main_topic}",
                'scale': 'Unknown'
            }
            
            # Current paradigm
            research_data['current_paradigm'] = {
                'name': f"Current approach to {main_topic}",
                'limitations': [
                    "Limitation 1 (placeholder)",
                    "Limitation 2 (placeholder)"
                ]
            }
            
            # Proposed solution
            research_data['proposed_solution'] = {
                'name': f"New approach to {main_topic}",
                'advantages': [
                    "Advantage 1 (placeholder)",
                    "Advantage 2 (placeholder)"
                ]
            }
            
            # Audience analysis
            research_data['audience_analysis'] = {
                'knowledge_level': 'moderate',
                'interests': ["Technology", "Innovation"],
                'background': f"Readers interested in {main_topic}"
            }
        
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
            score += 10
        
        # Add points for headings
        heading_count = len(metadata.get('headings', []))
        if heading_count >= self.readiness_thresholds['min_headings']:
            score += 10
        
        # Add points for paragraphs
        paragraph_count = metadata.get('paragraphs_count', 0)
        if paragraph_count >= self.readiness_thresholds['min_paragraphs']:
            score += 10
        
        # Add points for additional content features
        if metadata.get('has_code_blocks'):
            score += 5
        
        if metadata.get('has_tables'):
            score += 5
        
        if metadata.get('has_lists'):
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

### System/Industry Affected
- **Name:** {research_data.get('system_affected', {}).get('name', 'Unknown')}
- **Description:** {research_data.get('system_affected', {}).get('description', 'No description available')}
- **Scale:** {research_data.get('system_affected', {}).get('scale', 'Unknown')}

### Current Paradigm
- **Name:** {research_data.get('current_paradigm', {}).get('name', 'Unknown')}
- **Limitations:**
"""
        
        # Add limitations
        limitations = research_data.get('current_paradigm', {}).get('limitations', [])
        for limitation in limitations:
            report += f"  - {limitation}\n"
        
        report += """
### Proposed Solution
- **Name:** {0}
- **Advantages:**
""".format(research_data.get('proposed_solution', {}).get('name', 'Unknown'))
        
        # Add advantages
        advantages = research_data.get('proposed_solution', {}).get('advantages', [])
        for advantage in advantages:
            report += f"  - {advantage}\n"
        
        report += """
### Audience Analysis
- **Knowledge Level:** {0}
- **Background:** {1}
- **Interests:**
""".format(
            research_data.get('audience_analysis', {}).get('knowledge_level', 'Unknown'),
            research_data.get('audience_analysis', {}).get('background', 'Unknown')
        )
        
        # Add interests
        interests = research_data.get('audience_analysis', {}).get('interests', [])
        for interest in interests:
            report += f"  - {interest}\n"
        
        report += """
## Citations

The following sources were found during research:
"""
        
        # Add citations
        citations = research_data.get('citations', [])
        for i, citation in enumerate(citations, 1):
            report += f"""
### {i}. {citation.get('title', 'Unknown Title')}
- **URL:** {citation.get('url', 'No URL')}
- **Source:** {citation.get('source', 'Unknown Source')}
- **Description:** {citation.get('description', 'No description available')}
"""
        
        report += """
## Readiness Assessment

"""
        if readiness_score >= 80:
            report += "This blog post shows **excellent readiness** for review. It has sufficient citations, structure, and content to proceed to the review stage."
        elif readiness_score >= 60:
            report += "This blog post shows **good readiness** for review. Some improvements might enhance the overall quality, but it can proceed to the review stage."
        elif readiness_score >= 40:
            report += "This blog post shows **moderate readiness** for review. Consider adding more citations, images, or sections before proceeding to the review stage."
        else:
            report += "This blog post shows **low readiness** for review. Significant improvements are recommended before proceeding to the review stage."
        
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
            
            logger.info(f"Stored {len(image_ids)} images")
            
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
                "yaml_path": yaml_path
            }
            
        except Exception as e:
            logger.error(f"Error saving research results: {e}")
            raise
    
    def process_blog(self, file_path: str) -> Dict[str, Any]:
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
            research_data = self.gather_research(metadata)
            
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
    parser.add_argument('--opik-server', help='Opik MCP server URL')
    
    args = parser.parse_args()
    
    try:
        # Initialize the agent
        agent = ResearcherAgent(
            mongodb_uri=args.mongodb_uri,
            brave_api_key=args.brave_api_key,
            opik_server=args.opik_server
        )
        
        # Process the blog
        result = agent.process_blog(args.file_path)
        
        # Output result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
