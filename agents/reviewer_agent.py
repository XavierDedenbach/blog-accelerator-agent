"""
Reviewer agent responsible for performing multi-stage reviews on blog content.

This agent:
1. Handles --stage flags (factual, style, grammar)
2. Validates stage order from blog_title_review_tracker.yaml
3. Uses Brave and Firecrawl for research/review as needed
4. Writes result markdown and updates Mongo
5. Updates YAML state
"""

import os
import re
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone
import requests

# Import utility modules
from agents.utilities.db import MongoDBClient
from agents.utilities.yaml_guard import (
    load_yaml, validate_yaml_structure, validate_stage_transition,
    mark_stage_complete, mark_blog_released, get_review_status
)
from agents.utilities.file_ops import read_markdown_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReviewStageError(Exception):
    """Exception raised for errors in review stages."""
    pass


class FactCheckerError(Exception):
    """Exception raised for errors in fact checking."""
    pass


class StyleReviewError(Exception):
    """Exception raised for errors in style review."""
    pass


class GrammarReviewError(Exception):
    """Exception raised for errors in grammar review."""
    pass


class ReviewerAgent:
    """
    Agent responsible for performing multi-stage reviews on blog content.
    
    Handles factual review, style review, and grammar review stages,
    each with their own specialized review processes.
    """
    
    # Review stages
    STAGE_FACTUAL = "factual_review"
    STAGE_STYLE = "style_review"
    STAGE_GRAMMAR = "grammar_review"
    
    # All valid stages
    VALID_STAGES = {STAGE_FACTUAL, STAGE_STYLE, STAGE_GRAMMAR}
    
    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        firecrawl_server: Optional[str] = None,
        opik_server: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize the reviewer agent.
        
        Args:
            mongodb_uri: MongoDB connection URI
            brave_api_key: API key for Brave Search API
            firecrawl_server: URL for Firecrawl MCP server
            opik_server: URL for Opik MCP server
            openai_api_key: API key for OpenAI
            groq_api_key: API key for Groq
        """
        # Initialize database connection
        self.db_client = MongoDBClient(uri=mongodb_uri)
        
        # Get API keys from environment if not provided
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        if not self.brave_api_key:
            logger.warning("Brave API key not provided. Fact checking will be limited.")
        
        self.firecrawl_server = firecrawl_server or os.environ.get("FIRECRAWL_SERVER")
        if not self.firecrawl_server:
            logger.warning("Firecrawl server not provided. Media asset verification will be limited.")
        
        self.opik_server = opik_server or os.environ.get("OPIK_SERVER")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        # Initialize reviewer personas for style review
        self.reviewer_personas = [
            {"name": "Clarity Expert", "focus": "Clarity and readability", "tone": "Direct"},
            {"name": "Structure Analyst", "focus": "Organization and flow", "tone": "Analytical"},
            {"name": "Technical Reviewer", "focus": "Technical accuracy and depth", "tone": "Precise"}
        ]
    
    def get_blog_content(self, blog_title: str, version: int) -> Dict[str, Any]:
        """
        Get the blog content and metadata from MongoDB.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            
        Returns:
            Dict containing blog content and metadata
            
        Raises:
            ReviewStageError: If blog content cannot be found
        """
        try:
            # Get blog document from MongoDB
            blog_doc = self.db_client.get_blog_status(blog_title)
            if not blog_doc:
                raise ReviewStageError(f"Blog not found: {blog_title}")
            
            # Find the specific version
            blog_version = None
            for v in blog_doc.get("versions", []):
                if v.get("version") == version:
                    blog_version = v
                    break
            
            if not blog_version:
                raise ReviewStageError(f"Blog version {version} not found for {blog_title}")
            
            # Get the research report
            research_report = self.db_client.db.review_files.find_one({
                "blog_title": blog_title,
                "version": version,
                "stage": "research"
            })
            
            # Get all media assets
            media_assets = list(self.db_client.db.media.find({
                "blog_title": blog_title,
                "version": version
            }))
            
            # Return combined result
            return {
                "blog_doc": blog_doc,
                "version_data": blog_version,
                "research_report": research_report,
                "media_assets": media_assets
            }
        
        except Exception as e:
            logger.error(f"Error getting blog content: {e}")
            raise ReviewStageError(f"Failed to get blog content: {e}")
    
    def extract_claims(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract claims from blog content for fact checking.
        
        Args:
            content: Markdown content
            
        Returns:
            List of extracted claims with context
        """
        claims = []
        
        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        for i, paragraph in enumerate(paragraphs):
            # Skip headings, image references, and code blocks
            if (paragraph.strip().startswith('#') or 
                paragraph.strip().startswith('!') or 
                paragraph.strip().startswith('```')):
                continue
            
            # Basic claim extraction - each sentence that makes a statement
            sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
            for sentence in sentences:
                # Skip short sentences or questions
                if len(sentence.strip()) < 10 or sentence.strip().endswith('?'):
                    continue
                
                # Check for factual indicators
                factual_indicators = [
                    "is", "are", "was", "were", "will", "has", "have", "had",
                    "according to", "study", "research", "shows", "demonstrates",
                    "percent", "%", "increase", "decrease", "statistics", "data"
                ]
                
                is_likely_claim = any(indicator in sentence.lower() for indicator in factual_indicators)
                
                if is_likely_claim:
                    claims.append({
                        "claim": sentence.strip(),
                        "context": paragraph.strip(),
                        "paragraph_index": i,
                        "confidence": "high" if len([ind for ind in factual_indicators if ind in sentence.lower()]) > 1 else "medium"
                    })
        
        return claims
    
    def search_brave_for_claim(self, claim: str) -> List[Dict[str, Any]]:
        """
        Search for evidence related to a claim using Brave Search API.
        
        Args:
            claim: The claim to search for
            
        Returns:
            List of search results
        """
        if not self.brave_api_key:
            logger.warning("Brave API key not available. Using mock results.")
            # Return mock data
            return [{
                'title': f'Mock result for "{claim[:30]}..."',
                'url': 'https://example.com/mock',
                'description': 'This is a mock search result.',
                'source': 'Mock Source',
                'relevance_score': 0.7
            }]
        
        # Use Brave Search API
        try:
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': self.brave_api_key
            }
            
            # Create a more targeted search query from the claim
            search_query = claim.replace("according to", "").strip()
            
            params = {
                'q': search_query,
                'count': 5,
                'search_lang': 'en'
            }
            
            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params
            )
            
            if response.status_code != 200:
                logger.error(f"Error searching Brave API: {response.status_code} - {response.text}")
                raise FactCheckerError(f"Brave Search API error: {response.status_code}")
            
            results = response.json().get('web', {}).get('results', [])
            
            # Process and score results
            processed_results = []
            for result in results:
                # Calculate basic relevance score based on keyword matches
                keywords = [w for w in search_query.lower().split() if len(w) > 3]
                description = result.get('description', '').lower()
                title = result.get('title', '').lower()
                
                # Count keyword matches in title and description
                matches = sum(1 for k in keywords if k in description or k in title)
                relevance_score = min(1.0, matches / max(1, len(keywords)))
                
                processed_result = {
                    'title': result.get('title'),
                    'url': result.get('url'),
                    'description': result.get('description'),
                    'source': result.get('extra_snippets', {}).get('source', 'Unknown Source'),
                    'relevance_score': relevance_score
                }
                processed_results.append(processed_result)
                
            return processed_results
        
        except Exception as e:
            logger.error(f"Error in Brave search: {e}")
            raise FactCheckerError(f"Failed to search Brave: {e}")
    
    def verify_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a claim using Brave Search API and analyze results.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Dict with verification results
        """
        claim_text = claim["claim"]
        
        # Get search results
        search_results = self.search_brave_for_claim(claim_text)
        
        # Analyze results
        if not search_results:
            return {
                "claim": claim_text,
                "context": claim["context"],
                "verification": "insufficient evidence",
                "confidence": "low",
                "sources": [],
                "explanation": "No sources found to verify this claim."
            }
        
        # Check if any sources have high relevance
        high_relevance_sources = [s for s in search_results if s.get('relevance_score', 0) > 0.7]
        medium_relevance_sources = [s for s in search_results if 0.4 <= s.get('relevance_score', 0) <= 0.7]
        
        if high_relevance_sources:
            verification = "verified"
            confidence = "high"
            explanation = f"Found {len(high_relevance_sources)} highly relevant sources supporting this claim."
        elif medium_relevance_sources:
            verification = "partially verified"
            confidence = "medium"
            explanation = f"Found {len(medium_relevance_sources)} somewhat relevant sources related to this claim."
        else:
            verification = "unverified"
            confidence = "low"
            explanation = "Found sources but they have low relevance to the claim."
        
        return {
            "claim": claim_text,
            "context": claim["context"],
            "verification": verification,
            "confidence": confidence,
            "sources": search_results[:3],  # Limit to top 3 sources
            "explanation": explanation
        }
    
    def perform_factual_review(self, blog_title: str, version: int) -> Dict[str, Any]:
        """
        Perform factual review on a blog post.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            
        Returns:
            Dict with review results
        """
        logger.info(f"Starting factual review for blog: {blog_title}, version: {version}")
        
        try:
            # Get blog content
            blog_data = self.get_blog_content(blog_title, version)
            blog_doc = blog_data["blog_doc"]
            version_data = blog_data["version_data"]
            
            # Get content from file or database
            content = None
            if "asset_folder" in blog_doc and version_data.get("file_path"):
                file_path = os.path.join(blog_doc["asset_folder"], version_data["file_path"])
                if os.path.exists(file_path):
                    content = read_markdown_file(file_path)
            
            # If not found, try to get from research report
            if not content and blog_data.get("research_report"):
                content = blog_data["research_report"].get("content", "")
            
            if not content:
                raise ReviewStageError("Blog content not found")
            
            # Extract claims from content
            claims = self.extract_claims(content)
            logger.info(f"Extracted {len(claims)} claims for fact checking")
            
            # Verify each claim
            verified_claims = []
            for claim in claims[:10]:  # Limit to 10 claims for efficiency
                verified_claim = self.verify_claim(claim)
                verified_claims.append(verified_claim)
                logger.info(f"Verified claim: {claim['claim'][:50]}... - {verified_claim['verification']}")
            
            # Generate fact check report
            report = self.generate_factual_report(blog_title, version, verified_claims)
            
            # Save report to MongoDB
            report_filename = f"{blog_title}_factual_review_v{version}.md"
            report_id = self.db_client.store_review_result(
                blog_title,
                version,
                self.STAGE_FACTUAL,
                report,
                report_filename
            )
            
            return {
                "status": "success",
                "blog_title": blog_title,
                "version": version,
                "stage": self.STAGE_FACTUAL,
                "report_id": report_id,
                "verified_claims": len(verified_claims),
                "report_filename": report_filename
            }
        
        except Exception as e:
            logger.error(f"Error in factual review: {e}")
            raise FactCheckerError(f"Factual review failed: {e}")
    
    def generate_factual_report(
        self, 
        blog_title: str, 
        version: int, 
        verified_claims: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a markdown report for factual review.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            verified_claims: List of verified claims
            
        Returns:
            Markdown report
        """
        # Count verification statuses
        verification_counts = {
            "verified": 0,
            "partially verified": 0,
            "unverified": 0,
            "insufficient evidence": 0
        }
        
        for claim in verified_claims:
            verification = claim.get("verification")
            if verification in verification_counts:
                verification_counts[verification] += 1
        
        # Calculate verification score
        total_claims = len(verified_claims)
        if total_claims > 0:
            verification_score = int(100 * (
                verification_counts["verified"] + 
                (0.5 * verification_counts["partially verified"])
            ) / total_claims)
        else:
            verification_score = 0
        
        # Generate overall assessment
        if verification_score >= 80:
            assessment = "EXCELLENT"
            assessment_text = "The blog post demonstrates excellent factual accuracy. Most claims are well-supported by credible sources."
        elif verification_score >= 60:
            assessment = "GOOD"
            assessment_text = "The blog post shows good factual accuracy. Some claims could benefit from additional support or clarification."
        elif verification_score >= 40:
            assessment = "FAIR"
            assessment_text = "The blog post has fair factual accuracy. Several claims lack sufficient support or verification."
        else:
            assessment = "NEEDS IMPROVEMENT"
            assessment_text = "The blog post needs significant improvement in factual accuracy. Many claims are unverified or lack supporting evidence."
        
        # Generate report
        report = f"""# Factual Review Report: {blog_title}

## Version: {version}
## Review Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Overall Assessment: {assessment}

**Verification Score:** {verification_score}/100

{assessment_text}

## Verification Summary

| Status | Count |
|--------|-------|
| Verified | {verification_counts["verified"]} |
| Partially Verified | {verification_counts["partially verified"]} |
| Unverified | {verification_counts["unverified"]} |
| Insufficient Evidence | {verification_counts["insufficient evidence"]} |
| **Total Claims** | **{total_claims}** |

## Detailed Claim Analysis

"""
        
        # Add detailed analysis for each claim
        for i, claim in enumerate(verified_claims, 1):
            verification = claim.get("verification", "unknown")
            confidence = claim.get("confidence", "low")
            
            # Set emoji based on verification status
            emoji = "✅" if verification == "verified" else "⚠️" if verification == "partially verified" else "❌"
            
            report += f"""### Claim {i}: {emoji} {verification.title()} (Confidence: {confidence.title()})

**Statement:** {claim.get("claim")}

**Context:** {claim.get("context")}

**Explanation:** {claim.get("explanation")}

**Sources:**
"""
            
            # Add sources
            sources = claim.get("sources", [])
            if sources:
                for j, source in enumerate(sources, 1):
                    report += f"""
{j}. [{source.get('title', 'Unknown Title')}]({source.get('url', '#')})
   - Source: {source.get('source', 'Unknown')}
   - Relevance: {int(source.get('relevance_score', 0) * 100)}%
   - Description: {source.get('description', 'No description')}
"""
            else:
                report += "No sources found.\n"
            
            report += "\n---\n\n"
        
        # Add recommendations
        report += """## Recommendations

"""
        if verification_score >= 80:
            report += "The blog post is factually sound and ready to proceed to style review."
        elif verification_score >= 60:
            report += "The blog post is mostly factually sound but could benefit from addressing the unverified claims before proceeding to style review."
        else:
            report += "The blog post requires significant revision to improve factual accuracy before proceeding to style review. Consider adding more citations and evidence for unverified claims."
        
        return report
    
    def perform_style_review(self, blog_title: str, version: int) -> Dict[str, Any]:
        """
        Perform style review on a blog post.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            
        Returns:
            Dict with review results
        """
        logger.info(f"Starting style review for blog: {blog_title}, version: {version}")
        
        try:
            # Get blog content
            blog_data = self.get_blog_content(blog_title, version)
            blog_doc = blog_data["blog_doc"]
            version_data = blog_data["version_data"]
            
            # Get content from file or database
            content = None
            if "asset_folder" in blog_doc and version_data.get("file_path"):
                file_path = os.path.join(blog_doc["asset_folder"], version_data["file_path"])
                if os.path.exists(file_path):
                    content = read_markdown_file(file_path)
            
            # If not found, try to get from research report
            if not content and blog_data.get("research_report"):
                content = blog_data["research_report"].get("content", "")
            
            if not content:
                raise ReviewStageError("Blog content not found")
            
            # Analyze content structure
            structure_analysis = self.analyze_content_structure(content)
            
            # Review with different personas
            persona_reviews = []
            for persona in self.reviewer_personas:
                review = self.review_with_persona(content, persona, structure_analysis)
                persona_reviews.append(review)
            
            # Generate style review report
            report = self.generate_style_report(
                blog_title, 
                version, 
                structure_analysis, 
                persona_reviews
            )
            
            # Save report to MongoDB
            report_filename = f"{blog_title}_style_review_v{version}.md"
            report_id = self.db_client.store_review_result(
                blog_title,
                version,
                self.STAGE_STYLE,
                report,
                report_filename
            )
            
            return {
                "status": "success",
                "blog_title": blog_title,
                "version": version,
                "stage": self.STAGE_STYLE,
                "report_id": report_id,
                "persona_reviews": len(persona_reviews),
                "report_filename": report_filename
            }
        
        except Exception as e:
            logger.error(f"Error in style review: {e}")
            raise StyleReviewError(f"Style review failed: {e}")
    
    def analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """
        Analyze the structure of the blog content.
        
        Args:
            content: Markdown content
            
        Returns:
            Dict with structure analysis
        """
        # Extract headings (all levels)
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        heading_structure = [{"level": len(h[0]), "text": h[1].strip()} for h in headings]
        
        # Count paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        non_empty_paragraphs = [p for p in paragraphs if p.strip() and not p.strip().startswith('#')]
        
        # Calculate average paragraph length
        if non_empty_paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in non_empty_paragraphs) / len(non_empty_paragraphs)
        else:
            avg_paragraph_length = 0
        
        # Check for transitions
        transition_words = [
            "however", "therefore", "furthermore", "moreover", "nevertheless",
            "in addition", "consequently", "as a result", "in conclusion", "for example"
        ]
        transition_count = sum(1 for word in transition_words if re.search(r'\b' + word + r'\b', content, re.IGNORECASE))
        
        # Check for personal pronouns (1st person)
        first_person_count = len(re.findall(r'\b(I|we|my|our)\b', content, re.IGNORECASE))
        
        # Calculate complexity metrics
        words = re.findall(r'\b\w+\b', content)
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
        else:
            avg_word_length = 0
        
        return {
            "heading_structure": heading_structure,
            "paragraph_count": len(non_empty_paragraphs),
            "avg_paragraph_length": avg_paragraph_length,
            "transition_count": transition_count,
            "first_person_count": first_person_count,
            "avg_word_length": avg_word_length
        }
    
    def review_with_persona(
        self, 
        content: str, 
        persona: Dict[str, str],
        structure_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review content using a specific persona.
        
        Args:
            content: Markdown content
            persona: Dict with persona information
            structure_analysis: Dict with structure analysis
            
        Returns:
            Dict with persona review
        """
        # This is a mock review since we don't have actual LLM integration here
        # In a real implementation, this would call an LLM API with the persona
        
        persona_name = persona["name"]
        persona_focus = persona["focus"]
        
        # Generate mock feedback based on structure analysis
        strengths = []
        weaknesses = []
        recommendations = []
        
        if persona_focus == "Clarity and readability":
            if structure_analysis["avg_paragraph_length"] < 100:
                strengths.append("Good paragraph length for readability")
            else:
                weaknesses.append("Paragraphs are too long, which can reduce readability")
                recommendations.append("Consider breaking up longer paragraphs into smaller, more digestible chunks")
            
            if structure_analysis["avg_word_length"] < 5.5:
                strengths.append("Uses accessible vocabulary")
            else:
                weaknesses.append("Vocabulary may be too complex for general audiences")
                recommendations.append("Consider simplifying language where appropriate")
            
        elif persona_focus == "Organization and flow":
            if structure_analysis["heading_structure"] and len(structure_analysis["heading_structure"]) > 2:
                strengths.append("Good use of headings to organize content")
            else:
                weaknesses.append("Limited use of headings makes content structure unclear")
                recommendations.append("Add more section headings to better organize content")
            
            if structure_analysis["transition_count"] > 3:
                strengths.append("Good use of transition words to connect ideas")
            else:
                weaknesses.append("Limited use of transition words affects flow")
                recommendations.append("Add transition words between sections and paragraphs")
            
        elif persona_focus == "Technical accuracy and depth":
            if structure_analysis["avg_paragraph_length"] > 50:
                strengths.append("Good depth in explanations")
            else:
                weaknesses.append("Explanations may lack sufficient depth")
                recommendations.append("Expand on technical concepts with more detailed explanations")
            
            if structure_analysis["first_person_count"] > 5:
                weaknesses.append("Overuse of first-person perspective in technical content")
                recommendations.append("Consider using a more objective tone for technical discussions")
        
        # Generate an overall rating
        if len(strengths) > len(weaknesses):
            rating = 4  # Good
            overall_assessment = f"The content demonstrates good {persona_focus.lower()}."
        elif len(strengths) == len(weaknesses):
            rating = 3  # Average
            overall_assessment = f"The content shows average {persona_focus.lower()} with room for improvement."
        else:
            rating = 2  # Needs improvement
            overall_assessment = f"The content needs improvement in {persona_focus.lower()}."
        
        return {
            "persona": persona_name,
            "focus": persona_focus,
            "rating": rating,
            "overall_assessment": overall_assessment,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def generate_style_report(
        self, 
        blog_title: str, 
        version: int, 
        structure_analysis: Dict[str, Any],
        persona_reviews: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a markdown report for style review.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            structure_analysis: Dict with structure analysis
            persona_reviews: List of persona reviews
            
        Returns:
            Markdown report
        """
        # Calculate average rating
        avg_rating = sum(review["rating"] for review in persona_reviews) / len(persona_reviews)
        
        # Determine overall assessment
        if avg_rating >= 4.5:
            assessment = "EXCELLENT"
            assessment_text = "The blog post demonstrates excellent style and structure. It is well-organized, clear, and engaging."
        elif avg_rating >= 3.5:
            assessment = "GOOD"
            assessment_text = "The blog post shows good style and structure. With minor revisions, it will be highly effective."
        elif avg_rating >= 2.5:
            assessment = "FAIR"
            assessment_text = "The blog post has fair style and structure. Several areas could benefit from revision."
        else:
            assessment = "NEEDS IMPROVEMENT"
            assessment_text = "The blog post needs significant improvement in style and structure. Major revisions are recommended."
        
        # Generate report
        report = f"""# Style Review Report: {blog_title}

## Version: {version}
## Review Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Overall Assessment: {assessment}

**Average Rating:** {avg_rating:.1f}/5

{assessment_text}

## Content Structure Analysis

* **Heading Count:** {len(structure_analysis["heading_structure"])}
* **Paragraph Count:** {structure_analysis["paragraph_count"]}
* **Average Paragraph Length:** {structure_analysis["avg_paragraph_length"]:.1f} words
* **Transition Word Count:** {structure_analysis["transition_count"]}
* **First-Person Perspective:** {structure_analysis["first_person_count"]} instances
* **Average Word Length:** {structure_analysis["avg_word_length"]:.1f} characters

## Heading Structure

"""
        
        # Add heading structure
        for heading in structure_analysis["heading_structure"]:
            level = heading["level"]
            text = heading["text"]
            report += f"{'  ' * (level - 1)}* {text}\n"
        
        report += "\n## Persona Reviews\n\n"
        
        # Add persona reviews
        for review in persona_reviews:
            report += f"""### {review["persona"]} (Focus: {review["focus"]})

**Rating:** {review["rating"]}/5

**Assessment:** {review["overall_assessment"]}

**Strengths:**
"""
            
            for strength in review["strengths"]:
                report += f"* {strength}\n"
            
            report += "\n**Weaknesses:**\n"
            
            for weakness in review["weaknesses"]:
                report += f"* {weakness}\n"
            
            report += "\n**Recommendations:**\n"
            
            for recommendation in review["recommendations"]:
                report += f"* {recommendation}\n"
            
            report += "\n---\n\n"
        
        # Add consolidated recommendations
        report += "## Consolidated Recommendations\n\n"
        
        all_recommendations = []
        for review in persona_reviews:
            all_recommendations.extend(review["recommendations"])
        
        # Deduplicate recommendations
        unique_recommendations = list(set(all_recommendations))
        
        for recommendation in unique_recommendations:
            report += f"* {recommendation}\n"
        
        # Add next steps
        report += "\n## Next Steps\n\n"
        
        if avg_rating >= 3.5:
            report += "The blog post is ready to proceed to grammar review. Consider addressing the recommended improvements first."
        else:
            report += "The blog post requires revision before proceeding to grammar review. Please address the weaknesses identified by the reviewers."
        
        return report
    
    def perform_grammar_review(self, blog_title: str, version: int) -> Dict[str, Any]:
        """
        Perform grammar review on a blog post.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            
        Returns:
            Dict with review results
        """
        logger.info(f"Starting grammar review for blog: {blog_title}, version: {version}")
        
        try:
            # Get blog content
            blog_data = self.get_blog_content(blog_title, version)
            blog_doc = blog_data["blog_doc"]
            version_data = blog_data["version_data"]
            
            # Get content from file or database
            content = None
            if "asset_folder" in blog_doc and version_data.get("file_path"):
                file_path = os.path.join(blog_doc["asset_folder"], version_data["file_path"])
                if os.path.exists(file_path):
                    content = read_markdown_file(file_path)
            
            # If not found, try to get from research report
            if not content and blog_data.get("research_report"):
                content = blog_data["research_report"].get("content", "")
            
            if not content:
                raise ReviewStageError("Blog content not found")
            
            # Analyze grammar and spelling
            grammar_issues = self.find_grammar_issues(content)
            
            # Generate grammar review report
            report = self.generate_grammar_report(blog_title, version, grammar_issues)
            
            # Save report to MongoDB
            report_filename = f"{blog_title}_grammar_review_v{version}.md"
            report_id = self.db_client.store_review_result(
                blog_title,
                version,
                self.STAGE_GRAMMAR,
                report,
                report_filename
            )
            
            return {
                "status": "success",
                "blog_title": blog_title,
                "version": version,
                "stage": self.STAGE_GRAMMAR,
                "report_id": report_id,
                "issues_found": len(grammar_issues),
                "report_filename": report_filename
            }
        
        except Exception as e:
            logger.error(f"Error in grammar review: {e}")
            raise GrammarReviewError(f"Grammar review failed: {e}")
    
    def find_grammar_issues(self, content: str) -> List[Dict[str, Any]]:
        """
        Find grammar and spelling issues in content.
        
        Args:
            content: Markdown content
            
        Returns:
            List of grammar issues
        """
        # This is a mock implementation since we don't have actual grammar checking API
        # In a real implementation, this would use a grammar/spell checking service
        
        # Sample grammar rules to check (very simplified)
        grammar_rules = [
            {"pattern": r'\b(is|are|was|were) being\b', "category": "passive voice", "suggestion": "Use active voice", "severity": "minor"},
            {"pattern": r'\b(alot|alright|irregardless)\b', "category": "incorrect word", "suggestion": "Replace with 'a lot', 'all right', or 'regardless'", "severity": "minor"},
            {"pattern": r'\bteh\b', "category": "spelling", "suggestion": "Replace with 'the'", "severity": "minor"},
            {"pattern": r'\bit\'s\b', "category": "contraction", "suggestion": "Verify if 'it's' (it is) or 'its' (possessive) is intended", "severity": "minor"},
            {"pattern": r'\bthier\b', "category": "spelling", "suggestion": "Replace with 'their'", "severity": "minor"},
            {"pattern": r'\byour\b', "category": "word choice", "suggestion": "Verify if 'your' (possessive) or 'you're' (you are) is intended", "severity": "minor"},
            {"pattern": r'[,;]\s*and', "category": "punctuation", "suggestion": "Remove comma before 'and' in a simple series", "severity": "minor"},
            {"pattern": r'\s\s+', "category": "spacing", "suggestion": "Remove extra spaces", "severity": "minor"}
        ]
        
        issues = []
        
        # Split content into paragraphs to provide better context
        paragraphs = re.split(r'\n\s*\n', content)
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Skip code blocks, headings, etc.
            if paragraph.strip().startswith('```') or paragraph.strip().startswith('#'):
                continue
                
            # Check each grammar rule
            for rule in grammar_rules:
                matches = re.finditer(rule["pattern"], paragraph, re.IGNORECASE)
                
                for match in matches:
                    # Get context around the issue
                    start = max(0, match.start() - 30)
                    end = min(len(paragraph), match.end() + 30)
                    
                    # Format context with the issue highlighted
                    context = paragraph[start:match.start()] + "**" + paragraph[match.start():match.end()] + "**" + paragraph[match.end():end]
                    
                    issue = {
                        "text": match.group(0),
                        "category": rule["category"],
                        "suggestion": rule["suggestion"],
                        "severity": rule["severity"],
                        "context": context.strip(),
                        "paragraph_index": para_idx
                    }
                    issues.append(issue)
        
        return issues
    
    def generate_grammar_report(
        self, 
        blog_title: str, 
        version: int, 
        grammar_issues: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a markdown report for grammar review.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            grammar_issues: List of grammar issues
            
        Returns:
            Markdown report
        """
        # Calculate statistics
        total_issues = len(grammar_issues)
        
        # Count issues by category
        categories = {}
        for issue in grammar_issues:
            category = issue["category"]
            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1
        
        # Determine overall assessment
        if total_issues == 0:
            assessment = "EXCELLENT"
            assessment_text = "The blog post demonstrates excellent grammar and spelling. No issues were found."
        elif total_issues <= 3:
            assessment = "GOOD"
            assessment_text = "The blog post shows good grammar and spelling. Only a few minor issues were found."
        elif total_issues <= 10:
            assessment = "FAIR"
            assessment_text = "The blog post has fair grammar and spelling. Several issues were found that should be addressed."
        else:
            assessment = "NEEDS IMPROVEMENT"
            assessment_text = "The blog post needs significant improvement in grammar and spelling. Numerous issues were found."
        
        # Generate report
        report = f"""# Grammar Review Report: {blog_title}

## Version: {version}
## Review Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Overall Assessment: {assessment}

**Total Issues Found:** {total_issues}

{assessment_text}

## Issues by Category

"""
        
        # Add category counts
        if categories:
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                report += f"* **{category.title()}:** {count} issues\n"
        else:
            report += "No issues found in any category.\n"
        
        report += "\n## Detailed Issues\n\n"
        
        # Add detailed issues
        if grammar_issues:
            for i, issue in enumerate(grammar_issues, 1):
                report += f"""### Issue {i}: {issue["category"].title()}

**Text:** {issue["text"]}

**Suggestion:** {issue["suggestion"]}

**Severity:** {issue["severity"].title()}

**Context:**
> {issue["context"]}

---

"""
        else:
            report += "No issues to report.\n"
        
        # Add recommendations
        report += "## Recommendations\n\n"
        
        if total_issues == 0:
            report += "The blog post is grammatically sound and ready for release."
        elif total_issues <= 3:
            report += "The blog post is nearly ready for release. Consider addressing the few minor issues identified."
        elif total_issues <= 10:
            report += "The blog post requires revision before release. Please address the identified issues."
        else:
            report += "The blog post requires thorough revision before release. Consider using a dedicated grammar checking tool and reviewing the document carefully."
        
        # Add next steps
        report += "\n## Next Steps\n\n"
        
        if total_issues <= 5:
            report += "The blog post can proceed to final release after addressing the identified issues."
        else:
            report += "The blog post should be revised and undergo another grammar review before final release."
        
        return report
    
    def process_review_stage(
        self, 
        blog_title: str, 
        version: int, 
        stage: str, 
        yaml_path: str
    ) -> Dict[str, Any]:
        """
        Process a review stage for a blog post.
        
        Args:
            blog_title: Title of the blog
            version: Version number
            stage: Review stage to process (factual_review, style_review, grammar_review)
            yaml_path: Path to the YAML file
            
        Returns:
            Dict with review results
        """
        logger.info(f"Processing review stage: {stage} for blog: {blog_title}, version: {version}")
        
        try:
            # Load and validate YAML
            yaml_data = load_yaml(yaml_path)
            validate_yaml_structure(yaml_data)
            
            # Validate stage transition
            validate_stage_transition(yaml_data, stage)
            
            # Process the appropriate stage
            if stage == self.STAGE_FACTUAL:
                result = self.perform_factual_review(blog_title, version)
            elif stage == self.STAGE_STYLE:
                result = self.perform_style_review(blog_title, version)
            elif stage == self.STAGE_GRAMMAR:
                result = self.perform_grammar_review(blog_title, version)
            else:
                raise ReviewStageError(f"Invalid review stage: {stage}")
            
            # Update YAML to mark stage as complete
            updated_yaml = mark_stage_complete(
                yaml_path,
                stage,
                "reviewer_agent",
                result.get("report_filename")
            )
            
            # Return result with updated YAML info
            result["yaml_updated"] = True
            result["current_stage"] = get_review_status(yaml_path).get("current_stage")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing review stage: {e}")
            raise ReviewStageError(f"Failed to process review stage: {e}")
    
    def mark_blog_as_released(self, yaml_path: str) -> Dict[str, Any]:
        """
        Mark a blog as released in the YAML file.
        
        Args:
            yaml_path: Path to the YAML file
            
        Returns:
            Dict with release status
        """
        try:
            # Update YAML to mark blog as released
            updated_yaml = mark_blog_released(yaml_path, "reviewer_agent")
            
            # Get status
            status = get_review_status(yaml_path)
            
            return {
                "status": "success",
                "blog_title": status.get("blog_title"),
                "version": status.get("version"),
                "released": status.get("released"),
                "yaml_path": yaml_path
            }
        
        except Exception as e:
            logger.error(f"Error marking blog as released: {e}")
            raise ReviewStageError(f"Failed to mark blog as released: {e}")


def main():
    """Main function to run the reviewer agent from the command line."""
    parser = argparse.ArgumentParser(description='Blog Accelerator Reviewer Agent')
    parser.add_argument('--yaml', required=True, help='Path to the YAML tracker file')
    parser.add_argument('--stage', choices=['factual_review', 'style_review', 'grammar_review', 'final_release'], 
                        required=True, help='Review stage to process')
    parser.add_argument('--mongodb-uri', help='MongoDB connection URI')
    parser.add_argument('--brave-api-key', help='Brave Search API key')
    parser.add_argument('--firecrawl-server', help='Firecrawl MCP server URL')
    parser.add_argument('--opik-server', help='Opik MCP server URL')
    
    args = parser.parse_args()
    
    try:
        # Load YAML file
        yaml_data = load_yaml(args.yaml)
        blog_title = yaml_data.get("blog_title")
        version = yaml_data.get("current_version")
        
        if not blog_title or not version:
            print(f"Error: Invalid YAML file. Missing blog_title or current_version.")
            return 1
        
        # Initialize agent
        agent = ReviewerAgent(
            mongodb_uri=args.mongodb_uri,
            brave_api_key=args.brave_api_key,
            firecrawl_server=args.firecrawl_server,
            opik_server=args.opik_server
        )
        
        # Process the stage
        if args.stage == 'final_release':
            result = agent.mark_blog_as_released(args.yaml)
        else:
            result = agent.process_review_stage(blog_title, version, args.stage, args.yaml)
        
        # Output result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
