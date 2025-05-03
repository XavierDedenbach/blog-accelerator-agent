"""
Tests for the reviewer agent in reviewer_agent.py.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, ANY

from agents.reviewer_agent import (
    ReviewerAgent, ReviewStageError, FactCheckerError, 
    StyleReviewError, GrammarReviewError
)


@pytest.fixture
def mock_db_client():
    """Create a mock MongoDB client for testing."""
    with patch('agents.reviewer_agent.MongoDBClient') as mock_client:
        # Setup for the db collections
        mock_blogs = MagicMock()
        mock_review_files = MagicMock()
        mock_media = MagicMock()
        
        # Setup find and find_one return values
        mock_blogs.find_one.return_value = {
            "title": "test-blog",
            "current_version": 1,
            "asset_folder": "/path/to/assets",
            "versions": [
                {
                    "version": 1,
                    "file_path": "test-blog_v1.md",
                    "timestamp": "2024-08-15T12:00:00Z",
                    "review_status": {
                        "factual_review": {"complete": False, "result_file": None},
                        "style_review": {"complete": False, "result_file": None},
                        "grammar_review": {"complete": False, "result_file": None},
                        "final_release": {"complete": False}
                    },
                    "readiness_score": 75
                }
            ]
        }
        
        mock_review_files.find_one.return_value = {
            "blog_title": "test-blog",
            "version": 1,
            "stage": "research",
            "filename": "test-blog_research_report_v1.md",
            "content": "# Test Blog Content\n\nThis is a test blog post.",
            "timestamp": "2024-08-15T12:00:00Z"
        }
        
        mock_media.find.return_value = [
            {
                "blog_title": "test-blog",
                "version": 1,
                "type": "image",
                "source": "local",
                "url": None,
                "stored_base64": "test_base64_data",
                "alt_text": "Test Image"
            }
        ]
        
        # Setup db attribute and its collections attribute
        mock_db = MagicMock()
        mock_db.blogs = mock_blogs
        mock_db.review_files = mock_review_files
        mock_db.media = mock_media
        
        # Setup client instance
        client_instance = mock_client.return_value
        client_instance.db = mock_db
        
        # Setup get_blog_status method
        client_instance.get_blog_status.return_value = mock_blogs.find_one.return_value
        
        # Setup store_review_result method
        client_instance.store_review_result.return_value = "mock_report_id"
        
        yield client_instance


@pytest.fixture
def mock_yaml_functions():
    """Create mock YAML functions for testing."""
    with patch('agents.reviewer_agent.load_yaml') as mock_load:
        with patch('agents.reviewer_agent.validate_yaml_structure') as mock_validate_structure:
            with patch('agents.reviewer_agent.validate_stage_transition') as mock_validate_transition:
                with patch('agents.reviewer_agent.mark_stage_complete') as mock_mark_complete:
                    with patch('agents.reviewer_agent.mark_blog_released') as mock_mark_released:
                        with patch('agents.reviewer_agent.get_review_status') as mock_get_status:
                            # Setup mock return values
                            mock_load.return_value = {
                                "blog_title": "test-blog",
                                "current_version": 1,
                                "review_pipeline": {
                                    "factual_review": {
                                        "complete": False,
                                        "completed_by": None,
                                        "result_file": None,
                                        "timestamp": None
                                    },
                                    "style_review": {
                                        "complete": False,
                                        "completed_by": None,
                                        "result_file": None,
                                        "timestamp": None
                                    },
                                    "grammar_review": {
                                        "complete": False,
                                        "completed_by": None,
                                        "result_file": None,
                                        "timestamp": None
                                    }
                                },
                                "final_release": {
                                    "complete": False,
                                    "released_by": None,
                                    "timestamp": None
                                }
                            }
                            
                            mock_mark_complete.return_value = mock_load.return_value
                            mock_mark_released.return_value = mock_load.return_value
                            
                            mock_get_status.return_value = {
                                "blog_title": "test-blog",
                                "version": 1,
                                "current_stage": "style_review",
                                "all_stages_complete": False,
                                "released": False
                            }
                            
                            yield (mock_load, mock_validate_structure, mock_validate_transition, 
                                   mock_mark_complete, mock_mark_released, mock_get_status)


@pytest.fixture
def mock_requests():
    """Create a mock requests module for testing."""
    with patch('agents.reviewer_agent.requests') as mock_req:
        # Setup the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Mock Result 1",
                        "url": "https://example.com/1",
                        "description": "This is a mock result 1",
                        "extra_snippets": {"source": "Example Source 1"}
                    },
                    {
                        "title": "Mock Result 2",
                        "url": "https://example.com/2",
                        "description": "This is a mock result 2",
                        "extra_snippets": {"source": "Example Source 2"}
                    }
                ]
            }
        }
        mock_req.get.return_value = mock_response
        yield mock_req


@pytest.fixture
def mock_file_ops():
    """Mock the file operations module."""
    with patch('agents.reviewer_agent.read_markdown_file') as mock_read:
        # Setup mock return value
        mock_read.return_value = """# Test Blog Title

This is a sample blog post for testing the reviewer agent.

## Introduction

This is the introduction to the topic. According to recent studies, this is a claim that should be fact-checked.

## Main Section

This is the main content of the blog. The second test claim states that reviewer agents are effective.

![Test Image](images/test.png)

## Conclusion

This is the conclusion. The final claim suggests that this approach will revolutionize blog writing.
"""
        yield mock_read


def test_init_reviewer_agent():
    """Test initializing the reviewer agent."""
    # Test with default parameters
    agent = ReviewerAgent()
    assert agent.brave_api_key is None or isinstance(agent.brave_api_key, str)
    assert agent.firecrawl_server is None or isinstance(agent.firecrawl_server, str)
    assert agent.opik_server is None or isinstance(agent.opik_server, str)
    
    # Test with explicit parameters
    agent = ReviewerAgent(
        mongodb_uri="mongodb://testhost:27017",
        brave_api_key="test_brave_key",
        firecrawl_server="http://test-firecrawl:4000",
        opik_server="http://test-opik:7000"
    )
    assert agent.brave_api_key == "test_brave_key"
    assert agent.firecrawl_server == "http://test-firecrawl:4000"
    assert agent.opik_server == "http://test-opik:7000"
    
    # Check that reviewer personas are initialized
    assert len(agent.reviewer_personas) > 0
    assert "name" in agent.reviewer_personas[0]
    assert "focus" in agent.reviewer_personas[0]


def test_get_blog_content(mock_db_client):
    """Test getting blog content."""
    agent = ReviewerAgent()
    agent.db_client = mock_db_client
    
    result = agent.get_blog_content("test-blog", 1)
    
    # Verify the result contains expected keys
    assert "blog_doc" in result
    assert "version_data" in result
    assert "research_report" in result
    assert "media_assets" in result
    
    # Verify the content
    assert result["blog_doc"]["title"] == "test-blog"
    assert result["version_data"]["version"] == 1
    assert "content" in result["research_report"]
    assert len(result["media_assets"]) == 1


def test_extract_claims():
    """Test extracting claims from content."""
    agent = ReviewerAgent()
    
    content = """# Test Blog Title

## Introduction

According to recent studies, this is a claim that should be fact-checked.

## Main Section

This is the main content of the blog. The second test claim states that reviewer agents are effective.

Research shows that automated review processes can increase productivity by 30%.

## Conclusion

This is a question, not a claim?

This is the conclusion. The final claim suggests that this approach will revolutionize blog writing.
"""
    
    claims = agent.extract_claims(content)
    
    # Verify claims are extracted
    assert len(claims) > 0
    
    # Check that the first claim contains the expected text
    assert "According to recent studies" in claims[0]["claim"]
    
    # Check that a claim with "research shows" is included
    research_claims = [c for c in claims if "research shows" in c["claim"].lower()]
    assert len(research_claims) > 0
    assert "30%" in research_claims[0]["claim"]
    
    # Check that questions are not included as claims
    question_claims = [c for c in claims if "question" in c["claim"].lower()]
    assert len(question_claims) == 0


def test_search_brave_for_claim(mock_requests):
    """Test searching Brave for claim evidence."""
    agent = ReviewerAgent(brave_api_key="test_brave_key")
    
    claim = "Research shows that automated review processes can increase productivity."
    results = agent.search_brave_for_claim(claim)
    
    # Verify API call and results
    mock_requests.get.assert_called_once()
    assert len(results) == 2
    assert "title" in results[0]
    assert "url" in results[0]
    assert "description" in results[0]
    assert "source" in results[0]
    assert "relevance_score" in results[0]


def test_search_brave_without_api_key():
    """Test searching Brave without an API key."""
    agent = ReviewerAgent(brave_api_key=None)
    
    claim = "This is a test claim."
    results = agent.search_brave_for_claim(claim)
    
    # Verify mock results
    assert len(results) == 1
    assert "Mock result" in results[0]["title"]
    assert "test claim" in results[0]["title"]


def test_verify_claim(mock_requests):
    """Test verifying a claim."""
    agent = ReviewerAgent(brave_api_key="test_brave_key")
    
    claim = {
        "claim": "Research shows that automated review processes can increase productivity.",
        "context": "Research shows that automated review processes can increase productivity by 30%.",
        "paragraph_index": 3,
        "confidence": "high"
    }
    
    result = agent.verify_claim(claim)
    
    # Verify result structure
    assert "claim" in result
    assert "context" in result
    assert "verification" in result
    assert "confidence" in result
    assert "sources" in result
    assert "explanation" in result
    
    # Verify content
    assert result["claim"] == claim["claim"]
    assert result["context"] == claim["context"]
    assert result["verification"] in ["verified", "partially verified", "unverified", "insufficient evidence"]
    assert result["confidence"] in ["high", "medium", "low"]
    assert len(result["sources"]) > 0
    assert len(result["explanation"]) > 0


def test_perform_factual_review(mock_db_client, mock_file_ops, mock_requests):
    """Test performing factual review."""
    agent = ReviewerAgent(brave_api_key="test_brave_key")
    agent.db_client = mock_db_client
    
    # Mock extract_claims and verify_claim methods
    with patch.object(agent, 'extract_claims') as mock_extract:
        with patch.object(agent, 'verify_claim') as mock_verify:
            # Setup mock returns
            mock_extract.return_value = [
                {
                    "claim": "Test claim 1",
                    "context": "Test context 1",
                    "paragraph_index": 1,
                    "confidence": "high"
                },
                {
                    "claim": "Test claim 2",
                    "context": "Test context 2",
                    "paragraph_index": 2,
                    "confidence": "medium"
                }
            ]
            
            mock_verify.side_effect = [
                {
                    "claim": "Test claim 1",
                    "context": "Test context 1",
                    "verification": "verified",
                    "confidence": "high",
                    "sources": [{"title": "Source 1", "url": "https://example.com"}],
                    "explanation": "This claim is verified"
                },
                {
                    "claim": "Test claim 2",
                    "context": "Test context 2",
                    "verification": "partially verified",
                    "confidence": "medium",
                    "sources": [{"title": "Source 2", "url": "https://example.com"}],
                    "explanation": "This claim is partially verified"
                }
            ]
            
            result = agent.perform_factual_review("test-blog", 1)
            
            # Verify methods were called
            mock_extract.assert_called_once()
            assert mock_verify.call_count == 2
            
            # Verify MongoDB storage
            agent.db_client.store_review_result.assert_called_once()
            
            # Verify result
            assert result["status"] == "success"
            assert result["blog_title"] == "test-blog"
            assert result["version"] == 1
            assert result["stage"] == "factual_review"
            assert result["verified_claims"] == 2
            assert "report_id" in result
            assert "report_filename" in result


def test_generate_factual_report():
    """Test generating factual review report."""
    agent = ReviewerAgent()
    
    verified_claims = [
        {
            "claim": "Research shows that automated review processes can increase productivity.",
            "context": "Research shows that automated review processes can increase productivity by 30%.",
            "verification": "verified",
            "confidence": "high",
            "sources": [
                {
                    "title": "Automation Benefits Study", 
                    "url": "https://example.com/1",
                    "description": "Study on automation benefits",
                    "source": "Academic Journal",
                    "relevance_score": 0.9
                }
            ],
            "explanation": "This claim is well-supported by research."
        },
        {
            "claim": "This approach will revolutionize blog writing.",
            "context": "The final claim suggests that this approach will revolutionize blog writing.",
            "verification": "partially verified",
            "confidence": "medium",
            "sources": [
                {
                    "title": "Blog Writing Trends", 
                    "url": "https://example.com/2",
                    "description": "Analysis of blog writing trends",
                    "source": "Industry Blog",
                    "relevance_score": 0.6
                }
            ],
            "explanation": "Some evidence supports this claim, but it's partly opinion."
        }
    ]
    
    report = agent.generate_factual_report("test-blog", 1, verified_claims)
    
    # Check report structure
    assert "# Factual Review Report: test-blog" in report
    assert "## Version: 1" in report
    assert "Verification Score:" in report
    assert "Verification Summary" in report
    assert "Detailed Claim Analysis" in report
    
    # Check claim details
    assert "Research shows that automated review processes" in report
    assert "✅ Verified" in report
    assert "⚠️ Partially Verified" in report
    assert "Automation Benefits Study" in report
    assert "Blog Writing Trends" in report
    
    # Check recommendations
    assert "Recommendations" in report


def test_analyze_content_structure():
    """Test analyzing content structure."""
    agent = ReviewerAgent()
    
    content = """# Test Blog Title

This is the introduction paragraph.

## Section 1

This is the first section paragraph. However, we should consider the alternatives.

## Section 2

This is the second section paragraph. Furthermore, it includes transition words.

```python
def test_function():
    return "Hello, world!"
```

I think this approach is good, and we should implement it.

## Conclusion

In conclusion, this is the final paragraph.
"""
    
    result = agent.analyze_content_structure(content)
    
    # Verify result structure
    assert "heading_structure" in result
    assert "paragraph_count" in result
    assert "avg_paragraph_length" in result
    assert "transition_count" in result
    assert "first_person_count" in result
    assert "avg_word_length" in result
    
    # Verify content
    assert len(result["heading_structure"]) == 4  # Title + 3 sections
    assert result["paragraph_count"] > 0
    assert result["avg_paragraph_length"] > 0
    assert result["transition_count"] >= 2  # "However" and "Furthermore"
    assert result["first_person_count"] >= 1  # "I" and "we"
    assert result["avg_word_length"] > 0


def test_review_with_persona():
    """Test reviewing with a persona."""
    agent = ReviewerAgent()
    
    content = """# Test Blog Title

This is the introduction paragraph.

## Section 1

This is the first section paragraph. However, we should consider the alternatives.

## Section 2

This is the second section paragraph. Furthermore, it includes transition words.

## Conclusion

In conclusion, this is the final paragraph.
"""
    
    structure_analysis = agent.analyze_content_structure(content)
    
    persona = {
        "name": "Clarity Expert",
        "focus": "Clarity and readability",
        "tone": "Direct"
    }
    
    result = agent.review_with_persona(content, persona, structure_analysis)
    
    # Verify result structure
    assert "persona" in result
    assert "focus" in result
    assert "rating" in result
    assert "overall_assessment" in result
    assert "strengths" in result
    assert "weaknesses" in result
    assert "recommendations" in result
    
    # Verify content
    assert result["persona"] == "Clarity Expert"
    assert result["focus"] == "Clarity and readability"
    assert isinstance(result["rating"], int)
    assert 1 <= result["rating"] <= 5
    assert len(result["strengths"]) > 0 or len(result["weaknesses"]) > 0
    assert len(result["recommendations"]) >= 0


def test_perform_style_review(mock_db_client, mock_file_ops):
    """Test performing style review."""
    agent = ReviewerAgent()
    agent.db_client = mock_db_client
    
    # Mock analyze_content_structure and review_with_persona methods
    with patch.object(agent, 'analyze_content_structure') as mock_analyze:
        with patch.object(agent, 'review_with_persona') as mock_review:
            # Setup mock returns
            mock_analyze.return_value = {
                "heading_structure": [
                    {"level": 1, "text": "Test Blog Title"},
                    {"level": 2, "text": "Introduction"},
                    {"level": 2, "text": "Main Section"},
                    {"level": 2, "text": "Conclusion"}
                ],
                "paragraph_count": 5,
                "avg_paragraph_length": 25.5,
                "transition_count": 3,
                "first_person_count": 2,
                "avg_word_length": 4.5
            }
            
            mock_review.side_effect = [
                {
                    "persona": "Clarity Expert",
                    "focus": "Clarity and readability",
                    "rating": 4,
                    "overall_assessment": "The content demonstrates good clarity and readability.",
                    "strengths": ["Good paragraph length for readability"],
                    "weaknesses": [],
                    "recommendations": []
                },
                {
                    "persona": "Structure Analyst",
                    "focus": "Organization and flow",
                    "rating": 3,
                    "overall_assessment": "The content shows average organization and flow with room for improvement.",
                    "strengths": ["Good use of headings to organize content"],
                    "weaknesses": ["Limited use of transition words affects flow"],
                    "recommendations": ["Add transition words between sections and paragraphs"]
                },
                {
                    "persona": "Technical Reviewer",
                    "focus": "Technical accuracy and depth",
                    "rating": 4,
                    "overall_assessment": "The content demonstrates good technical accuracy and depth.",
                    "strengths": ["Good depth in explanations"],
                    "weaknesses": [],
                    "recommendations": []
                }
            ]
            
            result = agent.perform_style_review("test-blog", 1)
            
            # Verify methods were called
            mock_analyze.assert_called_once()
            assert mock_review.call_count == len(agent.reviewer_personas)
            
            # Verify MongoDB storage
            agent.db_client.store_review_result.assert_called_once()
            
            # Verify result
            assert result["status"] == "success"
            assert result["blog_title"] == "test-blog"
            assert result["version"] == 1
            assert result["stage"] == "style_review"
            assert result["persona_reviews"] == len(agent.reviewer_personas)
            assert "report_id" in result
            assert "report_filename" in result


def test_generate_style_report():
    """Test generating style review report."""
    agent = ReviewerAgent()
    
    structure_analysis = {
        "heading_structure": [
            {"level": 1, "text": "Test Blog Title"},
            {"level": 2, "text": "Introduction"},
            {"level": 2, "text": "Main Section"},
            {"level": 2, "text": "Conclusion"}
        ],
        "paragraph_count": 5,
        "avg_paragraph_length": 25.5,
        "transition_count": 3,
        "first_person_count": 2,
        "avg_word_length": 4.5
    }
    
    persona_reviews = [
        {
            "persona": "Clarity Expert",
            "focus": "Clarity and readability",
            "rating": 4,
            "overall_assessment": "The content demonstrates good clarity and readability.",
            "strengths": ["Good paragraph length for readability"],
            "weaknesses": [],
            "recommendations": []
        },
        {
            "persona": "Structure Analyst",
            "focus": "Organization and flow",
            "rating": 3,
            "overall_assessment": "The content shows average organization and flow with room for improvement.",
            "strengths": ["Good use of headings to organize content"],
            "weaknesses": ["Limited use of transition words affects flow"],
            "recommendations": ["Add transition words between sections and paragraphs"]
        }
    ]
    
    report = agent.generate_style_report("test-blog", 1, structure_analysis, persona_reviews)
    
    # Check report structure
    assert "# Style Review Report: test-blog" in report
    assert "## Version: 1" in report
    assert "Average Rating:" in report
    assert "Content Structure Analysis" in report
    assert "Heading Structure" in report
    assert "Persona Reviews" in report
    
    # Check structure details
    assert "Heading Count: 4" in report
    assert "Paragraph Count: 5" in report
    assert "Average Paragraph Length: 25.5" in report
    
    # Check persona review details
    assert "Clarity Expert (Focus: Clarity and readability)" in report
    assert "Rating: 4/5" in report
    assert "Good paragraph length for readability" in report
    assert "Structure Analyst (Focus: Organization and flow)" in report
    assert "Add transition words between sections and paragraphs" in report
    
    # Check recommendations
    assert "Consolidated Recommendations" in report
    assert "Next Steps" in report


def test_find_grammar_issues():
    """Test finding grammar issues."""
    agent = ReviewerAgent()
    
    content = """# Test Blog Title

This is teh introduction paragraph with a typo.

## Section 1

This is alot of text with a common mistake.

## Section 2

It's vs its is a common confusion. Thier spelling is incorrect too.

## Conclusion

This,  and that need better spacing.
"""
    
    issues = agent.find_grammar_issues(content)
    
    # Verify issues are found
    assert len(issues) > 0
    
    # Check for specific issues
    spelling_issues = [i for i in issues if i["category"] == "spelling"]
    assert len(spelling_issues) > 0
    
    # Check issue structure
    for issue in issues:
        assert "text" in issue
        assert "category" in issue
        assert "suggestion" in issue
        assert "severity" in issue
        assert "context" in issue
        assert "paragraph_index" in issue


def test_perform_grammar_review(mock_db_client, mock_file_ops):
    """Test performing grammar review."""
    agent = ReviewerAgent()
    agent.db_client = mock_db_client
    
    # Mock find_grammar_issues method
    with patch.object(agent, 'find_grammar_issues') as mock_find:
        # Setup mock return
        mock_find.return_value = [
            {
                "text": "teh",
                "category": "spelling",
                "suggestion": "Replace with 'the'",
                "severity": "minor",
                "context": "This is **teh** introduction paragraph",
                "paragraph_index": 1
            },
            {
                "text": "alot",
                "category": "incorrect word",
                "suggestion": "Replace with 'a lot'",
                "severity": "minor",
                "context": "This is **alot** of text with",
                "paragraph_index": 3
            }
        ]
        
        result = agent.perform_grammar_review("test-blog", 1)
        
        # Verify method was called
        mock_find.assert_called_once()
        
        # Verify MongoDB storage
        agent.db_client.store_review_result.assert_called_once()
        
        # Verify result
        assert result["status"] == "success"
        assert result["blog_title"] == "test-blog"
        assert result["version"] == 1
        assert result["stage"] == "grammar_review"
        assert result["issues_found"] == 2
        assert "report_id" in result
        assert "report_filename" in result


def test_generate_grammar_report():
    """Test generating grammar review report."""
    agent = ReviewerAgent()
    
    grammar_issues = [
        {
            "text": "teh",
            "category": "spelling",
            "suggestion": "Replace with 'the'",
            "severity": "minor",
            "context": "This is **teh** introduction paragraph",
            "paragraph_index": 1
        },
        {
            "text": "alot",
            "category": "incorrect word",
            "suggestion": "Replace with 'a lot'",
            "severity": "minor",
            "context": "This is **alot** of text with",
            "paragraph_index": 3
        },
        {
            "text": "thier",
            "category": "spelling",
            "suggestion": "Replace with 'their'",
            "severity": "minor",
            "context": "**Thier** spelling is incorrect",
            "paragraph_index": 5
        }
    ]
    
    report = agent.generate_grammar_report("test-blog", 1, grammar_issues)
    
    # Check report structure
    assert "# Grammar Review Report: test-blog" in report
    assert "## Version: 1" in report
    assert "Total Issues Found: 3" in report
    assert "Issues by Category" in report
    assert "Detailed Issues" in report
    
    # Check issue details
    assert "Spelling" in report
    assert "Incorrect Word" in report
    assert "Replace with 'the'" in report
    assert "Replace with 'a lot'" in report
    assert "Replace with 'their'" in report
    
    # Check recommendations
    assert "Recommendations" in report
    assert "Next Steps" in report


def test_process_review_stage(mock_db_client, mock_yaml_functions, mock_file_ops):
    """Test processing a review stage."""
    agent = ReviewerAgent()
    agent.db_client = mock_db_client
    
    # Mock stage-specific review methods
    with patch.object(agent, 'perform_factual_review') as mock_factual:
        with patch.object(agent, 'perform_style_review') as mock_style:
            with patch.object(agent, 'perform_grammar_review') as mock_grammar:
                # Setup mock returns
                mock_factual.return_value = {
                    "status": "success",
                    "blog_title": "test-blog",
                    "version": 1,
                    "stage": "factual_review",
                    "report_id": "mock_report_id",
                    "report_filename": "test-blog_factual_review_v1.md"
                }
                
                mock_style.return_value = {
                    "status": "success",
                    "blog_title": "test-blog",
                    "version": 1,
                    "stage": "style_review",
                    "report_id": "mock_report_id",
                    "report_filename": "test-blog_style_review_v1.md"
                }
                
                mock_grammar.return_value = {
                    "status": "success",
                    "blog_title": "test-blog",
                    "version": 1,
                    "stage": "grammar_review",
                    "report_id": "mock_report_id",
                    "report_filename": "test-blog_grammar_review_v1.md"
                }
                
                # Test factual review
                result_factual = agent.process_review_stage(
                    "test-blog", 1, "factual_review", "/path/to/yaml"
                )
                
                # Test style review
                result_style = agent.process_review_stage(
                    "test-blog", 1, "style_review", "/path/to/yaml"
                )
                
                # Test grammar review
                result_grammar = agent.process_review_stage(
                    "test-blog", 1, "grammar_review", "/path/to/yaml"
                )
                
                # Verify methods were called correctly
                mock_factual.assert_called_once_with("test-blog", 1)
                mock_style.assert_called_once_with("test-blog", 1)
                mock_grammar.assert_called_once_with("test-blog", 1)
                
                # Verify YAML functions were called
                mock_yaml_functions[0].assert_called()  # load_yaml
                mock_yaml_functions[1].assert_called()  # validate_yaml_structure
                mock_yaml_functions[2].assert_called()  # validate_stage_transition
                mock_yaml_functions[3].assert_called()  # mark_stage_complete
                
                # Verify results
                for result in [result_factual, result_style, result_grammar]:
                    assert result["status"] == "success"
                    assert "yaml_updated" in result
                    assert "current_stage" in result


def test_mark_blog_as_released(mock_yaml_functions):
    """Test marking a blog as released."""
    agent = ReviewerAgent()
    
    result = agent.mark_blog_as_released("/path/to/yaml")
    
    # Verify YAML functions were called
    mock_yaml_functions[4].assert_called_once_with("/path/to/yaml", "reviewer_agent")
    mock_yaml_functions[5].assert_called_once_with("/path/to/yaml")
    
    # Verify result
    assert result["status"] == "success"
    assert "blog_title" in result
    assert "version" in result
    assert "released" in result
    assert "yaml_path" in result


def test_main_function():
    """Test the main function of the reviewer agent."""
    with patch('sys.argv', ['reviewer_agent.py', '--yaml', '/path/to/yaml', '--stage', 'factual_review']):
        with patch('agents.reviewer_agent.load_yaml') as mock_load:
            with patch('agents.reviewer_agent.ReviewerAgent') as mock_agent_class:
                # Setup mock return values
                mock_load.return_value = {
                    "blog_title": "test-blog",
                    "current_version": 1
                }
                
                # Mock the agent instance
                mock_agent = MagicMock()
                mock_agent.process_review_stage.return_value = {"status": "success"}
                mock_agent_class.return_value = mock_agent
                
                # Mock print to capture output
                with patch('builtins.print') as mock_print:
                    # Call main function
                    from agents.reviewer_agent import main
                    exit_code = main()
                    
                    # Verify that the agent was created and process_review_stage was called
                    mock_agent_class.assert_called_once()
                    mock_agent.process_review_stage.assert_called_once_with(
                        "test-blog", 1, "factual_review", "/path/to/yaml"
                    )
                    
                    # Verify that the result was printed
                    mock_print.assert_called_once()
                    
                    # Verify exit code
                    assert exit_code == 0 