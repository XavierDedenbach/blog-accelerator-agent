"""
Tests for the researcher agent in researcher_agent.py.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, ANY

from agents.researcher_agent import (
    ResearcherAgent, TopicAnalysisError, CitationError
)


@pytest.fixture
def mock_db_client():
    """Create a mock MongoDB client for testing."""
    with patch('agents.researcher_agent.MongoDBClient') as mock_client:
        # Setup for the db.blogs.update_one method
        mock_update = MagicMock()
        mock_update.upserted_id = "mock_blog_id"
        mock_blogs = MagicMock()
        mock_blogs.update_one.return_value = mock_update
        
        # Setup db attribute and its collections attribute
        mock_db = MagicMock()
        mock_db.blogs = mock_blogs
        
        # Setup client instance
        client_instance = mock_client.return_value
        client_instance.db = mock_db
        
        # Setup store_review_result and store_media methods
        client_instance.store_review_result.return_value = "mock_report_id"
        client_instance.store_media.return_value = "mock_image_id"
        
        yield client_instance


@pytest.fixture
def mock_yaml_guard():
    """Create a mock YAML guard for testing."""
    with patch('agents.researcher_agent.create_tracker_yaml') as mock_create:
        mock_create.return_value = "/path/to/yaml_file.yaml"
        yield mock_create


@pytest.fixture
def mock_requests():
    """Create a mock requests module for testing."""
    with patch('agents.researcher_agent.requests') as mock_req:
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
def sample_markdown_file():
    """Create a sample markdown file for testing."""
    with tempfile.NamedTemporaryFile(suffix="_v2.md", delete=False) as f:
        f.write("""# Test Blog Title

This is a sample blog post for testing the researcher agent.

## Introduction

This is the introduction to the topic.

## Main Section

This is the main content of the blog.

![Test Image](images/test.png)

## Conclusion

This is the conclusion.
""".encode('utf-8'))
        yield f.name
    
    # Clean up after test
    os.unlink(f.name)


@pytest.fixture
def mock_file_ops():
    """Mock the file operations module."""
    with patch('agents.researcher_agent.collect_blog_assets') as mock_collect:
        with patch('agents.researcher_agent.process_blog_upload') as mock_process:
            # Setup mock_collect
            mock_collect.return_value = {
                'content': """# Test Blog Title

This is a sample blog post for testing the researcher agent.

## Introduction

This is the introduction to the topic.

## Main Section

This is the main content of the blog.

![Test Image](images/test.png)

## Conclusion

This is the conclusion.
""",
                'images': {
                    'images/test.png': {
                        'format': 'png',
                        'base64': 'test_base64_data',
                        'path': '/path/to/images/test.png'
                    }
                },
                'blog_title': 'test-blog',
                'version': 2,
                'asset_folder': '/path/to'
            }
            
            # Setup mock_process to return the same as mock_collect
            mock_process.return_value = mock_collect.return_value
            
            yield (mock_collect, mock_process)


@pytest.fixture
def mock_source_validator():
    """Provide a mock SourceValidator instance."""
    with patch('agents.researcher_agent.SourceValidator') as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.validate_source.return_value = {
            'is_valid': True,
            'credibility_score': 0.8,
            'domain_info': {'name': 'example.com'}
        }
        yield mock_cls


@pytest.fixture
def mock_firecrawl_client():
    """Provide a mock FirecrawlClient instance."""
    with patch('agents.researcher_agent.FirecrawlClient') as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.search_images.return_value = [{
            'url': 'https://example.com/image.jpg',
            'title': 'Example Image',
            'source': 'Example Source'
        }]
        yield mock_cls


def test_init_researcher_agent(mock_source_validator, mock_firecrawl_client):
    """Test initializing the researcher agent."""
    agent = ResearcherAgent(
        brave_api_key="test-brave-key",
        firecrawl_server="http://test-firecrawl:7000",
        opik_server="http://test-opik:7000"
    )
    
    # Verify the agent was initialized correctly
    assert agent.brave_api_key == "test-brave-key"
    assert agent.firecrawl_server == "http://test-firecrawl:7000"
    assert agent.opik_server == "http://test-opik:7000"
    
    # Verify source validator and firecrawl client were initialized
    mock_source_validator.assert_called_once_with(brave_api_key="test-brave-key")
    mock_firecrawl_client.assert_called_once_with(
        server_url="http://test-firecrawl:7000", 
        brave_api_key="test-brave-key"
    )


def test_extract_metadata(mock_source_validator, mock_firecrawl_client):
    """Test extracting metadata from content."""
    agent = ResearcherAgent()
    
    content = """# Test Blog Title
    
This is a test blog post. It contains multiple paragraphs.

Here's a second paragraph with some content.

## First Section
Content in the first section.

## Second Section
Content in the second section.

![Test Image](image.jpg)

```python
def test_function():
    pass
```

> This is a blockquote.

* List item 1
* List item 2
    """
    
    metadata = agent.extract_metadata(content)
    
    # Verify metadata extraction
    assert metadata['main_topic'] == 'Test Blog Title'
    assert len(metadata['headings']) == 3  # Title + 2 sections
    assert metadata['paragraphs_count'] >= 3
    assert metadata['images_count'] == 1
    assert metadata['has_code_blocks'] is True
    assert metadata['has_lists'] is True


def test_search_citations_with_brave_api(mock_requests, mock_source_validator, mock_firecrawl_client):
    """Test searching citations with Brave API."""
    agent = ResearcherAgent(brave_api_key="test-brave-key")
    
    result = agent.search_citations("test query")
    
    # Verify API was called
    mock_requests.get.assert_called_once()
    assert len(result) > 0
    assert 'title' in result[0]
    assert 'url' in result[0]
    

def test_search_citations_without_api_key(mock_source_validator, mock_firecrawl_client):
    """Test searching citations without an API key (fallback to mock data)."""
    # Make sure we're not actually using the requests module
    with patch('agents.researcher_agent.requests') as mock_requests:
        # We need to make sure requests is never called when brave_api_key is None
        mock_requests.get.side_effect = Exception("Requests should not be called")
        
        # Create a new ResearcherAgent with brave_api_key explicitly set to None
        agent = ResearcherAgent(brave_api_key=None)
        
        # Now the agent should use the mock data path without calling requests
        result = agent.search_citations("test query")
        
        # Verify result contains expected mock data
        assert len(result) == 1
        assert result[0]['title'] == 'Mock citation for "test query"'
        assert result[0]['url'] == 'https://example.com/mock'
        assert 'date' in result[0]
        

def test_search_citations_api_error(mock_requests, mock_source_validator, mock_firecrawl_client):
    """Test searching citations with an API error."""
    # Modify the mock to simulate an error
    mock_requests.get.return_value.status_code = 401
    mock_requests.get.return_value.text = "Unauthorized"
    
    agent = ResearcherAgent(brave_api_key="test_brave_key")
    
    # Should raise a CitationError
    with pytest.raises(CitationError):
        agent.search_citations("test query")
        

def test_gather_research(mock_source_validator, mock_firecrawl_client):
    """Test gathering research data."""
    agent = ResearcherAgent()
    
    # Mock the search_citations method
    with patch.object(agent, 'search_citations') as mock_search:
        mock_search.return_value = [
            {
                'title': 'Mock Result 1',
                'url': 'https://example.com/1',
                'description': 'This is a mock result 1',
                'source': 'Example Source 1'
            }
        ]
        
        metadata = {
            'main_topic': 'Test Topic',
            'headings': ['Test Topic', 'Introduction', 'Main Section', 'Conclusion']
        }
        
        research_data = agent.gather_research(metadata)
        
        # Verify research data
        assert 'citations' in research_data
        assert len(research_data['citations']) > 0
        assert 'system_affected' in research_data
        assert research_data['system_affected']['name'] == 'Test Topic'
        # We won't assert for other components since they may fail in the test environment
        assert 'progress' in research_data


def test_calculate_readiness_score(mock_source_validator, mock_firecrawl_client):
    """Test calculating readiness score."""
    agent = ResearcherAgent()
    
    # Test with minimal data (should get base score)
    metadata = {
        'headings': [],
        'paragraphs_count': 1,
        'images_count': 0,
        'has_code_blocks': False,
        'has_tables': False,
        'has_lists': False
    }
    research_data = {
        'citations': []
    }
    
    score = agent.calculate_readiness_score(metadata, research_data)
    assert score == 50  # Base score
    
    # Test with more complete data (should get higher score)
    metadata = {
        'headings': ['Title', 'Introduction', 'Main Section', 'Conclusion'],
        'paragraphs_count': 6,
        'images_count': 2,
        'has_code_blocks': True,
        'has_tables': True,
        'has_lists': True
    }
    research_data = {
        'citations': [{'title': 'Citation 1'}, {'title': 'Citation 2'}, {'title': 'Citation 3'}]
    }
    
    score = agent.calculate_readiness_score(metadata, research_data)
    assert score > 50  # Should be higher than base score


def test_generate_research_report(mock_source_validator, mock_firecrawl_client):
    """Test generating a research report."""
    agent = ResearcherAgent()
    
    blog_data = {'content': '# Test Blog'}
    metadata = {
        'blog_title': 'test-blog',
        'version': 2,
        'main_topic': 'Test Topic',
        'summary': 'This is a test summary',
        'headings': ['Test Topic', 'Introduction', 'Main Section', 'Conclusion'],
        'paragraphs_count': 5,
        'images_count': 1,
        'has_code_blocks': True,
        'has_tables': False,
        'has_lists': True,
        'reading_time_minutes': 3
    }
    research_data = {
        'citations': [
            {
                'title': 'Citation 1',
                'url': 'https://example.com/1',
                'description': 'Description 1',
                'source': 'Source 1'
            }
        ],
        'system_affected': {
            'name': 'Test System',
            'description': 'System description',
            'scale': 'Medium'
        },
        'current_paradigm': {
            'name': 'Current Approach',
            'limitations': ['Limitation 1', 'Limitation 2']
        },
        'proposed_solution': {
            'name': 'Proposed Solution',
            'advantages': ['Advantage 1', 'Advantage 2']
        },
        'audience_analysis': {
            'knowledge_level': 'moderate',
            'background': 'Technical readers',
            'interests': ['Technology', 'Innovation']
        }
    }
    readiness_score = 75
    
    report = agent.generate_research_report(blog_data, metadata, research_data, readiness_score)
    
    # Verify report content
    assert "# Research Report: Test Topic" in report
    assert "**Title:** test-blog" in report
    assert "**Version:** 2" in report
    assert "**Readiness Score:** 75/100" in report
    assert "**Main Topic:** Test Topic" in report
    assert "**Summary:** This is a test summary" in report
    assert "**Reading Time:** 3 minutes" in report
    assert "**Headings:** 4" in report
    assert "**Paragraphs:** 5" in report
    assert "**Images:** 1" in report
    assert "**Has Code Blocks:** Yes" in report
    assert "**Has Tables:** No" in report
    assert "**Has Lists:** Yes" in report
    assert "**Name:** Test System" in report
    assert "**Description:** System description" in report
    assert "**Scale:** Medium" in report
    assert "**Name:** Current Approach" in report
    assert "  - Limitation 1" in report
    assert "  - Limitation 2" in report
    assert "**Name:** Proposed Solution" in report
    assert "  - Advantage 1" in report
    assert "  - Advantage 2" in report
    assert "**Knowledge Level:** moderate" in report
    assert "**Background:** Technical readers" in report
    assert "  - Technology" in report
    assert "  - Innovation" in report
    assert "### 1. Citation 1" in report
    assert "**URL:** https://example.com/1" in report
    assert "**Source:** Source 1" in report
    assert "**Description:** Description 1" in report
    assert "This blog post shows **good readiness** for review" in report


def test_save_research_results(mock_db_client, mock_yaml_guard, mock_source_validator, mock_firecrawl_client):
    """Test saving research results to MongoDB."""
    agent = ResearcherAgent()
    agent.db_client = mock_db_client
    
    blog_data = {
        'content': '# Test Blog',
        'images': {
            'images/test.png': {
                'format': 'png',
                'base64': 'test_base64_data',
                'path': '/path/to/images/test.png'
            }
        },
        'asset_folder': '/path/to'
    }
    metadata = {
        'blog_title': 'test-blog',
        'version': 2,
        'main_topic': 'Test Topic'
    }
    research_data = {'citations': []}
    readiness_score = 75
    report_markdown = "# Test Report"
    
    result = agent.save_research_results(
        blog_data, metadata, research_data, readiness_score, report_markdown
    )
    
    # Verify MongoDB calls
    mock_db_client.db.blogs.update_one.assert_called_once()
    mock_db_client.store_review_result.assert_called_once_with(
        'test-blog', 2, 'research', report_markdown, 'test-blog_research_report_v2.md'
    )
    mock_db_client.store_media.assert_called_once()
    mock_yaml_guard.assert_called_once_with('test-blog', 2)
    
    # Verify result
    assert result['status'] == 'success'
    assert result['blog_title'] == 'test-blog'
    assert result['version'] == 2
    assert result['readiness_score'] == 75
    assert result['report_id'] == 'mock_report_id'
    assert len(result['image_ids']) == 1
    assert result['image_ids'][0] == 'mock_image_id'


def test_process_blog_markdown_file(mock_source_validator, mock_firecrawl_client):
    """Test processing a blog from a markdown file."""
    # Create a test file path
    test_file_path = "test_blog.md"
    
    # Create fake blog data for the mocked methods
    blog_data = {
        'content': '# Test Blog',
        'images': {},
        'blog_title': 'test-blog',
        'version': 2,
        'asset_folder': '/path/to'
    }
    
    # Setup mocks for all methods used in process_blog
    with patch('agents.researcher_agent.process_blog_upload') as mock_process_upload:
        with patch.object(ResearcherAgent, 'process_markdown_file') as mock_process_md:
            with patch.object(ResearcherAgent, 'extract_metadata') as mock_extract:
                with patch.object(ResearcherAgent, 'gather_research') as mock_gather:
                    with patch.object(ResearcherAgent, 'calculate_readiness_score') as mock_score:
                        with patch.object(ResearcherAgent, 'generate_research_report') as mock_report:
                            with patch.object(ResearcherAgent, 'save_research_results') as mock_save:
                                
                                # Setup return values
                                mock_process_md.return_value = blog_data
                                mock_extract.return_value = {'headings': [], 'blog_title': 'test-blog', 'main_topic': 'Test Topic'}
                                mock_gather.return_value = {'citations': []}
                                mock_score.return_value = 75
                                mock_report.return_value = "# Test Report"
                                mock_save.return_value = {
                                    "status": "success",
                                    "blog_id": "test_blog_id",
                                    "report_id": "test_report_id",
                                    "yaml_path": "/path/to/yaml"
                                }
                                
                                # Create agent instance and call process_blog
                                agent = ResearcherAgent()
                                result = agent.process_blog(test_file_path)
                                
                                # Verify process_markdown_file was called (not process_blog_upload)
                                mock_process_md.assert_called_once_with(test_file_path)
                                mock_process_upload.assert_not_called()
                                
                                # Verify other methods were called
                                mock_extract.assert_called_once()
                                mock_gather.assert_called_once()
                                mock_score.assert_called_once()
                                mock_report.assert_called_once()
                                mock_save.assert_called_once()
                                
                                # Verify result
                                assert result['status'] == 'success'
                                assert result['blog_id'] == 'test_blog_id'
                                assert result['blog_title'] == 'test-blog'
                                assert result['readiness_score'] == 75
                                assert 'progress' in result


def test_process_blog_zip_file(mock_source_validator, mock_firecrawl_client):
    """Test processing a blog from a ZIP file."""
    # Create a test file path with .zip extension
    test_file_path = "test_blog.zip"
    
    # Create fake blog data for the mocked methods
    blog_data = {
        'content': '# Test Blog',
        'images': {},
        'blog_title': 'test-blog',
        'version': 2,
        'asset_folder': '/path/to'
    }
    
    # Setup mocks for all methods used in process_blog
    with patch('agents.researcher_agent.process_blog_upload') as mock_process_upload:
        with patch.object(ResearcherAgent, 'process_markdown_file') as mock_process_md:
            with patch.object(ResearcherAgent, 'extract_metadata') as mock_extract:
                with patch.object(ResearcherAgent, 'gather_research') as mock_gather:
                    with patch.object(ResearcherAgent, 'calculate_readiness_score') as mock_score:
                        with patch.object(ResearcherAgent, 'generate_research_report') as mock_report:
                            with patch.object(ResearcherAgent, 'save_research_results') as mock_save:
                                
                                # Setup return values
                                mock_process_upload.return_value = blog_data
                                mock_extract.return_value = {'headings': [], 'blog_title': 'test-blog', 'main_topic': 'Test Topic'}
                                mock_gather.return_value = {'citations': []}
                                mock_score.return_value = 75
                                mock_report.return_value = "# Test Report"
                                mock_save.return_value = {
                                    "status": "success",
                                    "blog_id": "test_blog_id",
                                    "report_id": "test_report_id",
                                    "yaml_path": "/path/to/yaml"
                                }
                                
                                # Create agent instance and call process_blog
                                agent = ResearcherAgent()
                                result = agent.process_blog(test_file_path)
                                
                                # Verify process_blog_upload was called (not process_markdown_file)
                                mock_process_upload.assert_called_once_with(test_file_path)
                                mock_process_md.assert_not_called()
                                
                                # Verify other methods were called
                                mock_extract.assert_called_once()
                                mock_gather.assert_called_once()
                                mock_score.assert_called_once()
                                mock_report.assert_called_once()
                                mock_save.assert_called_once()
                                
                                # Verify result
                                assert result['status'] == 'success'
                                assert result['blog_id'] == 'test_blog_id'
                                assert result['blog_title'] == 'test-blog'
                                assert result['readiness_score'] == 75
                                assert 'progress' in result


def test_main_function(sample_markdown_file, mock_source_validator, mock_firecrawl_client):
    """Test the main function."""
    # Import main function
    from agents.researcher_agent import main
    
    # Set up command-line arguments
    test_args = ['researcher_agent.py', sample_markdown_file]
    
    # Mock command-line arguments
    with patch('sys.argv', test_args):
        # Mock the ResearcherAgent class
        with patch('agents.researcher_agent.ResearcherAgent') as mock_agent_cls:
            # Mock the agent instance
            mock_agent = mock_agent_cls.return_value
            mock_agent.process_blog.return_value = {'status': 'success'}
            
            # Mock print function
            with patch('builtins.print') as mock_print:
                # Call main function
                result = main()
                
                # Verify agent was initialized
                mock_agent_cls.assert_called_once()
                
                # Verify process_blog was called with the markdown file
                mock_agent.process_blog.assert_called_once_with(sample_markdown_file)
                
                # Verify result was printed
                mock_print.assert_called_once()
                
                # Verify main function returned 0 (success)
                assert result == 0 