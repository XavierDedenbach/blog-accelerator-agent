"""
Tests for the MongoDB utilities in db.py.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from bson.objectid import ObjectId

from agents.utilities.db import MongoDBClient


@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client with required methods."""
    with patch('agents.utilities.db.MongoClient') as mock_client:
        # Create mock collections
        mock_blogs = MagicMock()
        mock_review_files = MagicMock()
        mock_media = MagicMock()
        
        # Setup collection access via __getitem__
        mock_collections = {'blogs': mock_blogs, 'review_files': mock_review_files, 'media': mock_media}
        
        def getitem_side_effect(name):
            return mock_collections.get(name, MagicMock())
        
        # Setup mock db with collections
        mock_db = MagicMock()
        mock_db.__getitem__.side_effect = getitem_side_effect
        
        # Also directly attach the collections as properties
        mock_db.blogs = mock_blogs
        
        # Setup client
        mock_client_instance = MagicMock()
        mock_client_instance.blog_accelerator = mock_db
        mock_client.return_value = mock_client_instance
        
        # Create DB client for test
        db_client = MongoDBClient(uri="mongodb://testhost:27017")
        
        # Save mock collections for later assertions
        db_client.mock_collections = mock_collections
        
        yield db_client


def test_insert_or_archive_new_document(mock_mongo_client):
    """Test inserting a new document without archiving."""
    # Setup
    collection_name = "test_collection"
    mock_collection = MagicMock()
    mock_mongo_client.mock_collections[collection_name] = mock_collection
    mock_mongo_client.db[collection_name] = mock_collection
    
    # Setup mock behaviors
    mock_collection.find_one.return_value = None
    mock_collection.insert_one.return_value = MagicMock()
    mock_collection.insert_one.return_value.inserted_id = ObjectId("5f50c31e8a91e51cacb2cde1")
    
    # Exercise
    document = {"blog_title": "test-blog", "version": 1, "content": "test content"}
    result = mock_mongo_client.insert_or_archive(collection_name, document)
    
    # Verify
    mock_collection.find_one.assert_called_once()
    mock_collection.insert_one.assert_called_once_with(document)
    assert result == "5f50c31e8a91e51cacb2cde1"


def test_insert_or_archive_existing_document(mock_mongo_client):
    """Test inserting a document when an older version exists (should archive)."""
    # Setup
    collection_name = "test_collection"
    mock_collection = MagicMock()
    mock_mongo_client.mock_collections[collection_name] = mock_collection
    mock_mongo_client.db[collection_name] = mock_collection
    
    existing_doc = {
        "_id": ObjectId("5f50c31e8a91e51cacb2cde1"),
        "blog_title": "test-blog", 
        "version": 1
    }
    mock_collection.find_one.return_value = existing_doc
    mock_collection.insert_one.return_value = MagicMock()
    # ObjectId requires a 24-character hex string
    mock_collection.insert_one.return_value.inserted_id = ObjectId("6f60d41f9ba2f62dadbcdf02")
    
    # Exercise
    document = {"blog_title": "test-blog", "version": 2, "content": "updated content"}
    result = mock_mongo_client.insert_or_archive(collection_name, document)
    
    # Verify
    mock_collection.find_one.assert_called_once()
    mock_collection.update_one.assert_called_once()
    mock_collection.insert_one.assert_called_once_with(document)
    assert result == "6f60d41f9ba2f62dadbcdf02"


def test_get_latest_version(mock_mongo_client):
    """Test retrieving the latest version of a document."""
    # Setup
    collection_name = "test_collection"
    mock_collection = MagicMock()
    mock_mongo_client.mock_collections[collection_name] = mock_collection
    mock_mongo_client.db[collection_name] = mock_collection
    
    expected_doc = {
        "_id": ObjectId("5f50c31e8a91e51cacb2cde1"),
        "blog_title": "test-blog", 
        "version": 2,
        "content": "latest content"
    }
    mock_collection.find_one.return_value = expected_doc
    
    # Exercise
    result = mock_mongo_client.get_latest_version(collection_name, "test-blog")
    
    # Verify
    assert result == expected_doc
    mock_collection.find_one.assert_called_once()


def test_update_blog_metadata(mock_mongo_client):
    """Test updating blog metadata."""
    # Setup
    mock_blogs = MagicMock()
    mock_mongo_client.db.blogs = mock_blogs
    
    update_result = MagicMock()
    update_result.modified_count = 1
    mock_blogs.update_one.return_value = update_result
    
    # Exercise
    metadata = {"readiness_score": 80, "review_status.factual_review.complete": True}
    result = mock_mongo_client.update_blog_metadata("test-blog", 1, metadata)
    
    # Verify
    assert result is True
    mock_blogs.update_one.assert_called_once()


def test_store_review_result(mock_mongo_client):
    """Test storing a review result document."""
    # Setup - using patch to not reimplement insert_or_archive
    with patch.object(mock_mongo_client, 'insert_or_archive', return_value="7g70e51g0cb3g73ebe4eg3") as mock_insert:
        # Exercise
        result = mock_mongo_client.store_review_result(
            "test-blog", 1, "factual_review", "Review content", "test-blog_review1.md"
        )
        
        # Verify
        assert result == "7g70e51g0cb3g73ebe4eg3"
        mock_insert.assert_called_once()
        # Validate the document structure passed to insert_or_archive
        doc = mock_insert.call_args[0][1]
        assert doc["blog_title"] == "test-blog"
        assert doc["version"] == 1
        assert doc["stage"] == "factual_review"
        assert doc["content"] == "Review content"
        assert doc["filename"] == "test-blog_review1.md"
        assert isinstance(doc["timestamp"], datetime)


def test_store_media(mock_mongo_client):
    """Test storing media with URL."""
    # Setup - using patch to not reimplement insert_or_archive
    with patch.object(mock_mongo_client, 'insert_or_archive', return_value="8h80f61h1dc4h84fcf5fh4") as mock_insert:
        # Exercise
        result = mock_mongo_client.store_media(
            "test-blog", 1, "image", "Firecrawl MCP", 
            url="https://example.com/image.jpg",
            alt_text="Example image"
        )
        
        # Verify
        assert result == "8h80f61h1dc4h84fcf5fh4"
        mock_insert.assert_called_once()
        # Validate the document structure passed to insert_or_archive
        doc = mock_insert.call_args[0][1]
        assert doc["blog_title"] == "test-blog"
        assert doc["version"] == 1
        assert doc["type"] == "image"
        assert doc["source"] == "Firecrawl MCP"
        assert doc["url"] == "https://example.com/image.jpg"
        assert doc["alt_text"] == "Example image"


def test_store_media_validation(mock_mongo_client):
    """Test validation error when neither URL nor base64 is provided."""
    # Exercise & Verify
    with pytest.raises(ValueError) as excinfo:
        mock_mongo_client.store_media(
            "test-blog", 1, "image", "Firecrawl MCP"
        )
    assert "Either url or base64_data must be provided" in str(excinfo.value)


def test_get_blog_status(mock_mongo_client):
    """Test retrieving blog status."""
    # Setup
    mock_blogs = MagicMock()
    mock_mongo_client.db.blogs = mock_blogs
    
    expected_doc = {
        "_id": ObjectId("5f50c31e8a91e51cacb2cde1"),
        "title": "test-blog",
        "current_version": 2
    }
    mock_blogs.find_one.return_value = expected_doc
    
    # Exercise
    result = mock_mongo_client.get_blog_status("test-blog")
    
    # Verify
    assert result == expected_doc
    mock_blogs.find_one.assert_called_once_with({"title": "test-blog"}) 