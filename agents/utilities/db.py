"""
Database utilities for MongoDB operations with version awareness.
Handles blog metadata, review results, media writes, and enhanced research storage.
"""

import os
from typing import Dict, Any, List, Optional, Union
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timezone
from pymongo.database import Database


class MongoDBClient:
    """MongoDB client wrapper with version-aware operations for blog data."""

    def __init__(self, uri: Optional[str] = None):
        """
        Initialize MongoDB client using URI from environment or parameter.
        
        Args:
            uri: MongoDB connection URI, defaults to MONGODB_URI environment variable
        """
        self.uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        self.client = MongoClient(self.uri)
        self.db = self.client.blog_accelerator
        
    def insert_or_archive(self, collection: str, document: Dict[str, Any], 
                          version_field: str = "version", 
                          title_field: str = "blog_title") -> str:
        """
        Insert a new document or archive old versions.
        
        If a document with the same title but different version exists, 
        the old document is archived and the new one is inserted.
        
        Args:
            collection: Name of MongoDB collection
            document: Document to insert
            version_field: Field name containing version number
            title_field: Field name containing blog title
            
        Returns:
            ID of inserted document
        """
        coll = self.db[collection]
        
        title = document.get(title_field)
        version = document.get(version_field)
        
        if not title or not version:
            raise ValueError(f"Document must contain {title_field} and {version_field}")
        
        # Check if we need to archive older versions
        existing = coll.find_one({title_field: title, version_field: {"$ne": version}})
        if existing:
            # Archive the old document by adding an 'archived' field
            coll.update_one(
                {"_id": existing["_id"]},
                {"$set": {"archived": True, "archived_at": datetime.now(timezone.utc)}}
            )
            
        # Insert the new document
        result = coll.insert_one(document)
        return str(result.inserted_id)
    
    def get_latest_version(self, collection: str, title: str, 
                          title_field: str = "blog_title") -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a document by title.
        
        Args:
            collection: Name of MongoDB collection
            title: Blog title to search for
            title_field: Field name containing blog title
            
        Returns:
            Latest version document or None if not found
        """
        coll = self.db[collection]
        return coll.find_one(
            {title_field: title, "archived": {"$ne": True}},
            sort=[("version", -1)]
        )
    
    def update_blog_metadata(self, title: str, version: int, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific blog version.
        
        Args:
            title: Blog title
            version: Version number
            metadata: Metadata to update
            
        Returns:
            True if update succeeded, False otherwise
        """
        result = self.db.blogs.update_one(
            {"title": title, "versions.version": version},
            {"$set": {f"versions.$.{k}": v for k, v in metadata.items()}}
        )
        return result.modified_count > 0
    
    def store_review_result(self, title: str, version: int, 
                           stage: str, content: str, filename: str) -> str:
        """
        Store a review result document.
        
        Args:
            title: Blog title
            version: Version number
            stage: Review stage (factual_review, style_review, grammar_review)
            content: Markdown content of review
            filename: Filename of review document
            
        Returns:
            ID of inserted document
        """
        document = {
            "blog_title": title,
            "version": version,
            "stage": stage,
            "filename": filename,
            "content": content,
            "timestamp": datetime.now(timezone.utc)
        }
        return self.insert_or_archive("review_files", document)
    
    def store_media(self, title: str, version: int, 
                   media_type: str, source: str, 
                   url: Optional[str] = None, 
                   base64_data: Optional[str] = None,
                   alt_text: Optional[str] = None,
                   category: Optional[str] = None) -> str:
        """
        Store media (images, etc.) associated with a blog.
        
        Args:
            title: Blog title
            version: Version number
            media_type: Type of media (image, video, etc.)
            source: Source of media (Firecrawl MCP, local, etc.)
            url: Optional URL to media
            base64_data: Optional base64 encoded data
            alt_text: Optional alt text for accessibility
            category: Optional category (diagram, illustration, photo, infographic)
            
        Returns:
            ID of inserted media document
        """
        if not url and not base64_data:
            raise ValueError("Either url or base64_data must be provided")
            
        document = {
            "blog_title": title,
            "version": version,
            "type": media_type,
            "source": source,
            "url": url,
            "stored_base64": base64_data,
            "alt_text": alt_text,
            "category": category,
            "timestamp": datetime.now(timezone.utc)
        }
        return self.insert_or_archive("media", document)
    
    def get_blog_status(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a blog including version and review status.
        
        Args:
            title: Blog title
            
        Returns:
            Blog document or None if not found
        """
        return self.db.blogs.find_one({"title": title})
    
    def store_research_component(
        self,
        title: str,
        version: int,
        component_type: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Store research component data (industry analysis, solution analysis, etc.).
        
        Args:
            title: Blog title
            version: Version number
            component_type: Type of component (industry_analysis, solution_analysis, etc.)
            data: Component data
            
        Returns:
            ID of inserted document
        """
        document = {
            "blog_title": title,
            "version": version,
            "component_type": component_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc)
        }
        return self.insert_or_archive("research_components", document)
    
    def get_research_component(
        self,
        title: str,
        version: int,
        component_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get research component data.
        
        Args:
            title: Blog title
            version: Version number
            component_type: Type of component
            
        Returns:
            Component data or None if not found
        """
        return self.db.research_components.find_one({
            "blog_title": title,
            "version": version,
            "component_type": component_type,
            "archived": {"$ne": True}
        })
    
    def store_visual_asset_collection(
        self,
        title: str,
        version: int,
        topic: str,
        assets: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Store collection of visual assets.
        
        Args:
            title: Blog title
            version: Version number
            topic: Topic of visual assets
            assets: Dictionary mapping categories to lists of visual assets
            
        Returns:
            ID of inserted document
        """
        document = {
            "blog_title": title,
            "version": version,
            "topic": topic,
            "assets": assets,
            "assets_count": sum(len(category_assets) for category_assets in assets.values()),
            "categories": list(assets.keys()),
            "timestamp": datetime.now(timezone.utc)
        }
        return self.insert_or_archive("visual_assets", document)
    
    def get_visual_assets(
        self,
        title: str,
        version: int,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get visual assets for a blog.
        
        Args:
            title: Blog title
            version: Version number
            category: Optional category to filter by
            limit: Maximum number of assets to return
            
        Returns:
            List of visual assets
        """
        # Get the visual assets collection
        collection = self.db.visual_assets.find_one({
            "blog_title": title,
            "version": version,
            "archived": {"$ne": True}
        })
        
        if not collection:
            return []
        
        assets = collection.get("assets", {})
        
        # Filter by category if provided
        if category and category in assets:
            return assets[category][:limit]
        
        # Otherwise, return assets from all categories
        all_assets = []
        for cat_assets in assets.values():
            all_assets.extend(cat_assets)
        
        return all_assets[:limit]
    
    def store_analogy(
        self,
        title: str,
        version: int,
        category: str,
        analogies: List[Dict[str, Any]]
    ) -> str:
        """
        Store analogies for audience explanation.
        
        Args:
            title: Blog title
            version: Version number
            category: Category of analogies (challenges, solutions)
            analogies: List of analogy objects
            
        Returns:
            ID of inserted document
        """
        document = {
            "blog_title": title,
            "version": version,
            "category": category,
            "analogies": analogies,
            "count": len(analogies),
            "timestamp": datetime.now(timezone.utc)
        }
        return self.insert_or_archive("analogies", document)
    
    def get_analogies(
        self,
        title: str,
        version: int,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get analogies for a blog.
        
        Args:
            title: Blog title
            version: Version number
            category: Optional category to filter by
            
        Returns:
            List of analogies
        """
        query = {
            "blog_title": title,
            "version": version,
            "archived": {"$ne": True}
        }
        
        if category:
            query["category"] = category
        
        documents = list(self.db.analogies.find(query))
        
        # Extract analogies from documents
        all_analogies = []
        for doc in documents:
            all_analogies.extend(doc.get("analogies", []))
        
        return all_analogies
    
    def update_research_stats(
        self,
        title: str,
        version: int,
        stats: Dict[str, Any]
    ) -> bool:
        """
        Update research statistics in blog document.
        
        Args:
            title: Blog title
            version: Version number
            stats: Research statistics
            
        Returns:
            True if update succeeded, False otherwise
        """
        result = self.db.blogs.update_one(
            {"title": title, "versions.version": version},
            {"$set": {
                "versions.$.research_stats": stats,
                "versions.$.research_updated_at": datetime.now(timezone.utc).isoformat()
            }}
        )
        return result.modified_count > 0

def get_db_client() -> Database:
    """Helper function to get a MongoDB database client."""
    client = MongoDBClient()
    return client.db
