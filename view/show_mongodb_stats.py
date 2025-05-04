from agents.utilities.db import MongoDBClient
import json

def main():
    print("Connecting to MongoDB...")
    db = MongoDBClient('mongodb://localhost:27017')
    
    print("\nDatabase Collections:")
    collections = db.db.list_collection_names()
    for collection in collections:
        count = db.db[collection].count_documents({})
        print(f"  {collection}: {count} documents")
    
    print("\nBlogs Collection:")
    blogs = list(db.db.blogs.find({}))
    for blog in blogs:
        blog['_id'] = str(blog['_id'])  # Convert ObjectId to string for display
        print(f"  Title: {blog.get('title')}")
        print(f"  Current Version: {blog.get('current_version')}")
        print(f"  Versions: {len(blog.get('versions', []))}")
        
    print("\nReview Files Collection:")
    review_stages = db.db.review_files.distinct('stage')
    for stage in review_stages:
        count = db.db.review_files.count_documents({'stage': stage})
        print(f"  {stage}: {count} documents")
    
if __name__ == "__main__":
    main() 