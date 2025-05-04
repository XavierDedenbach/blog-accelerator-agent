from agents.utilities.db import MongoDBClient

def main():
    print("Connecting to MongoDB...")
    db = MongoDBClient('mongodb://localhost:27017')
    
    print("\nReview Files:")
    for review in db.db.review_files.find({'blog_title': 'hardware-companies-shouldnt-run-pure-agile'}):
        print(f"{review.get('stage')}: {review.get('filename')}")
    
    print("\nBlog Status:")
    blog = db.get_blog_status('hardware-companies-shouldnt-run-pure-agile')
    if blog:
        print(f"Title: {blog.get('title')}")
        print(f"Current Version: {blog.get('current_version')}")
        versions = blog.get('versions', [])
        if versions:
            latest = versions[-1]
            print(f"Latest Version: {latest.get('version')}")
            print(f"Readiness Score: {latest.get('readiness_score')}")
            
            review_status = latest.get('review_status', {})
            for stage, status in review_status.items():
                print(f"  {stage}: {'Complete' if status.get('complete') else 'Incomplete'}")

if __name__ == "__main__":
    main() 