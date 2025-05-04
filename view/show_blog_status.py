from agents.utilities.db import MongoDBClient
from agents.utilities.yaml_guard import get_review_status

def main():
    # Variables
    blog_title = 'hardware-companies-shouldnt-run-pure-agile'
    yaml_path = f'data/tracker_yaml/{blog_title}_review_tracker.yaml'
    
    # Get MongoDB data
    print("Getting MongoDB data...")
    db = MongoDBClient('mongodb://localhost:27017')
    
    # Get blog data
    blog = db.get_blog_status(blog_title)
    if blog:
        print("\n===== BLOG STATUS FROM MONGODB =====")
        print(f"Title: {blog.get('title')}")
        print(f"Current Version: {blog.get('current_version')}")
        
        # Get latest version
        versions = blog.get('versions', [])
        if versions:
            latest = versions[-1]
            print(f"Latest Version: {latest.get('version')}")
            print(f"Readiness Score: {latest.get('readiness_score')}")
            
            # Review status
            print("\nReview Status from MongoDB:")
            review_status = latest.get('review_status', {})
            for stage, status in review_status.items():
                print(f"  {stage}: {'Complete' if status.get('complete') else 'Incomplete'}")
    
    # Get YAML review status
    try:
        print("\n===== REVIEW STATUS FROM YAML =====")
        yaml_status = get_review_status(yaml_path)
        print(f"Blog Title: {yaml_status.get('blog_title')}")
        print(f"Version: {yaml_status.get('version')}")
        print(f"Current Stage: {yaml_status.get('current_stage')}")
        print(f"All Stages Complete: {yaml_status.get('all_stages_complete')}")
        print(f"Released: {yaml_status.get('released')}")
        
        # Review stages
        print("\nStages:")
        for stage, details in yaml_status.get('stages', {}).items():
            print(f"  {stage}: {'Complete' if details.get('complete') else 'Incomplete'}")
            if details.get('complete'):
                print(f"    Completed By: {details.get('completed_by')}")
                print(f"    Result File: {details.get('result_file')}")
                print(f"    Timestamp: {details.get('timestamp')}")
    except Exception as e:
        print(f"Error reading YAML: {e}")
    
    # Count all review files
    print("\n===== REVIEW FILES =====")
    for stage in ['research', 'factual_review', 'style_review', 'grammar_review']:
        count = db.db.review_files.count_documents({
            'blog_title': blog_title,
            'stage': stage
        })
        print(f"{stage}: {count} documents")

if __name__ == "__main__":
    main() 