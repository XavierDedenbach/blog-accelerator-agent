from agents.utilities.db import MongoDBClient

def main():
    print("Connecting to MongoDB...")
    db = MongoDBClient('mongodb://localhost:27017')
    
    print("\nRetrieving research report...")
    blog_title = 'hardware-companies-shouldnt-run-pure-agile'
    version = 1
    
    # Get research report
    research_report = db.db.review_files.find_one({
        'blog_title': blog_title,
        'version': version,
        'stage': 'research'
    })
    
    if research_report:
        print("\n===== RESEARCH REPORT =====")
        content = research_report.get('content')
        
        # Print the first 500 lines or the full content if shorter
        lines = content.split('\n')
        print('\n'.join(lines[:100]) + "...\n")
        
        print(f"\nTotal lines in research report: {len(lines)}")
    else:
        print("\nNo research report found.")

if __name__ == "__main__":
    main() 