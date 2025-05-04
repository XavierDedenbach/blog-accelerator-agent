from agents.utilities.db import MongoDBClient

def main():
    print("Connecting to MongoDB...")
    db = MongoDBClient('mongodb://localhost:27017')
    
    print("\nRetrieving review content...")
    blog_title = 'hardware-companies-shouldnt-run-pure-agile'
    version = 1
    
    # Get factual review
    factual_review = db.db.review_files.find_one({
        'blog_title': blog_title,
        'version': version,
        'stage': 'factual_review'
    })
    
    if factual_review:
        print("\n===== FACTUAL REVIEW =====")
        print(factual_review.get('content')[:1000] + "...\n")
    
    # Get style review
    style_review = db.db.review_files.find_one({
        'blog_title': blog_title,
        'version': version,
        'stage': 'style_review'
    })
    
    if style_review:
        print("\n===== STYLE REVIEW =====")
        print(style_review.get('content')[:1000] + "...\n")
    
    # Get grammar review
    grammar_review = db.db.review_files.find_one({
        'blog_title': blog_title,
        'version': version,
        'stage': 'grammar_review'
    })
    
    if grammar_review:
        print("\n===== GRAMMAR REVIEW =====")
        print(grammar_review.get('content')[:1000] + "...\n")

if __name__ == "__main__":
    main() 