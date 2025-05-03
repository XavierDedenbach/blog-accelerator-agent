#!/usr/bin/env python
"""
Simple test script for research components.
This tests just the research loop without requiring the full agent pipeline.
"""

import os
import asyncio
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import research components - use dynamic imports to handle potential import errors
try:
    from agents.research.audience_analysis import AudienceAnalyzer
    audience_module_available = True
except ImportError as e:
    logger.error(f"Could not import AudienceAnalyzer: {e}")
    audience_module_available = False

async def test_audience_analysis():
    """Test the audience analysis component."""
    if not audience_module_available:
        logger.error("Audience analysis module not available. Skipping test.")
        return {"error": "Module not available"}
    
    try:
        # Get API key from environment
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not openai_api_key and not groq_api_key:
            logger.error("No API keys available for LLM. Set OPENAI_API_KEY or GROQ_API_KEY environment variables.")
            return {"error": "No API keys available"}
        
        # Test topic
        topic = "Hardware companies shouldn't run pure agile"
        
        # Initialize analyzer with default segments
        analyzer = AudienceAnalyzer(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            use_default_segments=True
        )
        
        # Run analysis
        logger.info(f"Running audience analysis for topic: {topic}")
        start_time = datetime.now()
        result = await analyzer.analyze_audience(topic)
        end_time = datetime.now()
        
        # Log results
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Analysis completed in {duration:.2f} seconds")
        logger.info(f"Found {len(result.get('audience_segments', []))} audience segments")
        
        # Save results
        output_file = f"demo/audience_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in audience analysis test: {e}")
        return {"error": str(e)}

async def main():
    """Main test function."""
    try:
        # Test audience analysis
        audience_result = await test_audience_analysis()
        
        # Print a summary
        print("\n===== TEST RESULTS =====")
        if "error" in audience_result:
            print(f"❌ Audience Analysis: {audience_result['error']}")
        else:
            print(f"✅ Audience Analysis: Found {len(audience_result.get('audience_segments', []))} segments")
            
            # Show the first segment details as an example
            if audience_result.get('audience_segments'):
                segment = audience_result['audience_segments'][0]
                print(f"\nExample segment: {segment.get('name')}")
                print(f"Description: {segment.get('description')}")
                print(f"Knowledge level: {segment.get('knowledge_level')}")
                
                # Show content strategy recommendations
                if 'content_strategies' in segment:
                    print("\nRecommended content strategies:")
                    strategies = segment['content_strategies']
                    if 'recommended_formats' in strategies:
                        print(f"Formats: {', '.join(strategies['recommended_formats'][:3])}")
                    if 'tone_recommendations' in strategies:
                        print(f"Tone: {', '.join(strategies['tone_recommendations'][:3])}")
    
    except Exception as e:
        logger.error(f"Error in main test function: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 