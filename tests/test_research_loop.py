#!/usr/bin/env python
"""
Simple test script for research components.
This tests the full research loop including all components.
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("Could not import dotenv. Make sure python-dotenv is installed.")
    logger.warning("Continuing without loading .env file...")

# Import research components - use dynamic imports to handle potential import errors
try:
    from agents.research.industry_analysis import IndustryAnalyzer
    industry_module_available = True
except ImportError as e:
    logger.error(f"Could not import IndustryAnalyzer: {e}")
    industry_module_available = False

try:
    from agents.research.solution_analysis import SolutionAnalyzer
    solution_module_available = True
except ImportError as e:
    logger.error(f"Could not import SolutionAnalyzer: {e}")
    solution_module_available = False

try:
    from agents.research.paradigm_analysis import ParadigmAnalyzer
    paradigm_module_available = True
except ImportError as e:
    logger.error(f"Could not import ParadigmAnalyzer: {e}")
    paradigm_module_available = False

try:
    from agents.research.audience_analysis import AudienceAnalyzer
    audience_module_available = True
except ImportError as e:
    logger.error(f"Could not import AudienceAnalyzer: {e}")
    audience_module_available = False

try:
    from agents.research.analogy_generator import AnalogyGenerator
    analogy_module_available = True
except ImportError as e:
    logger.error(f"Could not import AnalogyGenerator: {e}")
    analogy_module_available = False

try:
    from agents.utilities.source_validator import SourceValidator
    source_validator_available = True
except ImportError as e:
    logger.error(f"Could not import SourceValidator: {e}")
    source_validator_available = False

try:
    from agents.utilities.firecrawl_client import FirecrawlClient
    firecrawl_client_available = True
except ImportError as e:
    logger.error(f"Could not import FirecrawlClient: {e}")
    firecrawl_client_available = False

async def test_industry_analysis(topic, source_validator=None):
    """Test the industry analysis component."""
    if not industry_module_available:
        logger.error("Industry analysis module not available. Skipping test.")
        return {"error": "Module not available"}
    
    try:
        # Get API key from environment
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not openai_api_key and not groq_api_key:
            logger.error("No API keys available for LLM. Check your .env file for OPENAI_API_KEY or GROQ_API_KEY.")
            return {"error": "No API keys available"}
        
        # Initialize analyzer
        analyzer = IndustryAnalyzer(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            source_validator=source_validator
        )
        
        # Run analysis
        logger.info(f"Running industry analysis for topic: {topic}")
        start_time = datetime.now()
        result = await analyzer.analyze_industry(topic)
        end_time = datetime.now()
        
        # Log results
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Industry analysis completed in {duration:.2f} seconds")
        logger.info(f"Found {len(result.get('challenges', []))} industry challenges")
        
        # Save results
        output_file = f"demo/industry_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Industry analysis results saved to {output_file}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in industry analysis test: {e}")
        return {"error": str(e)}

async def test_solution_analysis(topic, challenges, source_validator=None):
    """Test the solution analysis component."""
    if not solution_module_available:
        logger.error("Solution analysis module not available. Skipping test.")
        return {"error": "Module not available"}
    
    try:
        # Get API key from environment
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not openai_api_key and not groq_api_key:
            logger.error("No API keys available for LLM. Check your .env file for OPENAI_API_KEY or GROQ_API_KEY.")
            return {"error": "No API keys available"}
        
        # Initialize analyzer
        analyzer = SolutionAnalyzer(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            source_validator=source_validator
        )
        
        # Run analysis
        solution = f"Alternative approach to {topic}"
        logger.info(f"Running solution analysis for topic: {topic}")
        start_time = datetime.now()
        result = await analyzer.analyze_solution(topic, solution, challenges)
        end_time = datetime.now()
        
        # Log results
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Solution analysis completed in {duration:.2f} seconds")
        logger.info(f"Found {len(result.get('pro_arguments', []))} pro arguments and {len(result.get('counter_arguments', []))} counter arguments")
        
        # Save results
        output_file = f"demo/solution_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Solution analysis results saved to {output_file}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in solution analysis test: {e}")
        return {"error": str(e)}

async def test_paradigm_analysis(topic, source_validator=None):
    """Test the paradigm analysis component."""
    if not paradigm_module_available:
        logger.error("Paradigm analysis module not available. Skipping test.")
        return {"error": "Module not available"}
    
    try:
        # Get API key from environment
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not openai_api_key and not groq_api_key:
            logger.error("No API keys available for LLM. Check your .env file for OPENAI_API_KEY or GROQ_API_KEY.")
            return {"error": "No API keys available"}
        
        # Initialize analyzer
        analyzer = ParadigmAnalyzer(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            source_validator=source_validator
        )
        
        # Run analysis
        logger.info(f"Running paradigm analysis for topic: {topic}")
        start_time = datetime.now()
        result = await analyzer.analyze_paradigms(topic)
        end_time = datetime.now()
        
        # Log results
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Paradigm analysis completed in {duration:.2f} seconds")
        logger.info(f"Found {len(result.get('historical_paradigms', []))} historical paradigms")
        
        # Save results
        output_file = f"demo/paradigm_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Paradigm analysis results saved to {output_file}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in paradigm analysis test: {e}")
        return {"error": str(e)}

async def test_audience_analysis(topic, source_validator=None):
    """Test the audience analysis component."""
    if not audience_module_available:
        logger.error("Audience analysis module not available. Skipping test.")
        return {"error": "Module not available"}
    
    try:
        # Get API key from environment
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not openai_api_key and not groq_api_key:
            logger.error("No API keys available for LLM. Check your .env file for OPENAI_API_KEY or GROQ_API_KEY.")
            return {"error": "No API keys available"}
        
        # Initialize analyzer with default segments
        analyzer = AudienceAnalyzer(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            source_validator=source_validator,
            use_default_segments=True
        )
        
        # Run analysis
        logger.info(f"Running audience analysis for topic: {topic}")
        start_time = datetime.now()
        result = await analyzer.analyze_audience(topic)
        end_time = datetime.now()
        
        # Log results
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Audience analysis completed in {duration:.2f} seconds")
        logger.info(f"Found {len(result.get('audience_segments', []))} audience segments")
        
        # Save results
        output_file = f"demo/audience_analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Audience analysis results saved to {output_file}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in audience analysis test: {e}")
        return {"error": str(e)}

async def test_analogy_generation(topic, source_validator=None, firecrawl_client=None):
    """Test the analogy generator component."""
    if not analogy_module_available:
        logger.error("Analogy generator module not available. Skipping test.")
        return {"error": "Module not available"}
    
    try:
        # Get API key from environment
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        groq_api_key = os.environ.get('GROQ_API_KEY')
        
        if not openai_api_key and not groq_api_key:
            logger.error("No API keys available for LLM. Check your .env file for OPENAI_API_KEY or GROQ_API_KEY.")
            return {"error": "No API keys available"}
        
        # Initialize generator
        generator = AnalogyGenerator(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            source_validator=source_validator,
            firecrawl_client=firecrawl_client
        )
        
        # Run generation
        logger.info(f"Running analogy generation for topic: {topic}")
        start_time = datetime.now()
        result = await generator.generate_analogies(topic)
        end_time = datetime.now()
        
        # Log results
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Analogy generation completed in {duration:.2f} seconds")
        logger.info(f"Generated {len(result.get('generated_analogies', []))} analogies")
        
        # Save results
        output_file = f"demo/analogy_generation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Analogy generation results saved to {output_file}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in analogy generation test: {e}")
        return {"error": str(e)}

async def main():
    """Main test function."""
    try:
        # Test topic
        topic = "Hardware companies shouldn't run pure agile"
        
        # Initialize shared utilities
        source_validator = None
        firecrawl_client = None
        
        if source_validator_available:
            brave_api_key = os.environ.get('BRAVE_API_KEY')
            source_validator = SourceValidator(brave_api_key=brave_api_key)
        
        if firecrawl_client_available:
            firecrawl_server = os.environ.get('FIRECRAWL_SERVER')
            firecrawl_client = FirecrawlClient(server_url=firecrawl_server)
        
        # Run all research components
        results = {}
        
        # 1. Industry analysis
        results['industry'] = await test_industry_analysis(topic, source_validator)
        
        # 2. Solution analysis (using challenges from industry analysis)
        challenges = []
        if 'error' not in results['industry']:
            challenges = results['industry'].get('challenges', [])
        results['solution'] = await test_solution_analysis(topic, challenges, source_validator)
        
        # 3. Paradigm analysis
        results['paradigm'] = await test_paradigm_analysis(topic, source_validator)
        
        # 4. Audience analysis
        results['audience'] = await test_audience_analysis(topic, source_validator)
        
        # 5. Analogy generation
        results['analogy'] = await test_analogy_generation(topic, source_validator, firecrawl_client)
        
        # Print a summary
        print("\n===== RESEARCH TEST RESULTS =====")
        
        if "error" in results['industry']:
            print(f"❌ Industry Analysis: {results['industry']['error']}")
        else:
            print(f"✅ Industry Analysis: Found {len(results['industry'].get('challenges', []))} challenges")
        
        if "error" in results['solution']:
            print(f"❌ Solution Analysis: {results['solution']['error']}")
        else:
            print(f"✅ Solution Analysis: Generated {len(results['solution'].get('pro_arguments', []))} pro arguments and {len(results['solution'].get('counter_arguments', []))} counter arguments")
        
        if "error" in results['paradigm']:
            print(f"❌ Paradigm Analysis: {results['paradigm']['error']}")
        else:
            print(f"✅ Paradigm Analysis: Found {len(results['paradigm'].get('historical_paradigms', []))} paradigms")
        
        if "error" in results['audience']:
            print(f"❌ Audience Analysis: {results['audience']['error']}")
        else:
            print(f"✅ Audience Analysis: Found {len(results['audience'].get('audience_segments', []))} segments")
        
        if "error" in results['analogy']:
            print(f"❌ Analogy Generation: {results['analogy']['error']}")
        else:
            print(f"✅ Analogy Generation: Created {len(results['analogy'].get('generated_analogies', []))} analogies")
        
        # Save combined results
        combined_results = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        combined_file = f"demo/full_research_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(combined_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        logger.info(f"Combined research results saved to {combined_file}")
    
    except Exception as e:
        logger.error(f"Error in main test function: {e}")
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 