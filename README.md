# Blog Accelerator Agent

An AI-powered tool for accelerating research and review of blog content.

## Project Overview

The Blog Accelerator Agent automates deep research and multi-layered review processes while preserving human authorship for ideation and writing. The agent operates in two primary modes:

1. **Research Mode**: Performs exhaustive topic research including industry analysis, solution evaluation, paradigm assessment, and visual asset collection.
2. **Review Mode**: Conducts a three-stage review process (factual, style, and grammar) with user approval gates.

## Recent Enhancements

We've recently implemented significant enhancements to the research capabilities:

### Completed Components:

- ✅ Source validation system with domain credibility scoring and blacklist management
- ✅ Firecrawl MCP integration for comprehensive visual asset collection (50-100 assets)
- ✅ Enhanced MongoDB schema for complex research data storage
- ✅ Industry analysis module with 10+ critical challenges identification and user persona targeting
- ✅ Solution analysis with pro/counter arguments and metrics tracking
- ✅ Paradigm analysis module for historical context assessment
- ✅ Audience analysis module for knowledge gap identification
- ✅ Analogy generator for simplified explanations
- ✅ Visual Asset Collector with intelligent search and categorization (diagrams, charts, photos, etc.)
- ✅ Standardized six-step sequential thinking approach across all analyzer components
- ✅ Counter-argument testing framework for all analysis steps
- ✅ Comprehensive test coverage for new components
- ✅ Setup script for configuring the research environment
- ✅ Complete researcher_agent.py update to utilize all modular components

### Next Development Steps:

- 🔄 Implementation of advanced LLM-powered content generation templates
- 🔄 Natural language interface for research query refinement
- 🔄 Multi-language support for global content strategy
- 🔄 Enhanced content trend prediction using time-series analysis
- 🔄 Development of content-specific SEO optimization module
- 🔄 Integration with social listening tools for audience sentiment analysis

## Getting Started

### Prerequisites

- Python 3.8+
- MongoDB
- OpenAI API key or Groq API key
- Brave Search API key (Premium tier)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/blog-accelerator-agent.git
   cd blog-accelerator-agent
   ```

2. Set up the research environment:
   ```
   python setup_research.py
   ```

3. Install dependencies:
   ```
   python setup_research.py --install-deps
   ```

4. Configure your API keys in the `.env` file.

## Usage

### Research Mode

```bash
python agents/researcher_agent.py path/to/your/blog_post.md --brave-api-key YOUR_BRAVE_KEY --openai-api-key YOUR_OPENAI_KEY
```

### Review Mode

```bash
python agents/reviewer_agent.py --stage factual --yaml blog_title_review_tracker.yaml
```

## Project Structure

```
blog-accelerator-agent/
├── tests/
│   ├── test_db.py
│   ├── test_yaml_guard.py
│   ├── test_file_ops.py
│   ├── test_researcher_agent.py
│   ├── test_reviewer_agent.py
│   ├── test_source_validator.py
│   ├── test_firecrawl_client.py
│   ├── test_industry_analysis.py
│   ├── test_solution_analysis.py
│   ├── test_paradigm_analysis.py
│   ├── test_audience_analysis.py
│   ├── test_analogy_generator.py
│   ├── test_visual_asset_collector.py
│   └── conftest.py
├── agents/
│   ├── research/
│   │   ├── industry_analysis.py
│   │   ├── solution_analysis.py
│   │   ├── paradigm_analysis.py
│   │   ├── audience_analysis.py
│   │   ├── analogy_generator.py
│   │   ├── visual_asset_collector.py
│   │   └── __init__.py
│   ├── utilities/
│   │   ├── db.py
│   │   ├── file_ops.py
│   │   ├── source_validator.py
│   │   ├── firecrawl_client.py
│   │   └── yaml_guard.py
│   ├── researcher_agent.py
│   └── reviewer_agent.py
├── api/
│   ├── main.py
│   └── endpoints/
│       ├── process.py
│       └── review.py
├── data/
│   ├── analogies/
│   ├── blacklist/
│   ├── firecrawl_cache/
│   ├── research_components/
│   ├── tracker_yaml/
│   ├── uploads/
│   └── visual_assets/
├── storage/
├── docker-compose.yml
├── setup_research.py
├── .env
└── README.md
```

## Development Status

### Dev - 2025-05-03
- ✅ Developed and deployed AI-driven research scoring system for quality assessment
- ✅ Implemented cross-component data sharing for improved research coherence
- ✅ Enhanced MongoDB integration with optimized schema for faster retrieval
- ✅ Added real-time content trend analysis using external API integrations
- ✅ Improved visual asset processing with auto-tagging and categorization
- ✅ Implemented dynamic research depth adjustment based on topic complexity
- ✅ Added support for custom research templates and workflows
- ✅ Integrated with Google Scholar for academic source validation
- ✅ Enhanced test suite with performance benchmarking and load testing
- ✅ Deployed containerized solution with Kubernetes orchestration

### Dev - 2024-08-25
- ✅ Implemented Visual Asset Collector component in `agents/research/visual_asset_collector.py`
- ✅ Added comprehensive visual search, categorization, and filtering functionality
- ✅ Integrated with Firecrawl client for image search and retrieval
- ✅ Implemented solution visuals (50 assets) and paradigm visuals (15 assets) collection
- ✅ Created metadata generation including captions, descriptions, and content connections
- ✅ Added comprehensive tests for the Visual Asset Collector component
- ✅ Standardized six-step sequential thinking approach across all analyzer components (industry, solution, paradigm, audience)
- ✅ Added counter-argument testing (step 6) to all analysis prompts
- ✅ Enhanced evidence collection and validation in all analyzer components
- ✅ Enhanced Industry Analyzer with user persona focus in `agents/research/industry_analysis.py` to better target specific user types mentioned in topics
- ✅ Updated test approach for Industry Analyzer to prioritize how the agent will work in production rather than test environment
- ✅ Improved test resilience by focusing on integration tests and added pytest-asyncio support

### Dev - 2024-08-15
- ✅ Implemented paradigm analysis module in `agents/research/paradigm_analysis.py`
- ✅ Implemented audience analysis module in `agents/research/audience_analysis.py`
- ✅ Implemented analogy generator in `agents/research/analogy_generator.py`
- ✅ Updated researcher_agent.py to use all modular components
- ✅ Created tests for new research components

### Dev - 2024-08-12
- ✅ Implemented API endpoints in `api/main.py`, `api/endpoints/process.py`, and `api/endpoints/review.py`
- ✅ Created FastAPI endpoints for blog upload and review processes
- 🔜 Working on deploying the full system with Docker

### Dev - 2024-08-11
- ✅ Implemented reviewer agent in `reviewer_agent.py`
- ✅ Created tests for reviewer agent operations
- ✅ Working on API endpoints next

### Dev - 2024-08-10
- ✅ Implemented file operations in `file_ops.py`
- ✅ Created tests for file operations
- ✅ Implemented researcher agent in `researcher_agent.py`

### Dev - 2024-08-09
- ✅ Implemented YAML validation guard in `yaml_guard.py`
- ✅ Created tests for YAML validation operations

### Dev - 2024-08-08
- ✅ Implemented MongoDB utilities in `db.py`
- ✅ Created tests for MongoDB operations

## Running Tests

```bash
pytest tests/
```

## Running the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

Visit http://localhost:8080/docs for the OpenAPI documentation.

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
MONGODB_URI=mongodb://localhost:27017
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
BRAVE_API_KEY=your_brave_key
FIRECRAWL_SERVER=http://localhost:4000
OPIK_SERVER=http://localhost:7000
```

# Research Component Details

## Industry Analysis
- Identifies 10+ critical challenges in the industry/system
- Analyzes risk factors, inefficiencies, costs, and bottlenecks
- Validates findings with authoritative sources

## Solution Analysis
- Generates 5-10 supporting arguments with evidence
- Generates 5-10 counter arguments with evidence
- Identifies key metrics for measuring progress

## Paradigm Analysis
- Maps historical paradigms related to the topic
- Analyzes transitions between paradigms
- Extracts lessons from historical examples
- Projects future paradigm possibilities

## Audience Analysis
- Identifies distinct audience segments
- Analyzes needs, pain points, and motivations
- Evaluates existing knowledge and expertise levels
- Recommends content strategies based on audience characteristics

## Analogy Generator
- Creates powerful analogies to explain complex concepts
- Evaluates and refines analogies for accuracy and clarity
- Provides visual representations for each analogy
- Searches for existing analogies in literature

## Visual Asset Collector
- Collects 50-100 solution-focused visual assets (diagrams, charts, etc.)
- Collects 10-20 paradigm-focused visual assets
- Categorizes visuals by type and relevance
- Intelligently filters for quality and topic alignment
- Generates captions and metadata
- Provides research connections and content placement suggestions
- Integrates with Firecrawl for efficient image search and retrieval

---

## 🛠️ Testing

Basic smoke tests can be run by:

```bash
docker exec -it blog-accelerator pytest
```

To run tests for specific components:

```bash
pytest tests/test_analogy_generator.py
```

---

## 📬 Contact

Built with ❤️ by Blog Accelerator Agent Team
