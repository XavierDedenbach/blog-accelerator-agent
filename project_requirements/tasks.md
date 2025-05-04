# Task Assignment: Blog Accelerator Agent

This document defines detailed development tasks for implementing the Blog Accelerator Agent according to the PRD and researcher agent architecture specifications. Tasks are organized by priority and reflect the current implementation status.

---

## 🧱 Codebase Structure

Current structure with necessary additions:

```
blog-accelerator-agent/
├── tests/
│   ├── test_db.py
│   ├── test_yaml_guard.py
│   ├── test_file_ops.py
│   ├── test_researcher_agent.py
│   ├── test_reviewer_agent.py
│   └── conftest.py
├── agents/
│   ├── researcher_agent.py
│   ├── reviewer_agent.py
│   ├── research/
│   │   ├── industry_analysis.py
│   │   ├── solution_analysis.py
│   │   ├── paradigm_analysis.py
│   │   ├── audience_analysis.py
│   │   ├── analogy_generator.py
│   │   └── visual_asset_collector.py
│   └── utilities/
│       ├── db.py
│       ├── file_ops.py
│       ├── yaml_guard.py
│       ├── source_validator.py
│       └── firecrawl_client.py
├── api/
│   ├── main.py
│   └── endpoints/
│       ├── process.py
│       └── review.py
├── data/
│   ├── uploads/
│   └── tracker_yaml/
├── storage/
├── docker-compose.yml
├── run_researcher_with_env.py
├── .env
└── README.md
```

---

## 🔨 Priority Research Agent Tasks

### 1. Industry Analyzer Refinement
* **Already Implemented**: Base sequential thinking, 10+ challenges, source finding
* **TODO**: Enhance user constraint focus in prompts to better target specific personas
* **Implementation Details**: Modify `identify_challenges_prompt` to emphasize the specific user types/roles who would be affected by the challenges rather than just general industry challenges
* **Specific Changes**: Add explicit instruction to tailor challenges to user roles mentioned in the topic (e.g., "policy makers", "developers") within the prompt's Step 1 reflection section

### 2. Integration of Sequential Thinking Across All Analyzers
* **Already Implemented**: Most analyzers have some form of sequential thinking in prompts
* **TODO**: Standardize the six-step sequential thinking approach across all components
* **Implementation Details**: Ensure all analysis prompts follow the standard pattern:
  1. Identify core constraints
  2. Consider systemic context
  3. Map stakeholder perspectives
  4. Identify challenges/solutions
  5. Generate supporting evidence
  6. Test counter-arguments
* **Specific Changes**: Audit all prompt templates and add any missing steps, particularly step 6 (counter-arguments testing)

### 3. Visual Asset Collector Implementation
* **TODO**: Create the visual asset collector component
* **Implementation Details**: 
  * Create `agents/research/visual_asset_collector.py` class
  * Implement methods for collecting 50-100 solution visuals and 10-20 paradigm visuals
  * Add Firecrawl integration for image search and retrieval
  * Implement categorization and filtering of images
* **Expected Output**: A component that returns categorized, filtered visual assets with metadata

### 4. Researcher Agent Integration Improvements
* **Already Implemented**: Basic API fallback, rate limiting
* **TODO**: Strengthen orchestration of all analyzer components
* **Implementation Details**:
  * Update `researcher_agent.py` to properly sequence all analysis steps
  * Add progress tracking/reporting
  * Ensure all sequential thinking steps are preserved through the process
  * Add detailed logging for each analysis phase
* **Specific Changes**: Implement proper component coordination in `analyze_topic()` method

### 5. MongoDB Integration Enhancement
* **Already Implemented**: Basic DB operations
* **TODO**: Enhance schema for research data with sequential thinking artifacts
* **Implementation Details**:
  * Update DB schema to store intermediate reasoning steps
  * Add fields for constraint analysis results
  * Create indexes for efficient retrieval
  * Implement proper versioning for research results
* **Expected Output**: Enhanced DB schema with comprehensive research data storage

### 6. Readiness Score Calculation Implementation
* **TODO**: Implement the readiness score algorithm based on PRD requirements
* **Implementation Details**:
  * Create scoring function in `researcher_agent.py`
  * Score components: challenges count, pro/con arguments, visual assets count, analogies count
  * Calculate A-F grade based on minimum requirements from PRD
  * Add detailed breakdown of score components
* **Expected Output**: Function that returns a letter grade with score breakdown

### 7. API Documentation and Testing
* **Already Implemented**: Basic API endpoints
* **TODO**: Document API structure and add comprehensive tests
* **Implementation Details**:
  * Add OpenAPI documentation to all endpoints
  * Create test suite for API routes
  * Implement mock responses for testing
  * Add validation for request/response schemas
* **Expected Output**: Documented API with test coverage

### 8. Docker Container Optimization
* **Already Implemented**: Basic Docker setup
* **TODO**: Optimize Docker configuration for production use
* **Implementation Details**:
  * Implement multi-stage builds to reduce image size
  * Add health checks and proper signal handling
  * Configure resource limits
  * Set up proper volume mounts for persistent data
  * Ensure all environment variables are properly handled
* **Expected Output**: Production-ready Docker configuration

### 9. Sequential Thinking Debug Dashboard
* **TODO**: Create debug visualization for sequential thinking steps
* **Implementation Details**:
  * Add detailed logging of each sequential thinking step
  * Create simple web dashboard to visualize the reasoning process
  * Show the progression from constraints to conclusions
  * Add capability to export reasoning chains
* **Expected Output**: Debug interface showing sequential reasoning steps

### 10. End-to-End Testing Suite
* **TODO**: Create comprehensive test suite with sample topics
* **Implementation Details**:
  * Create 5-10 representative test topics
  * Implement automated tests that run the full research pipeline
  * Add assertions to verify output quality and completeness
  * Create CI/CD pipeline for continuous testing
* **Expected Output**: Test suite that validates full research workflow

### 11. Web-Based Research Report Viewer
* **TODO**: Create a browser-based UI for viewing research reports
* **Implementation Details**:
  * Create a simple web server endpoint to serve HTML/CSS/JS for report viewing
  * Implement MongoDB data retrieval and formatting for web display
  * Add functionality to open reports in browser automatically after generation
  * Create clean, readable UI with proper formatting of research components
  * Implement filters and search functionality for large reports
* **Expected Output**: Browser tab automatically opens with formatted research report after processing completes
* **Integration Points**: Update `run_researcher_with_env.py` to launch browser, add new endpoint to API server

---

## 📝 Future Reviewer Agent Tasks (For Later)

### 1. Review Agent Framework
* Set up the base reviewer agent structure
* Implement YAML-based review state tracking
* Create version management for blog revisions

### 2. Factual Review Component
* Implement claim extraction
* Create source finding (3 supporting + 3 contradicting)
* Build consensus score calculation

### 3. Style Review Component
* Implement multi-persona review framework
* Create disagreement categorization
* Build severity assessment

### 4. Grammar Review Component
* Implement line-by-line grammar analysis
* Create fix recommendation generator
* Build grammar report formatter

---

## ✅ Engineering Process

### Check-ins
* Daily 2-sentence log in shared thread or README:
  * What was done
  * What's blocked

### PR Rules
* Must include relevant unit tests for any function, API route, or utility added
* Must map to PRD section or backend spec
* Must show I/O examples or test stub
* Must not skip YAML or versioning validations

### Review Flow
* 2 approvers minimum:
  * 1 peer (agent dev)
  * 1 architect (you)
* Snapshot of Mongo or YAML state required

### Logging
* All agents log to Opik MCP
* Logs must include timestamps and file versions

---

## 🧪 Critical Guardrails
* Sequential thinking must be applied in all analysis components
* Visual assets must be properly stored and categorized
* All research components must have appropriate citations
* Readiness score calculation must follow defined criteria

---

This task board is versioned and updated weekly.
