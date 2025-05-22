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
* **TODO**: Implement the enhanced readiness score algorithm based on nuanced analysis requirements
* **Implementation Details**:
  * Revise scoring function in `researcher_agent.py` with the following improvements:
    * Reduce base score from 50 to 30 points
    * Implement category-specific evaluators for visual assets, systemic thinking, citations, etc.
    * Create specialized evaluators for solution nuance and audience benefit clarity
    * Add quality gates requiring minimum 30% in all categories for B grade
    * Add automatic C grade (or lower) for missing visual assets or limited systemic thinking
    * Adjust grade thresholds to ensure A grades are reserved for truly exceptional content
  * New scoring criteria:
    * Grade C or below (≤ 70%) for content with any missing critical components
    * Grade B (71-85%) for content with minimum 30% in EACH category
    * Grade A (86-100%) requiring rich visual assets, strong systemic thinking, proper citations,
      nuanced solution research, and clear articulation of audience benefits
  * Add comprehensive feedback component that explains grade and provides improvement guidance
  * Create detailed test cases to validate scoring accuracy
* **Expected Output**: Enhanced function that returns more accurate letter grades with detailed explanation and improvement recommendations

### 7. API Documentation and Testing
* **Already Implemented**: Basic API endpoints
* **TODO**: Document API structure and add comprehensive tests
* **Implementation Details**:
  * Add OpenAPI documentation to all endpoints
  * Create test suite for API routes
  * Implement mock responses for testing
  * Add validation for request/response schemas
* **Expected Output**: Documented API with test coverage

### Task 8: Transition to Dockerized Environment and Production Optimization

*   **Objective:** Fully transition the Blog Accelerator Agent and its dependent services (MongoDB, Firecrawl, Opik) to a Dockerized environment using Docker Compose. Optimize the Docker setup for development efficiency and prepare for production deployment. Ensure all components function correctly within this containerized setup.

*   **8.1: Verify and Refine `blog-agent` Dockerfile**
    *   **Responsibility:** Agent
    *   **Details:**
        *   Review the existing `Dockerfile` for the `blog-agent` service.
        *   Implement multi-stage builds to minimize the final image size (e.g., separate build stage for installing dependencies and a smaller runtime stage).
        *   Ensure `requirements.txt` is efficiently copied and dependencies are installed correctly.
        *   Optimize layer caching.
        *   Verify that the application starts correctly within the container (e.g., FastAPI server runs on the specified port).
    *   **Expected Output:** An optimized `Dockerfile` for the `blog-agent` that produces a smaller, more efficient image.

*   **8.2: Enhance `docker-compose.yml` for Robustness and Inter-Service Communication**
    *   **Responsibility:** Agent
    *   **Details:**
        *   Confirm that all service names (`mongo`, `opik`, `firecrawl`) are correctly used by the `blog-agent` service for internal communication (e.g., `MONGODB_URI=mongodb://mongo:27017`).
        *   Add health checks for `mongo`, `opik`, and `firecrawl` services to ensure `blog-agent` only starts or attempts to connect when its dependencies are ready.
        *   Review and confirm volume mounts for persistent data (`mongo_data`, `./logs`, `./data`) are correctly configured and functional.
        *   Ensure graceful shutdown: Add proper signal handling in the `blog-agent` (if not already present) and ensure `docker-compose down` stops containers cleanly.
    *   **Expected Output:** An updated `docker-compose.yml` with health checks, verified service networking, and robust data persistence.

*   **8.3: Application Configuration for Dockerized Services**
    *   **Responsibility:** Agent
    *   **Details:**
        *   Audit the `ResearcherAgent` and any other relevant parts of the codebase (e.g., API clients, utility modules) to ensure they correctly use the environment variables for service URLs (e.g., `FIRECRAWL_SERVER`, `OPIK_SERVER`, `MONGODB_URI`) passed by `docker-compose.yml`.
        *   Ensure that default values in the code (if any) do not override these environment variables when running in Docker.
        *   Modify `run_researcher_with_env.py` or provide instructions if specific environment variables need to be set differently when running via `docker exec` or if it's intended primarily for local runs outside Docker.
    *   **Expected Output:** Application code that seamlessly connects to services using Docker Compose service discovery.

*   **8.4: Environment Variable Management and Security**
    *   **Responsibility:** Agent (guidance), User (implementation)
    *   **Details:**
        *   Confirm that all necessary environment variables (API keys, service URLs) are defined in `.env` and correctly passed to the `blog-agent` service via the `environment` section in `docker-compose.yml`.
        *   **Agent:** Will list any missing URLs or API keys based on the current codebase and `docker-compose.yml` during this task execution.
        *   **User:** Will ensure `.env` file is populated with the actual secrets.
        *   Advise on best practices for not committing `.env` files and using a `.env.example` for guidance.
    *   **Expected Output:** Clear documentation and setup for managing environment variables securely, and an updated `docker-compose.yml` if `BRAVE_API_KEY` and `OPENROUTER_API_KEY` need to be passed.

*   **8.5: Resource Limit Configuration (Placeholder for Production)**
    *   **Responsibility:** Agent (guidance), User (future implementation for production)
    *   **Details:**
        *   Research and recommend sensible default resource limits (CPU, memory) for each service in `docker-compose.yml`.
        *   Explain that these might need tuning based on actual usage in a production environment.
    *   **Expected Output:** Guidance on how to configure resource limits in `docker-compose.yml` for future production deployment.

*   **8.6: Comprehensive Testing in Dockerized Environment**
    *   **Responsibility:** User (execution), Agent (support)
    *   **Details:**
        *   User to run `docker-compose up --build -d` followed by thorough testing of all `ResearcherAgent` functionalities and API endpoints.
        *   User to verify database interactions, Firecrawl calls, and Opik logging all work correctly within the Docker network.
        *   User to report any issues encountered.
        *   Agent to assist in debugging any issues that arise from the Dockerization process.
    *   **Expected Output:** Confirmation that the entire system operates correctly within the Docker Compose setup.

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

### Task 12: Update Testing for Paradigm Analysis Module

**Objective:** Enhance the testing suite for the `paradigm_analysis` module to match the robustness and structure of the `industry_analysis` tests.

**Details:**
- Review `tests/agents/research/test_industry_analysis.py` and the `agents/research/industry_analysis.py` module to understand the current testing standards, including unit tests, integration tests, mocking strategies, and API key handling for integration tests.
- Implement similar comprehensive tests for the `paradigm_analysis` module and its corresponding test file (e.g., `tests/agents/research/test_paradigm_analysis.py`).
- **Key Focus Areas:**
    - Ensure proper mocking of LLM calls and external services (like `SourceValidator` if used) for unit tests. Refer to fixtures like `mock_llm`, `mock_source_validator`, and `industry_analyzer` in `test_industry_analysis.py` and the `mock_json_parser_parse_result_side_effect` for handling LLM outputs.
    - Implement an integration test that uses real API calls (similar to `test_analyze_industry_integration_real_llm`).
    - Address potential API key loading issues by referring to the `conftest.py` setup (`load_env_vars_and_debug` fixture for loading `.env` and the non-autouse `mock_env_variables` fixture for unit test isolation). Ensure integration tests correctly use real keys and unit tests use mocks or placeholders.
    - Ensure test cleanup, clear commenting, and robust error handling in tests.
- **Reference Files:**
    - `tests/agents/research/test_industry_analysis.py`
    - `agents/research/industry_analysis.py`
    - `tests/conftest.py` (for fixture patterns)
- **Goal:** Achieve a similar level of test coverage and reliability as demonstrated in the `industry_analysis` module.

### Task 13: Update Testing for Solution Analysis Module

**Objective:** Enhance the testing suite for the `solution_analysis` module, aligning with the standards set by `industry_analysis`.

**Details:**
- Review `tests/agents/research/test_industry_analysis.py` and `agents/research/industry_analysis.py` for best practices in testing structure, mocking, and integration test design.
- Apply these practices to the `solution_analysis` module and its test file.
- **Key Focus Areas:**
    - Implement comprehensive unit tests with effective mocking of LLMs and any external services. See `test_industry_analysis.py` for examples of `AsyncMock`, `MagicMock`, `patch`, and custom side effect functions for parsers.
    - Develop an integration test that interacts with live APIs, ensuring correct API key management as resolved for `test_analyze_industry_integration_real_llm`. Consult `tests/conftest.py` for the `load_env_vars_and_debug` and `mock_env_variables` fixture patterns.
    - Ensure tests are well-commented and debug prints (if any were used during development) are cleaned up or commented out.
- **Reference Files:**
    - `tests/agents/research/test_industry_analysis.py`
    - `agents/research/industry_analysis.py`
    - `tests/conftest.py`
- **Goal:** Elevate the `solution_analysis` tests to mirror the quality and thoroughness of the `industry_analysis` tests.

### Task 14: Update Testing for Analogy Generator Module

**Objective:** Refactor and improve the tests for the `analogy_generator` module based on the `industry_analysis` testing model.

**Details:**
- Study the testing approach in `tests/agents/research/test_industry_analysis.py` and the structure of `agents/research/industry_analysis.py`.
- Update the `analogy_generator` tests to incorporate similar unit and integration testing strategies.
- **Key Focus Areas:**
    - For unit tests: Ensure robust mocking of LLM interactions. The use of `AsyncMock` with custom `side_effect` functions to simulate various LLM responses (including error cases and malformed JSON) in `test_industry_analysis.py` is a good reference.
    - For integration tests: Create tests that make real API calls. Pay close attention to the API key loading mechanism (`load_env_vars_and_debug` in `conftest.py`) and how placeholder keys are managed for unit tests (`mock_env_variables` fixture).
    - Add clear comments and docstrings to tests.
- **Reference Files:**
    - `tests/agents/research/test_industry_analysis.py`
    - `agents/research/industry_analysis.py`
    - `tests/conftest.py`
- **Goal:** Ensure the `analogy_generator` module has a comprehensive and reliable test suite.

### Task 15: Update Testing for Audience Analysis Module

**Objective:** Modernize the testing for the `audience_analysis` module, using `industry_analysis` as a template for comprehensive testing.

**Details:**
- Analyze the testing patterns in `tests/agents/research/test_industry_analysis.py` and the corresponding module `agents/research/industry_analysis.py`.
- Revamp the tests for `audience_analysis` to include thorough unit and integration tests.
- **Key Focus Areas:**
    - Unit tests should feature detailed mocking of LLM responses, covering successful outputs, errors, and edge cases. The `mock_json_parser_parse_result_side_effect` in `test_industry_analysis.py` is a useful pattern for testing JSON parsing from LLM outputs.
    - Integration tests should connect to real APIs, with careful management of API keys. The solution involving `load_env_vars_and_debug` (autouse session fixture for `.env`) and `mock_env_variables` (non-autouse session fixture for mock values) in `tests/conftest.py` should be replicated.
    - Ensure high readability of test code through comments and clear structure.
- **Reference Files:**
    - `tests/agents/research/test_industry_analysis.py`
    - `agents/research/industry_analysis.py`
    - `tests/conftest.py`
- **Goal:** Bring the `audience_analysis` test suite up to the same high standard as `industry_analysis`.

### Task 16: Debug and Fix Researcher Agent Failures (Solution, Audience, Paradigm Analyzers)

**Objective:** Resolve runtime errors in the `ResearcherAgent` related to `SolutionAnalyzer` argument passing, JSON parsing failures in `AudienceAnalyzer` and `ParadigmAnalyzer`, and orchestrate dependent tasks.

**Details:**
- **Issue 1: `SolutionAnalyzer` Missing Arguments & Task Orchestration**
    - **File:** `agents/researcher_agent.py`
    - **Problem:** `TypeError` due to `analyze_solution` being called without `topic` and `challenges`. Also, `ParadigmAnalyzer` needs context from `IndustryAnalyzer`.
    - **Fix:** Modify the `gather_research` method in `ResearcherAgent`:
        1. Run `industry_analyzer.analyze_industry` first and await its completion.
        2. Extract `challenges` (list of names) and the full `industry_context` from the industry analysis results.
        3. Pass the `topic` and extracted `challenges` list to `solution_analyzer.analyze_solution`.
        4. Pass the `topic` and full `industry_context` to `paradigm_analyzer.analyze_paradigms`.
        5. Run `solution_analyzer`, `paradigm_analyzer`, and other independent LLM tasks (audience, analogy) concurrently after industry analysis is complete.
        6. Ensure `visual_asset_collector.collect_visuals` runs last, using results from the preceding dependent tasks.

- **Issue 2: JSON Parsing Errors in `AudienceAnalyzer` & `ParadigmAnalyzer` (and ensure for `SolutionAnalyzer`)**
    - **Files:** `agents/research/audience_analysis.py`, `agents/research/paradigm_analysis.py`, `agents/research/solution_analysis.py`
    - **Problem:** `json.JSONDecodeError` because LLM responses include non-JSON text (e.g., `<think>` blocks or are not purely JSON).
    - **Fix:**
        1. Implement a robust `_parse_llm_response_to_json(self, response_text: str, context: str) -> Any` helper method in `AudienceAnalyzer`, `ParadigmAnalyzer`, and verify/enhance in `SolutionAnalyzer`.
        2. This method should use regex to:
            a. Prioritize extracting content from ```json ... ``` markdown blocks.
            b. If no markdown block, attempt to strip common leading/trailing non-JSON text (e.g., `<think>...</think>` blocks).
            c. Log extensively for debugging.
        3. Replace all direct `json.loads()` calls on raw LLM responses in these analyzers with calls to this new helper method.
        4. Ensure appropriate error handling (e.g., raising analyzer-specific exceptions) if parsing still fails.
        5. For `ParadigmAnalyzer`, update `identify_paradigms_prompt` and relevant methods to accept and use `industry_context`.

**Reference Log Snippets:**
- SolutionAnalyzer: `TypeError: SolutionAnalyzer.analyze_solution() missing 2 required positional arguments: 'topic' and 'challenges'`
- AudienceAnalyzer/ParadigmAnalyzer: `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

**Goal:** Ensure the `ResearcherAgent` can complete its analysis runs without TypeErrors or JSON parsing errors, with improved contextual data flow between analyzers, leading to more stable and reliable research generation.

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
