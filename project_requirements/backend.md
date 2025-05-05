# Backend Architecture: Blog Accelerator Agent

This document describes the internal backend structure for the Blog Accelerator Agent. It includes database schema, API integrations, authentication logic, storage behaviors, and handling of edge cases.

---

## 1. Overview

The backend is built around:

* **FastAPI** (Python) as the main web server and API handler
* **MongoDB** for structured and unstructured content (blogs, images, metadata, review pipelines)
* **Docker** to containerize all services
* **LangGraph (LangChain)** to coordinate AI agents
* **Brave Search MCP** for fact-checking and citations
* **Firecrawl MCP** for image scraping and alt-data retrieval
* **Opik MCP** for agent observability/debugging

---

## 2. MongoDB Schema

### Image Upload Behavior

* When a user uploads a blog draft, they must upload both the markdown file and related image files in a single folder.
* The path to this folder is stored in the `asset_folder` field in the `blogs` collection.
* The backend parses the `.md` file to extract image links.
* Each referenced image is loaded from the same folder, encoded to base64, and stored in MongoDB under the `media` collection.
* This design allows the system to manage and version all content assets together.

### `blogs`

Includes a reference to an asset folder containing the markdown file and any image/media assets. These are uploaded together and tied to each blog version.

```json
{
  "_id": ObjectId,
  "title": "why-microgrids-will-replace-utilities",
  "current_version": 4,
  "asset_folder": "uploads/why-microgrids-v4/",
  "versions": [
    {
      "version": 4,
      "file_path": "review/why-microgrids_v4.md",
      "timestamp": "2024-05-02T20:10:00Z",
      "review_status": {
        "factual_review": { "complete": false, "result_file": null },
        "style_review": { "complete": false, "result_file": null },
        "grammar_review": { "complete": false, "result_file": null },
        "final_release": { "complete": false }
      },
      "readiness_score": null
    },
    // ...previous versions
  ]
}
```

### `review_files`

```json
{
  "blog_title": "why-microgrids-will-replace-utilities",
  "version": 4,
  "stage": "factual_review",
  "filename": "why-microgrids_review1_v4.md",
  "content": "...markdown...",
  "timestamp": "2024-05-02T20:31:00Z"
}
```

### `media`

```json
{
  "blog_title": "why-microgrids-will-replace-utilities",
  "version": 4,
  "type": "image",
  "source": "Firecrawl MCP",
  "url": "https://..."," +
    ""alt_text": "...",
  "stored_base64": "..."
}
```

---

## 3. Authentication

* **Local dev**: No auth
* **Production mode**: Token-based (JWT)
* Agents authenticate to FastAPI endpoints using internal shared secrets
* Only validated agents can write to `blogs` or `review_files`

---

## 4. Storage Rules

* Markdown files are **not stored in the filesystem** but in MongoDB
* Images are either:

  * Saved in base64 form
  * Linked via URL (with cache fallback)
* Each file is associated with `blog_title` and `version`
* Review stages **must match the current version** before triggering
* If user uploads a file with the same blog title but different version number:

  * Previous reviews are frozen
  * YAML is updated automatically

---

## 5. Agent Coordination Logic

### Agent → Agent

* LangGraph orchestrates task sequences (research → write → evaluate)
* Each agent writes logs to Opik MCP
* Sub-agents share context using shared memory via LangChain (e.g., VectorStore or Mongo)
* **Prompting Strategy:** Research agents employ sequential reflection within prompts, asking the LLM to consider core constraints before generating analysis to enhance nuance and depth.

### Sequential Thinking Implementation

* **Prompt Structure:** Each analysis prompt follows a stepwise structure:
  1. Identify core constraints relevant to the topic
  2. Consider systemic context
  3. Map stakeholder perspectives
  4. Identify challenges/solutions
  5. Generate supporting evidence
  6. Test counter-arguments

* **Agent Coordination:** Each analyzer (Industry, Solution, Paradigm) first runs constraint analysis before proceeding to main task
* **Memory Mechanism:** Constraints identified are stored and referenced in subsequent analysis
* **Enhanced MongoDB Schema:** Includes fields for storing constraint analysis alongside each research component

### Agent → Mongo

* All reads/writes go through a `db.py` utility
* All updates are version-guarded
* Each result file has a unique suffix

### Agent → FastAPI

* Blog processing tasks are routed via internal FastAPI endpoints:

  * `/process/new-topic`
  * `/review/factual`
  * `/review/style`
  * `/review/grammar`
  * `/review/approve`

### Agent → Brave / Firecrawl / Opik MCP

* Agents authenticate via API keys stored in environment variables
* Brave is used for source credibility
* Firecrawl is used for web scraping images and supporting media
* Opik MCP logs state transitions and message chains

---

## 6. Critical Edge Cases

### Case 1: User uploads new version before completing old review

* Solution:

  * New version becomes active
  * Old version reviews are archived
  * YAML is rewritten to reset state

### Case 2: Review proceeds before prior stage is complete

* Solution:

  * YAML validation checks `complete: true` on previous stage
  * If not set, block next stage

### Case 3: Invalid or malformed YAML

* Solution:

  * Reject updates via CLI/API with parse error
  * Agent logs alert to Opik MCP

### Case 4: Image scraping rate limits or 404s

* Solution:

  * Retry with exponential backoff
  * Fallback to Brave image search if Firecrawl fails

---

## 7. Debugging & Logs

* All agent interactions are logged to **Opik MCP**
* Each stage writes an execution trace
* Metadata about file status, versioning, and validation is stored in MongoDB and mirrored in Opik

---

## 7. Web UI for Research Reports

### API Endpoint

*   **Endpoint:** `/reports/view/{blog_title}/{version}`
*   **Method:** `GET`
*   **Description:** Serves the HTML page for viewing a specific research report.
*   **Logic:**
    *   Retrieves the research data (or pre-generated report markdown) for the specified `blog_title` and `version` from MongoDB.
    *   Renders an HTML template (e.g., using Jinja2) with the research data.
    *   Returns the rendered HTML page.

### Browser Integration

*   **Mechanism:** The script that initiates the research process (e.g., `run_researcher_with_env.py` or the main block in `agents/researcher_agent.py`) will use the standard Python `webbrowser` module.
*   **Workflow:**
    1.  After the `ResearcherAgent` successfully completes the `process_blog` method and saves the results, it returns a dictionary containing the status and the URL for the generated report (e.g., `http://localhost:8080/reports/view/your-blog-title/1`).
    2.  The calling script receives this result.
    3.  If the status is successful and a report URL is provided, the script calls `webbrowser.open(report_url)`.
    4.  This command attempts to open the user's default web browser to the specified URL.
*   **Fallback:** If `webbrowser.open()` fails (e.g., in a headless environment), the script should log the report URL to the console so the user can open it manually.

### Template Engine

*   `Jinja2` is recommended for templating due to its seamless integration with `FastAPI`.
*   Templates will reside in a `templates` directory.
*   CSS and **client-side JavaScript** files will be served as static assets by FastAPI.
*   The JavaScript will handle:
    *   **Filtering** the displayed report content based on user selections (e.g., showing only 'Audience Analysis' or 'Solution Analysis' sections) without requiring page reloads.
    *   Implementing a **client-side keyword search bar** to filter content textually within the browser and highlight matches.

### Security

*   The report viewing endpoint should initially be accessible only locally.
*   If deployed publicly, appropriate authentication (e.g., checking user session or JWT) should be added to protect access to potentially sensitive research data.

---

This backend document is versioned and will evolve as the project expands to include Notion integrations, UI, or file syncing.
