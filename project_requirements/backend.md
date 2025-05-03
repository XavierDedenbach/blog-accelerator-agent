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

This backend document is versioned and will evolve as the project expands to include Notion integrations, UI, or file syncing.
