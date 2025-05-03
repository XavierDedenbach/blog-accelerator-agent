# Task Assignment: Blog Accelerator Agent

This document defines the detailed development tasks for agents working on the project. It includes responsibilities, requirements, and check-in expectations to ensure alignment with the PRD and backend architecture.

---

## 🧱 Codebase Structure

Include a `tests/` directory with pytest-based test coverage for core modules.

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
│   └── utilities/
│       ├── db.py
│       ├── file_ops.py
│       └── yaml_guard.py
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
├── .env
└── README.md
```

---

## 🔨 Task Breakdown

### 1. `researcher_agent.py`

* Parse topic metadata from new markdown
* Parse embedded image references
* Call Brave MCP for citations
* Store outputs to `review_files` and `media` in MongoDB
* Assign readiness score

### 2. `reviewer_agent.py`

* Handle `--stage` flags (`factual`, `style`, `grammar`)
* Validate stage order from `blog_title_review_tracker.yaml`
* Use Brave and Firecrawl as needed
* Write result markdown and update Mongo
* Update YAML state

### 3. `utilities/db.py`

* Connect to MongoDB via URI
* Version-aware operations: `insert_or_archive()`, `get_latest_version()`
* Handle blog metadata, review results, and media writes

### 4. `utilities/yaml_guard.py`

* Validate review step transitions
* Enforce order of operations (e.g., grammar only after style)
* Provide CLI and API helpers for updating YAML state

### 5. `utilities/file_ops.py`

* Extract asset folder from uploaded ZIP
* Read `.md` and copy local images
* Generate base64 for Mongo
* Detect updated versions by filename suffix

### 6. `api/main.py`

* Register FastAPI app and route groups
* Add health check endpoint

### 7. `api/endpoints/process.py`

* Accept upload (zip with markdown + images)
* Extract and validate asset folder
* Store contents in Mongo
* Trigger researcher agent pipeline

### 8. `api/endpoints/review.py`

* Accept YAML check-off via PUT
* Validate state using `yaml_guard`
* Trigger reviewer agent for current stage
* On success, update Mongo and YAML

---

## ✅ Engineering Process

### Check-ins

* Daily 2-sentence log in shared thread or README:

  * What was done
  * What’s blocked

### PR Rules

* Must include relevant unit tests for any function, API route, or utility added
* Must map to PRD section or backend spec
* Must show I/O examples or test stub
* Must not skip YAML or versioning validations
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

* No review step runs if previous incomplete (YAML enforced)
* Markdown + image upload must occur together (ZIP or multipart)
* No overwriting of older versions in Mongo
* Only latest version is reviewed

---

This task board is versioned and updated weekly.
