# Blog Accelerator Agent

A Python application that accelerates blog content development through automated research and review processes.

## Project Structure

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

## Development Status

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

# blog-accelerator-agent

AI-assisted research and review tooling for thoughtful, human-written blog posts. This agent supports content creators by streamlining deep research and factual/style/grammar review, without generating the writing itself.

---

## 🧠 What It Does

* **Research Mode**: Breaks down opinionated topics, gathers structured analysis, and outputs rich markdown reports.
* **Review Mode**: Performs fact-checking, stylistic review by multiple AI personas, and grammar refinement before publication.

> ⚠️ No AI-written blog content. This tool is for augmentation, not automation.

---

## 🚀 Getting Started

1. Clone the repo
2. Create a `.env` file based on `.env.example`
3. Run the system with:

```bash
docker-compose up --build
```

4. Access services:

   * API: `http://localhost:8080`
   * Opik MCP: `http://localhost:7000`
   * Firecrawl MCP: `http://localhost:4000`
   * MongoDB: `mongodb://localhost:27017`

---

## 📄 Documentation

For full architecture, workflows, APIs, and feature breakdown, see the [Product Requirements Document (PRD)](./docs/PRD.md)

---

## ✅ Tasks in Progress

* [ ] Notion webhook for topic intake
* [ ] Structured analogy generation
* [ ] Reviewer personas: Packy, Tufte, Naval, etc.
* [ ] Consensus scoring in fact-check tables
* [ ] Image & infographic auto-fetching via Firecrawl MCP

---

## 🧪 Development Tips

* FastAPI backend runs inside Docker (`blog-agent`)
* Use Opik MCP to inspect agent thought chains and trace logs
* Store all content (text + images) directly in MongoDB
* Logs are saved under `./logs/`

---

## 🛠️ Testing

Basic smoke tests can be run by:

```bash
docker exec -it blog-accelerator pytest
```

> For full E2E flow testing, mock blog uploads to `./review/` and trigger pipeline endpoints.

---

## 📬 Contact

Built with ❤️ by [Xavier Dedenbach](https://github.com/xdede)
