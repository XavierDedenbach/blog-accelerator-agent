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
