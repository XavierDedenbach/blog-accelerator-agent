# Project Kickoff Guide: Blog Accelerator Agent

This guide provides a clear, step-by-step plan to begin active development using Cursor, aligned with your PRD, backend design, and task structure.

---

## 🔧 Initial Bootstrapping

1. **Scaffold the project structure** (see `tasks.md` for full layout)
2. **Create function stubs** in each agent/module file:

   * Define placeholder methods and docstrings
   * Make dependencies explicit (e.g., Mongo, Brave, file paths)
3. **Add `.env` file** with dummy keys and Mongo URI
4. **Implement Opik MCP logging hook**

---

## 🧪 Start With Tests First

* Use `pytest` from day 1
* Each function/module must have a paired test in the `tests/` directory
* Example: `insert_or_archive()` should be tested with a mock DB

---

## 👥 Assign Tasks

Each developer/agent must:

* Pick a task from `tasks.md`
* Open a draft PR titled: `WIP: [module name]`
* Add checklist referencing specific PRD/backend.md items

---

## ⚙️ Agent Triggering Rules

Each agent should work independently via CLI:

* `python researcher_agent.py blog_title.md`
* `python reviewer_agent.py --stage factual --yaml blog_title_review_tracker.yaml`

FastAPI integration comes later.

---

## 📥 Daily Check-ins

Each developer posts the following daily in `README.md` or a dev thread:

```markdown
### [Name] — [Date]
- ✅ Completed parsing of YAML stage transitions
- 🛑 Blocked on inserting nested media into Mongo
```

---

## 🔁 PR + Review Requirements

* One PR per logical function or feature
* 2 approvals required (peer + architect)
* PR must include:

  * Linked spec from PRD or backend.md
  * Unit test coverage
  * Mongo/YAML snapshot or mock

---

## 🚀 Recommended Kickoff Order

| Module               | File                      | Priority               |
| -------------------- | ------------------------- | ---------------------- |
| MongoDB utils        | `db.py`                   | ✅ First                |
| YAML validation      | `yaml_guard.py`           | ✅ First                |
| Markdown parser      | `file_ops.py`             | ⏩ Second               |
| Researcher interface | `researcher_agent.py`     | ⏩ Second               |
| Unit test scaffolds  | `tests/`, `conftest.py`   | ✅ Parallel             |
| Reviewer logic       | `reviewer_agent.py`       | 🕓 After YAML complete |
| Upload/review API    | `process.py`, `review.py` | 🔜 Later               |

---

This document ensures consistent and modular execution across all agents and contributors.
