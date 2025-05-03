# User Flow: Blog Accelerator Agent

This document describes the step-by-step user journey and agent behaviors for the AI-powered Blog Accelerator application. It includes both the **Research Mode** and **Review Mode**, describing **User > Agent**, **Agent > Agent**, and **Agent > User** interactions in simple, specific terms.

---

## Pages and Interfaces

The app is **CLI or API-driven**, not a GUI. All user interactions happen by:

* Uploading or submitting markdown files (e.g., blog drafts)
* Creating a new topic via a CLI/API endpoint
* Receiving email notifications
* Reading markdown output reports

---

## 1. Research Mode User Flow

### 1.1 User Submits a New Topic

* The user sends a new topic title (e.g., via a Notion webhook or CLI).
* The topic must include:

  * A clear opinion (claim or stance)
  * A clear audience
  * The affected industry or system
* If missing, the agent responds asking for clarification.

**User → Agent**

```json
{
  "topic": "Why microgrids will replace rural utilities",
  "audience": "rural policy makers",
  "industry": "electric utilities"
}
```

---

### 1.2 Agent Decomposes the Topic

* Agent breaks the topic into subcategories:

  * Industry or system affected
  * Proposed solution
  * Current dominant paradigm
  * Audience assumptions

**Agent → Agent**

* Internal agent chain splits the topic and creates question trees.

---

### 1.3 Agent Researches Each Subtopic

* The following is done sequentially:

  1. **Industry Affected**

     * Identify top 3–5 systemic challenges
     * Retrieve sources
     * Log URLs and quotes
  2. **Proposed Solution**

     * Describe how the solution addresses each challenge
     * Gather metrics and progress data
     * Collect 50–100 visual assets (via Firecrawl MCP)
     * List weaknesses and counterpoints
  3. **Dominant Paradigm**

     * Show how the existing system works
     * When it was established
     * Collect visual examples
     * Compare to 2–3 alternative solutions
  4. **Audience Fit**

     * Identify knowledge gaps
     * List acronyms + definitions
     * Recommend cuts or added context
     * Generate analogies

**Agent → Firecrawl MCP**

* For finding public media assets and scraping context.

**Agent → Brave Search MCP**

* For sourcing citations, counterpoints, and news.

**Agent → Agent**

* Various internal steps call sub-agents for citation verification, content summarization, image filtering.

---

### 1.4 Agent Writes Research Report

* All findings go into a single markdown file.
* Images are stored in MongoDB (encoded or linked).
* The report includes:

  * Title, summary
  * Each subtopic with findings
  * Inline citations
  * Visual links or base64
  * Readiness score and rationale

**Agent → MongoDB**

* Save markdown, image references, readiness score

**Agent → User**

* Notify user via email or CLI: "Research is complete"
* Provide download link or file location

---

## 2. Review Mode User Flow

### 2.1 User Uploads or Updates Draft Blog

* A draft is saved in the `/review` folder.
* Triggered automatically by file creation or CLI/API call.
* If the blog is a new version of an existing post, the file should be named with a version suffix (e.g., `why-microgrids_v4.md`).
* The agent will:

  1. Detect the new version.
  2. Update the `current_version` number in `blog_title_review_tracker.yaml`.
  3. Invalidate any previous review files.
  4. Reset all `review_pipeline` stages to `complete: false`.
  5. Log the previous draft and reviews to a `version_history` section of the YAML.

**User → Agent**

* Adds or replaces blog markdown file
* Ensures version bump or relies on CLI tool to version automatically

---

### 2.2 Review Stage Progression via YAML

#### Mechanism

* Each blog post has a companion file: `blog_title_review_tracker.yaml`.
* It tracks the current state of each review stage and version history.
* The user progresses the process by editing that file **or using CLI commands**.
* When a new markdown file is uploaded with a different version suffix (e.g., `_v5.md`), the system automatically:

  * Increments the `current_version` in the YAML
  * Archives the current review pipeline under a new `version_history` block
  * Resets all `review_pipeline` stages to `complete: false`
  * Clears out previous result filenames
  * Begins the review process from stage 1

#### YAML Version History Example

* Each blog post has a companion file: `blog_title_review_tracker.yaml`.
* It tracks the current state of each review stage.
* The user progresses the process by editing that file **or using CLI commands**.

#### Example YAML

```yaml
blog_title: "why-microgrids-will-replace-utilities"
current_version: 4

version_history:
  - version: 3
    review_pipeline:
      factual_review:
        complete: true
        completed_by: "agent"
        result_file: "why-microgrids_review1_v3.md"
        timestamp: 2024-05-02T20:31:00Z
      style_review:
        complete: true
        completed_by: "agent"
        result_file: "why-microgrids_review2_v3.md"
        timestamp: 2024-05-02T20:35:00Z
      grammar_review:
        complete: true
        completed_by: "agent"
        result_file: "why-microgrids_review3_v3.md"
        timestamp: 2024-05-02T20:40:00Z
    final_release:
      complete: true
      released_by: "user"
      timestamp: 2024-05-02T21:00:00Z

review_pipeline:
  factual_review:
    complete: false
    completed_by: null
    result_file: null
    timestamp: null

  style_review:
    complete: false
    completed_by: null
    result_file: null
    timestamp: null

  grammar_review:
    complete: false
    completed_by: null
    result_file: null
    timestamp: null

final_release:
  complete: false
  released_by: null
  timestamp: null
```

#### Flow

1. User completes review stage by updating the corresponding key (e.g., `style_review.complete: true`).
2. Agent watches or polls the file.
3. Agent checks that all **prior stages are marked `complete: true`**.
4. Agent begins next stage, updates the YAML with `completed_by`, `result_file`, and `timestamp`.

---

### 2.3 Agent Stage 1: Factual Review

* Extracts all claims
* For each claim:

  * Finds 3 supporting sources
  * Finds 3 contradicting sources
  * Writes 100-word summary of each
  * Assigns a 1–10 consensus score
* Outputs a table:

  * claim, line number, pro sources, con sources, score

**Agent → Brave Search MCP**

* For gathering facts and perspectives

**Agent → MongoDB**

* Save the review file: `your-draft_review1.md`

**Agent → User**

* Email or message: "Factual review ready"
* Waits for approval (via YAML or CLI)

---

### 2.4 Agent Stage 2: Style Review

* Five AI reviewers analyze the post:

  * Packy McCormick, Edward Tufte, Paul Graham, Naval, Casey Handmer
* For each reviewer:

  * Compare writing section
  * Log disagreement: clarity, position, visuals, math
  * Rate severity (1–5)
* Output a table:

  * reviewer, section, issue type, comment, severity

**Agent → MongoDB**

* Save `your-draft_review2.md`

**Agent → User**

* Message: "Style review ready"
* Waits for user revision or approval (via YAML or CLI)

---

### 2.5 Agent Stage 3: Grammar Review

* Performs line-by-line grammar scan
* Tags issues as:

  * clarity
  * confidence
  * bad grammar
* Suggests inline fixes

**Agent → MongoDB**

* Save `your-draft_review3.md`

**Agent → User**

* Message: "Grammar review ready"
* Waits for approval (via YAML or CLI)

---

### 2.6 User Releases Blog

* If all 3 reviews are accepted, user marks blog as "released" in YAML.
* System logs final version.

**User → Agent**

* Marks complete via YAML or CLI:

```bash
./agent approve-review --stage final_release
```

**Agent → MongoDB**

* Version tagged as `released`
* Optionally synced to external CMS

---

## Notes on Agent Coordination

* **Agent orchestration** is powered by LangGraph (LangChain)
* **Web search**: Brave MCP with Firecrawl fallback
* **Citation filtering**: blacklist-aware agent
* **Memory and review**: MongoDB context across tasks
* **Visual logs**: Opik MCP interface for debug and flow review

---

## Summary of Interactions

| Flow Step         | User ↔ Agent | Agent ↔ Agent | Agent ↔ External    |
| ----------------- | ------------ | ------------- | ------------------- |
| Submit Topic      | ✅            | ✅             | ❌                   |
| Topic Breakdown   | ❌            | ✅             | ❌                   |
| Research Phase    | ❌            | ✅             | ✅ (Brave/Firecrawl) |
| Write Report      | ❌            | ✅             | ✅ (MongoDB)         |
| Receive Report    | ✅            | ❌             | ❌                   |
| Upload Blog Draft | ✅            | ❌             | ❌                   |
| Factual Review    | ❌            | ✅             | ✅ (Brave)           |
| Style Review      | ❌            | ✅             | ❌                   |
| Grammar Review    | ❌            | ✅             | ❌                   |
| Mark Release      | ✅            | ❌             | ❌                   |

---

This document will evolve as more UI and automation features (like the Notion webhook and dashboard) are implemented.
