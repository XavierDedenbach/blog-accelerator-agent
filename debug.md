# Debugging Log Points

This document tracks the location and status of key debugging log points added to the Blog Accelerator Agent, primarily focusing on the `ResearcherAgent`.

**Status:**
*   `ACTIVE`: Log point is currently enabled.
*   `INACTIVE`: Log point is currently commented out or disabled.

## `run_researcher_with_env.py`

### `main()` function - Exception Handler

*   **Location:** `except Exception as e:` block within `main()`.
*   **Purpose:** Dumps the content of `agent.research_data` to the console if an uncaught exception occurs during `agent.process_blog()`.
*   **Trigger:** When `agent.process_blog()` raises an exception that isn't caught internally.
*   **Status:** `ACTIVE`

## `agents/researcher_agent.py`

### `gather_research()` - Inner Async Tasks

General logging added to the `try`, `except`, and `finally` blocks of each inner async task function (`citations_task`, `industry_task`, `solution_task`, `paradigm_task`, `audience_task`, `analogy_task`, `visual_asset_task`).

*   **Location:** Within each `async def task_name():` function inside `gather_research`.
*   **Purpose:** To track the success, specific errors (with traceback), and completion of each individual research component.
    *   Logs success + summary on successful execution within `try`.
    *   Logs detailed error + traceback on exception within `except`.
    *   Logs task completion within `finally`.
*   **Trigger:** During the execution of `gather_research`.
*   **Status:** `ACTIVE`

### `process_blog()` - Synchronous Steps

*   **Location:** Within the main `try...except` block of `process_blog`, specifically around the calls to `calculate_readiness_score`, `generate_research_report`, and `save_research_results`.
*   **Purpose:** To isolate failures occurring during the synchronous processing steps after `gather_research` has finished.
    *   Logs messages like "Attempting to calculate readiness score...", "Successfully calculated readiness score...", "Error occurred *during* readiness score calculation...".
*   **Trigger:** After `gather_research` returns and before `process_blog` returns its final result.
*   **Status:** `INACTIVE` (Previous attempt to add these failed, needs revisit if necessary). 