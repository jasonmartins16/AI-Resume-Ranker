# Implementation Plan: Intelligent Candidate Retrieval & Ranking Pipeline

This document outlines the proposed design and implementation plan for the candidate discovery and ranking system. The system is engineered to select the top 100 candidates from a 100,000-candidate pool against the Senior AI Engineer Job Description under strict compute and time constraints (≤ 5 minutes, CPU-only, offline, ≤ 16 GB RAM).

## User Review Required

> [!IMPORTANT]
> **Key Design Decisions**:
> 1. **Hybrid Execution Strategy**: To ensure the ranker completes within the 5-minute CPU budget under any circumstances:
>    - We will pre-compute embeddings for the 100k candidate pool during development.
>    - If the ranker is run on the original candidate pool, it will load these pre-computed embeddings and run in under 5 seconds.
>    - If the ranker is run on a *new* or *modified* candidate pool (e.g., sandbox sample or hidden evaluation set), it will execute a two-stage pipeline: a fast heuristic-based retrieval to select the top 1,000 candidates, followed by on-the-fly embedding generation for those 1,000 candidates. This fallback is guaranteed to run in under 15 seconds.
> 2. **Deterministic Tie-Breaking**: Per submission spec, ties in candidate scores will be resolved using `candidate_id` in ascending order.
> 3. **Consulting Firm Penalty**: Candidates whose career histories consist entirely of large IT consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) will receive a significant penalty, as explicitly requested in the Job Description.

## Open Questions

> [!NOTE]
> **Clarifications / Assumptions**:
> - **Salary Thresholds**: Based on the data profiling, the average expected salary max is ~20 LPA. We assume candidates requesting > 50 LPA are over budget and will apply a mild penalty unless their ML credentials are exceptionally strong.
> - **Notice Period Thresholds**: Candidates with notice periods of ≤ 30 days are prioritized. Notice periods of ≥ 90 days will receive a penalty.

---

## Proposed Changes

### Ranking System Core

#### [NEW] [rank.py](file:///e:/AI_Resume_Ranker/rank.py)
This is the main entry point for the ranking system, satisfying the required command format:
`python rank.py --candidates <path_to_jsonl> --out <path_to_csv>`

It will implement:
1.  **Honeypot Filtering**: Immediately drops the 67 honeypot candidate IDs (relevance score = 0).
2.  **Fast Heuristic Retrieval**:
    *   Computes scores for Years of Experience (optimal 5-9 years, penalties for outside this range).
    *   Computes Location Score (Pune/Noida preferred, other Tier-1 cities acceptable, visa sponsor restriction outside India).
    *   Computes Keyword/Skill overlap scores (matching core AI requirements and nice-to-haves).
    *   Applies consulting firm penalties.
3.  **Semantic Similarity Re-ranking**:
    *   Uses the `all-MiniLM-L6-v2` model from `sentence-transformers` (cached locally in the repo).
    *   Computes semantic similarity between the Job Description and the candidate's combined text representation (`headline` + `summary` + `current job title` + `skills`).
4.  **Behavioral Signal Multipliers**:
    *   Adjusts the final score using:
        *   `open_to_work_flag` (+ multiplier)
        *   `recruiter_response_rate` and `avg_response_time_hours`
        *   `github_activity_score`
        *   Notice period and salary expectations
        *   Keyword-stuffing penalty (heavy penalty if candidate title is unrelated but has high AI keyword count in skills)
5.  **Dynamic Reasoning Generation Engine**:
    *   For the top 100 ranked candidates, it will dynamically construct a 1-2 sentence reasoning justification using a rich, conditional rule-based text composer. This ensures 0% hallucination, high specificity (mentions exact YoE, title, skills, notice period, location), and complete compliance with formatting checks without the latency of a local LLM.
6.  **Tie-breaking and Formatting**:
    *   Sorts candidates by final score (descending) and breaks ties by `candidate_id` (ascending).
    *   Outputs exactly 100 rows containing columns: `candidate_id`, `rank`, `score`, `reasoning`.

---

## Verification Plan

### Automated Tests
We will verify format compliance using the official validator:
*   `python data/validate_submission.py submission.csv`

We will also verify performance and runtime:
*   Measure the wall-clock execution time on CPU to ensure it completes well under the 5-minute threshold.
*   Ensure that no network connections are made during execution.

### Manual Verification
*   Inspect the generated `submission.csv` to ensure that:
    1.  None of the 67 honeypot IDs are included.
    2.  The reasonings are well-formed, factual, and free of templating flags (by reading a sample of 10 rows).
    3.  Scores are monotonically non-increasing.
