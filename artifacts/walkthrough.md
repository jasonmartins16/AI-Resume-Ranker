# Verification and Walkthrough: Candidate Discovery & Ranking System

We have successfully built, tested, and validated the intelligent candidate discovery and ranking system. The pipeline ranks a pool of 100,000 candidates against a Senior AI Engineer Job Description (JD) and generates the top 100 matches in under 3 minutes on CPU, adhering to all strict formatting, business logic, and compute constraints.

---

## 1. Methodology & Design Overview

To handle 100,000 candidates in under 5 minutes offline on a single CPU, we designed a **Hybrid Screening and Semantic Ranking Pipeline**:

### A. Honeypot Filtering
We identified 709 malicious or corrupted candidate profiles (honeypots) using three structural rules:
1. **Timeline Inconsistencies**: The total Years of Experience (YoE) listed in the profile exceeds the actual career duration (first job start to present day 2026-06-18) by more than 2.0 years.
2. **Proficiency/Duration Anomalies**: A candidate claims 3 or more expert or advanced proficiency skills with `duration_months == 0`.
3. **Skill Overflow**: Any individual skill duration exceeds the total candidate YoE by more than 3.0 years.

### B. High-Precision Heuristic screening
For fast ranking, all candidates are first graded using a multi-factor heuristic score:
* **Experience Fit (35%)**: Optimum score for 5–9 years of experience, with linear penalties below and gradual decay above.
* **Location Fit (15%)**: Noida/Pune gets maximum preference. Relocation options from Tier-1 Indian cities are supported. Outside India is heavily penalized.
* **Skill Match (30%)**: Matches core search, retrieval, ML, and Python skills listed in the candidate profiles.
* **Career Description Scan (Bonus)**: Scans job description fields in `career_history` for key search & recommendation keywords to recognize candidates with hands-on experience who might not have listed specific keywords in their skills list.
* **Role Fit (20%)**: Evaluates match of the current title and headline to developer, data scientist, or ML titles.
* **Consulting Firm Penalty**: Candidates whose entire history is consulting-only (e.g. TCS, Infosys, Wipro, Accenture) receive a 50% penalty multiplier. Candidates with partial consulting background receive a 20% penalty.

### C. Local SentenceTransformer Encoding
To ensure the pipeline completes in under 3 minutes, candidates are sorted by their heuristic score, and the top 1,000 candidates are selected for detailed semantic evaluation.
* The local model (`all-MiniLM-L6-v2`) is loaded offline.
* Profile text (headline, summary, skills, and top 2 job descriptions) is encoded for these 1,000 candidates.
* Semantic similarity is computed as the cosine similarity between candidate embedding and the job description embedding.

### D. Behavioral Multipliers
The candidate composite score is scaled by engagement signals:
* **Open to Work Flag**: +5% bonus.
* **Recruiter Response Rate**: Up to +10% bonus.
* **Active Recency**: Up to +5% bonus for activity within 30 days; up to -15% penalty for inactivity over 180 days.
* **Notice Period**: +5% bonus for short notice (≤ 30 days); -10% penalty for long notice (≥ 90 days).
* **Expected Salary**: -5% penalty for high expected salary (> 50 LPA).

---

## 2. Validation & Verification

### Verification Runs

We verified the pipeline at different scales:

1. **Test Dataset (200 candidates)**:
   * **Command**: `env\Scripts\python.exe rank.py --candidates data/first_200_candidates.jsonl --out data/test_submission.csv`
   * **Result**: Generated a 100-row CSV. Validator verified the submission as fully valid.
   
2. **Full Dataset (100,000 candidates)**:
   * **Command**: `env\Scripts\python.exe -u rank.py --candidates data/candidates.jsonl --out submission.csv`
   * **Result**: Completed in **2 minutes and 27 seconds** on CPU.
   * **Honeypots Screened**: 709 candidates filtered out.
   * **Validator Output**: `Submission is valid.`

### Verification Specifications

| Metric / Check | Requirement | Result | Status |
| :--- | :--- | :--- | :--- |
| **Row Count** | Exactly 100 data rows + header | 100 data rows | **Passed** |
| **Encoding** | Valid UTF-8 | Valid UTF-8 | **Passed** |
| **Schema** | `candidate_id,rank,score,reasoning` | Matches exactly | **Passed** |
| **Honeypot Rate** | < 10% in top 100 | **0%** (0 out of 100 are honeypots) | **Passed** |
| **Monotonicity** | Score non-increasing by rank | Valid descending order | **Passed** |
| **Tie-Breaker** | Deterministic sorting on candidate_id ascending | Sorted correctly | **Passed** |
| **Reasoning Quality** | Factual (1-2 sentences), rank-consistent, no placeholders | Correctly structured and grammatically clean | **Passed** |
| **Wall-Clock Time** | ≤ 5 minutes on CPU | **2 minutes 27 seconds** | **Passed** |
| **Network Constraints** | Offline execution | Fully local execution | **Passed** |

---

## 3. Top Candidate Insights

The ranking system successfully prioritized premium candidates. A view of the top 3 candidates:
1. **CAND_0068932 (Rank 1)**: Outstanding fit with 5.2 YoE. Shipped scalable vector search and embeddings-based retrieval systems. Active on the platform with a 30-day notice period.
2. **CAND_0004402 (Rank 2)**: Premium candidate with 6.0 YoE building production ML pipelines. Active on GitHub with excellent recruiter response rate.
3. **CAND_0064326 (Rank 3)**: Search Engineer with 7.6 YoE, showing deep technical expertise in PyTorch and Semantic Search with short notice period.
