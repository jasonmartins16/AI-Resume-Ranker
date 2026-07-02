# Redrob Intelligent Candidate Discovery & Ranking Challenge
## Phase 1: Data Discovery & Profiling Report

This report presents the findings from the discovery and profiling phase conducted on the **100,000-candidate dataset** (`candidates.jsonl`). The analysis details data completeness, text length distributions, behavioral signal distributions, and uncovers the precise structures of the dataset's traps (honeypots, twins, and keyword stuffers).

---

## 1. Data Completeness & Quality Matrix

The candidate pool is exceptionally clean, with **0% missingness** across all profile fields, career histories, educations, and skills. 

| Field Path | Completeness (%) | Null/Missing Count | Data Type | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `candidate_id` | 100.0% | 0 | String (CAND_XXXXXXX) | Unique identifier |
| `profile.anonymized_name` | 100.0% | 0 | String | Anonymized full name |
| `profile.headline` | 100.0% | 0 | String | One-line professional headline |
| `profile.summary` | 100.0% | 0 | String | Professional summary |
| `profile.location` | 100.0% | 0 | String | City/region |
| `profile.country` | 100.0% | 0 | String | Country of residence |
| `profile.years_of_experience` | 100.0% | 0 | Float | Years of professional experience |
| `profile.current_title` | 100.0% | 0 | String | Job title of active role |
| `profile.current_company` | 100.0% | 0 | String | Current employer |
| `profile.current_company_size` | 100.0% | 0 | Enum | Company size range |
| `profile.current_industry` | 100.0% | 0 | String | Industry of current company |
| `career_history` | 100.0% | 0 | Array | 1 to 10 historical job roles |
| `education` | 100.0% | 0 | Array | 0 to 5 education entries |
| `skills` | 100.0% | 0 | Array | List of skills with proficiency |
| `redrob_signals` | 100.0% | 0 | Object | 23 behavioral platform signals |

---

## 2. Text Length Distributions & Embedding Strategy

Analyzing the length of free-form text fields is critical for designing the retrieval and embedding pipeline, as pre-trained models (e.g., Sentence-Transformers) have strict context window limits (typically 256 or 512 tokens).

### 2.1 Text Length Stats (in Characters)
*   **Profile Headline**: Mean of **39.6** characters (Min: 28, Median: 39, Max: 78).
*   **Profile Summary**: Mean of **524** characters (Min: 452, Median: 463, Max: 999).
*   **Career Description (Summed)**: Mean of **1189** characters (Min: 321, Median: 1162, Max: 3745).

### 2.2 Embedding Implications & Recommendations
> [!TIP]
> **Recommended Strategy: Hybrid Input Representation**
> - The combined character count of `headline` + `summary` averages **~564 characters** (~140 tokens). This fits comfortably within the 256-token limit of models like `all-MiniLM-L6-v2` or `bge-small-en-v1.5`.
> - Career descriptions are significantly longer, with a 95th percentile at **2308 characters** (~580 tokens). Embedding the full descriptions will cause truncation in standard encoders.
> - **Actionable Design**: Concatenate `profile.headline`, `profile.summary`, and the *first/current* job description from `career_history` (or current job + skills list) for the dense embedding step. Leave the full career descriptions for a secondary BM25 keyword index or LLM-based re-ranking.

---

## 3. Behavioral Signals Deep Dive

The dataset includes 23 simulated platform activity signals. Profiling their distributions reveals key characteristics:

*   **Availability**: **35.3%** of candidates are actively marked as `open_to_work`.
*   **Engagement**: Recruiter response rate averages **43.7%** (ranging from 2% to 95%). The median response time is **130 hours** (~5.4 days).
*   **GitHub Activity**: More than **50%** of candidates do not have a linked GitHub account (score = -1.0). The remaining active developers have scores up to 96.9.
*   **Salary Expectations**: Min expectations average **12.2 LPA** (INR Lakhs Per Annum), and max expectations average **19.8 LPA**.
*   **Notice Period**: Highly discretized around standard notice periods: 25% at 60 days, 50% at 90 days, and 75% at 120 days. Max notice period is 150 days.

---

## 4. Key Feature Correlations

The correlation matrix reveals strong underlying connections between platform activity metrics, but also exposes interesting anomalies:

1.  **Search to Views to Bookmarks**:
    *   `search_appearance_30d` $\leftrightarrow$ `profile_views_received_30d`: **0.250**
    *   `search_appearance_30d` $\leftrightarrow$ `saved_by_recruiters_30d`: **0.314**
    *   `profile_views_received_30d` $\leftrightarrow$ `saved_by_recruiters_30d`: **0.229**
    *   *Interpretation*: Recruiters searching for candidates leads to views, which in turn leads to them being saved/bookmarked. This is a consistent funnel.
2.  **The Experience $\leftrightarrow$ Salary Expectation Paradox**:
    *   `years_of_experience` $\leftrightarrow$ `salary_min`: **-0.149** (Negative!)
    *   `years_of_experience` $\leftrightarrow$ `salary_max`: **-0.166** (Negative!)
    *   *Interpretation*: In this synthetic dataset, salary expectations are negatively correlated with years of experience. This suggests the data generator might have inverted this relationship or introduced it as a unique signal constraint.
3.  **Search Appearances and Salary**:
    *   `search_appearance_30d` $\leftrightarrow$ `salary_min`: **0.345**
    *   `search_appearance_30d` $\leftrightarrow$ `salary_max`: **0.364**
    *   *Interpretation*: Higher-salary profiles appear in search results significantly more often.

---

## 5. Dataset Traps & Honeypot Analysis

The dataset contains three specific types of traps: **Honeypots**, **Behavioral Twins**, and **Keyword Stuffers**. The profiling step successfully isolated and analyzed these groups.

### 5.1 Hard Honeypots (67 Candidates)
We identified exactly **67 candidates** who violate fundamental consistency rules. Ranking these in the top 100 will trigger disqualification if the rate exceeds 10%. We have categorized them into three mutually exclusive groups:

#### Category A: Expert Skills with Zero Duration (21 Candidates)
These profiles list "expert" proficiency in multiple technical skills but have `duration_months` set to exactly `0` for all of them. In the rest of the 100k pool, 0 candidates have this property.
*   **Rule**: `count(skills where proficiency == 'expert' and duration_months == 0) >= 3`
*   **Candidate IDs**:
    `CAND_0003582`, `CAND_0016000`, `CAND_0033817`, `CAND_0033972`, `CAND_0036839`, `CAND_0042245`, `CAND_0046649`, `CAND_0046689`, `CAND_0048740`, `CAND_0055792`, `CAND_0056983`, `CAND_0060642`, `CAND_0061722`, `CAND_0063888`, `CAND_0065096`, `CAND_0070429`, `CAND_0072379`, `CAND_0073853`, `CAND_0095140`, `CAND_0095317`, `CAND_0095480`

#### Category B: YoE Less Than Single Job Duration (21 Candidates)
These profiles claim a total `years_of_experience` that is strictly less than the duration of a single job in their career history.
*   **Rule**: `any(job.duration_months / 12 > years_of_experience + 0.1)`
*   **Candidate IDs**:
    `CAND_0007353`, `CAND_0008960`, `CAND_0010294`, `CAND_0018515`, `CAND_0035104`, `CAND_0037000`, `CAND_0037539`, `CAND_0040075`, `CAND_0040853`, `CAND_0042453`, `CAND_0043721`, `CAND_0053734`, `CAND_0055685`, `CAND_0057711`, `CAND_0064077`, `CAND_0065710`, `CAND_0070189`, `CAND_0077239`, `CAND_0084182`, `CAND_0093364`, `CAND_0093547`

#### Category C: YoE Exceeds Career Timeline Span (25 Candidates)
These profiles claim a total `years_of_experience` that is significantly greater than the span of their entire career timeline (from their earliest job start date to their latest job end date or current date).
*   **Rule**: `years_of_experience > (latest_end_date - earliest_start_date) in years + 1.0`
*   **Candidate IDs**:
    `CAND_0003430`, `CAND_0005291`, `CAND_0007413`, `CAND_0010770`, `CAND_0013536`, `CAND_0024752`, `CAND_0025579`, `CAND_0033131`, `CAND_0036299`, `CAND_0038431`, `CAND_0039754`, `CAND_0052478`, `CAND_0055992`, `CAND_0065787`, `CAND_0066405`, `CAND_0071115`, `CAND_0074119`, `CAND_0077250`, `CAND_0086808`, `CAND_0090900`, `CAND_0091068`, `CAND_0091534`, `CAND_0093331`, `CAND_0095619`, `CAND_0096150`

> [!WARNING]
> **Honeypot Filter Rule**:
> In the ranking step, we must explicitly filter out all **67 honeypot candidate IDs** listed above and assign them a relevance score of `0.0`. This ensures they are never ranked in the top 100, preventing disqualification.

### 5.2 Behavioral Twins (37 Pairs / 74 Candidates)
We identified exactly **37 pairs** of candidates who share identical primary profile keys `(anonymized_name, current_title, current_company, years_of_experience)`.
*   *Characteristics*: When looking closely at these pairs (e.g., the two `Rajesh Sen` records), we find they have different locations, different educations, different skill lists, and different platform activity signals. 
*   *Significance*: These are not database duplicates; they represent "twins" who look similar on name and current role, but have completely separate qualifications. The ranking system must evaluate their details independently.

### 5.3 Keyword Stuffers
As noted in the Job Description, the pool contains "keyword stuffers" — candidates with unrelated titles (e.g., Marketing Manager, Accountant, Operations Manager) who list many high-level AI keywords (e.g., RAG, Pinecone) in their skills section to game recruiters.
*   *Strategy*: The ranking system will employ a semantic mapping that weights skills based on the candidate's career descriptions. An AI keyword in a skill list will be discounted if the career history descriptions do not support it, while adjacent matching skills (e.g., general software engineering, system design, data engineering) will be valued if they are backed by solid product-company experience.

---

## 6. Ranking Pipeline Architecture Recommendation

Based on the findings of Phase 1, the ranking model will incorporate a **three-stage scoring function**:

1.  **Hard Filters (Disqualification Prevention)**:
    *   Drop the 67 honeypot candidate IDs immediately (Score = 0).
    *   Filter out candidates outside location criteria (sponsor restrictions / Pune/Noida preference).
2.  **Semantic Match Score (Dense & Sparse Retrieval)**:
    *   **Dense Embedding**: Compute cosine similarity between the JD and candidate `headline` + `summary` + `current job` using a local CPU-friendly Sentence-Transformer.
    *   **Sparse Retrieval (BM25)**: Compute match between the JD and candidate career history descriptions.
    *   **Keyword Stuffing Penalty**: Apply a penalty if the candidate's title is unrelated (e.g. Marketing, Accountant) but their skills overlap with high-profile AI keywords.
3.  **Behavioral Signal Multiplier**:
    *   Adjust the semantic match score using platform activity signals:
        *   **Positive Multipliers**: `open_to_work_flag == True`, high `recruiter_response_rate` ($> 0.60$), and high recent views/search appearances.
        *   **Negative Multipliers**: Very long notice period ($> 90$ days), very low recruiter response rate ($< 0.15$), or inactive for $> 6$ months.
