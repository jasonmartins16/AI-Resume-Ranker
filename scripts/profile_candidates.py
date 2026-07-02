import json
import gzip
import os
import pandas as pd
import numpy as np
from datetime import datetime

CANDIDATES_PATH = "e:/AI_Resume_Ranker/data/candidates.jsonl"

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def run_profiling():
    print("Starting data profiling...")
    
    # We will accumulate stats streaming-style or load in chunks to be memory efficient.
    # Since we have pandas and it can easily load 100k rows in a few seconds, let's load key elements into a list of dicts.
    
    total_candidates = 0
    missing_counts = {}
    
    # For text lengths
    summary_lengths = []
    headline_lengths = []
    description_lengths = []
    
    # For numerical signals and categoricals
    signals_data = []
    
    # To identify honeypots
    honeypot_candidates = []
    
    # To identify twins
    # We will hash the redrob_signals values (excluding candidate_id) to find twins
    signals_hashes = {}
    twin_candidates = []
    
    # Job description keywords for keyword stuffers analysis
    jd_skills = {"embeddings", "retrieval", "ranking", "llms", "fine-tuning", "vector databases", "hybrid search", "evaluation frameworks", "sentence-transformers", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "ndcg", "mrr", "map", "lora", "qlora", "peft", "xgboost"}
    
    keyword_stuffers = []
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            total_candidates += 1
            
            cid = cand.get("candidate_id")
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            education = cand.get("education", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            
            # Missingness checks
            def check_missing(path_prefix, obj, fields):
                for fld in fields:
                    val = obj.get(fld)
                    key = f"{path_prefix}.{fld}"
                    if key not in missing_counts:
                        missing_counts[key] = 0
                    if val is None or val == "" or (isinstance(val, list) and len(val) == 0):
                        missing_counts[key] += 1
            
            check_missing("profile", profile, [
                "anonymized_name", "headline", "summary", "location", "country", 
                "years_of_experience", "current_title", "current_company", 
                "current_company_size", "current_industry"
            ])
            
            if "career_history" not in missing_counts:
                missing_counts["career_history"] = 0
            if not career:
                missing_counts["career_history"] += 1
                
            if "education" not in missing_counts:
                missing_counts["education"] = 0
            if not education:
                missing_counts["education"] += 1
                
            if "skills" not in missing_counts:
                missing_counts["skills"] = 0
            if not skills:
                missing_counts["skills"] += 1
                
            # Text lengths
            summary = profile.get("summary", "")
            summary_lengths.append(len(summary) if summary else 0)
            
            headline = profile.get("headline", "")
            headline_lengths.append(len(headline) if headline else 0)
            
            desc_len_sum = sum(len(job.get("description", "")) for job in career if job.get("description"))
            description_lengths.append(desc_len_sum)
            
            # Collect signals for correlation & distributions
            expected_salary = signals.get("expected_salary_range_inr_lpa", {})
            salary_min = expected_salary.get("min", np.nan)
            salary_max = expected_salary.get("max", np.nan)
            
            signals_data.append({
                "candidate_id": cid,
                "years_of_experience": profile.get("years_of_experience", np.nan),
                "profile_completeness_score": signals.get("profile_completeness_score", np.nan),
                "open_to_work_flag": int(signals.get("open_to_work_flag", 0)) if signals.get("open_to_work_flag") is not None else np.nan,
                "profile_views_received_30d": signals.get("profile_views_received_30d", np.nan),
                "applications_submitted_30d": signals.get("applications_submitted_30d", np.nan),
                "recruiter_response_rate": signals.get("recruiter_response_rate", np.nan),
                "avg_response_time_hours": signals.get("avg_response_time_hours", np.nan),
                "connection_count": signals.get("connection_count", np.nan),
                "endorsements_received": signals.get("endorsements_received", np.nan),
                "notice_period_days": signals.get("notice_period_days", np.nan),
                "salary_min": salary_min,
                "salary_max": salary_max,
                "github_activity_score": signals.get("github_activity_score", np.nan),
                "search_appearance_30d": signals.get("search_appearance_30d", np.nan),
                "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", np.nan),
                "interview_completion_rate": signals.get("interview_completion_rate", np.nan),
                "offer_acceptance_rate": signals.get("offer_acceptance_rate", np.nan),
                "verified_email": int(signals.get("verified_email", 0)) if signals.get("verified_email") is not None else np.nan,
                "verified_phone": int(signals.get("verified_phone", 0)) if signals.get("verified_phone") is not None else np.nan,
                "linkedin_connected": int(signals.get("linkedin_connected", 0)) if signals.get("linkedin_connected") is not None else np.nan,
                "preferred_work_mode": signals.get("preferred_work_mode", ""),
                "willing_to_relocate": int(signals.get("willing_to_relocate", 0)) if signals.get("willing_to_relocate") is not None else np.nan,
            })
            
            # Honeypot checks
            is_honeypot = False
            reasons = []
            
            # 1. Experience vs career history dates mismatch
            # E.g., "8 years of experience at a company founded 3 years ago"
            # Or general years of experience vs total duration/span of jobs
            total_yoe = profile.get("years_of_experience", 0)
            
            # Parse dates and check duration
            job_durations = []
            earliest_start = None
            latest_end = None
            
            for job in career:
                start_dt = parse_date(job.get("start_date"))
                end_dt = parse_date(job.get("end_date"))
                
                if start_dt:
                    if earliest_start is None or start_dt < earliest_start:
                        earliest_start = start_dt
                if job.get("is_current"):
                    latest_end = datetime(2026, 6, 18) # Current date based on local time metadata
                elif end_dt:
                    if latest_end is None or end_dt > latest_end:
                        latest_end = end_dt
            
            if earliest_start and latest_end:
                span_days = (latest_end - earliest_start).days
                span_years = span_days / 365.25
                # If years_of_experience is significantly greater than the span of their career history
                if total_yoe > span_years + 2.0:
                    is_honeypot = True
                    reasons.append(f"YoE ({total_yoe:.1f}) exceeds career timeline span ({span_years:.1f} years)")
            
            # Let's check individual jobs for "X years of experience at a company founded Y years ago"
            # If the description or company has some obvious duration inconsistency.
            # But wait, how do we know when a company was founded?
            # Maybe the company description or is it in the description text? E.g. "8 years of experience at a company founded 3 years ago"
            # Wait, let's search if a job's duration_months is impossible.
            # Or is there a job whose duration_months is, say, 96 (8 years) but the company's size is 1-10 and it says in description "founded 3 years ago"?
            # Let's inspect job descriptions for keywords like "founded X years ago".
            for job in career:
                desc = job.get("description", "").lower()
                dur_months = job.get("duration_months", 0)
                # Check if duration_months is larger than the span of start_date to end_date
                s_dt = parse_date(job.get("start_date"))
                e_dt = parse_date(job.get("end_date"))
                if s_dt and e_dt:
                    calc_months = (e_dt - s_dt).days / 30.4375
                    if dur_months > calc_months + 12: # Allowing a year of buffer
                        is_honeypot = True
                        reasons.append(f"Job duration ({dur_months} mo) exceeds date range ({calc_months:.1f} mo)")
                
                # Check for "founded X years ago" patterns
                if "founded" in desc:
                    # Look for "founded Y years ago" or "founded in YYYY"
                    # We can do simple checks
                    # E.g. "founded 3 years ago" and candidate worked there for 8 years (96 months)
                    import re
                    match = re.search(r"founded (\d+) years ago", desc)
                    if match:
                        founded_years_ago = int(match.group(1))
                        if dur_months / 12 > founded_years_ago + 0.5:
                            is_honeypot = True
                            reasons.append(f"Worked {dur_months/12:.1f} years at company founded {founded_years_ago} years ago")
                    
                    match_yr = re.search(r"founded in (\d{4})", desc)
                    if match_yr and s_dt:
                        founded_year = int(match_yr.group(1))
                        if s_dt.year < founded_year:
                            is_honeypot = True
                            reasons.append(f"Worked at company starting {s_dt.year} before it was founded in {founded_year}")
            
            # 2. "expert" proficiency in 10 skills with 0 years used
            expert_skills_count = sum(1 for sk in skills if sk.get("proficiency") in ["expert", "advanced"])
            expert_with_zero_dur = sum(1 for sk in skills if sk.get("proficiency") in ["expert", "advanced"] and sk.get("duration_months") == 0)
            if expert_with_zero_dur >= 5: # Let's see if 5 or more
                is_honeypot = True
                reasons.append(f"Has {expert_with_zero_dur} expert/advanced skills with 0 months duration")
                
            # 3. Impossible dates in education or other fields
            for edu in education:
                sy = edu.get("start_year")
                ey = edu.get("end_year")
                if sy and ey and sy > ey:
                    is_honeypot = True
                    reasons.append(f"Education start year {sy} > end year {ey}")
            
            # 4. Signup date vs last active date
            signup_dt = parse_date(signals.get("signup_date"))
            last_act_dt = parse_date(signals.get("last_active_date"))
            if signup_dt and last_act_dt and signup_dt > last_act_dt:
                is_honeypot = True
                reasons.append(f"Signup date {signals.get('signup_date')} after last active date {signals.get('last_active_date')}")
            
            if is_honeypot:
                honeypot_candidates.append({
                    "candidate_id": cid,
                    "reasons": reasons,
                    "profile": {
                        "name": profile.get("anonymized_name"),
                        "title": profile.get("current_title"),
                        "company": profile.get("current_company"),
                        "yoe": total_yoe
                    }
                })
                
            # Keyword stuffers check
            # Look for candidates with unrelated current titles but many AI keywords in their profile or skills
            title_lower = profile.get("current_title", "").lower()
            unrelated_titles = ["accountant", "marketing", "operations manager", "hr manager", "sales", "mechanical engineer", "civil engineer", "finance"]
            is_unrelated_title = any(ut in title_lower for ut in unrelated_titles)
            
            candidate_skills = {sk.get("name", "").lower() for sk in skills}
            overlap = candidate_skills.intersection({s.lower() for s in jd_skills})
            
            if is_unrelated_title and len(overlap) >= 5:
                keyword_stuffers.append({
                    "candidate_id": cid,
                    "title": profile.get("current_title"),
                    "skills_count": len(skills),
                    "ai_keywords_count": len(overlap),
                    "ai_keywords": list(overlap),
                    "summary": summary
                })
                
            # Behavioral twins check
            # We will create a tuple of behavioral signals to detect duplicates
            sig_tuple = (
                signals.get("profile_completeness_score"),
                signals.get("open_to_work_flag"),
                signals.get("profile_views_received_30d"),
                signals.get("applications_submitted_30d"),
                signals.get("recruiter_response_rate"),
                signals.get("avg_response_time_hours"),
                signals.get("connection_count"),
                signals.get("endorsements_received"),
                signals.get("notice_period_days"),
                salary_min,
                salary_max,
                signals.get("github_activity_score"),
                signals.get("search_appearance_30d"),
                signals.get("saved_by_recruiters_30d"),
                signals.get("interview_completion_rate"),
                signals.get("offer_acceptance_rate"),
                signals.get("verified_email"),
                signals.get("verified_phone"),
                signals.get("linkedin_connected"),
                signals.get("preferred_work_mode"),
                signals.get("willing_to_relocate")
            )
            
            if sig_tuple not in signals_hashes:
                signals_hashes[sig_tuple] = []
            signals_hashes[sig_tuple].append(cid)
            
    # Process twins
    for sig_tup, cids in signals_hashes.items():
        if len(cids) > 1:
            twin_candidates.append({
                "candidate_ids": cids,
                "count": len(cids),
                "signals": {
                    "completeness": sig_tup[0],
                    "open_to_work": sig_tup[1],
                    "views": sig_tup[2],
                    "applications": sig_tup[3],
                    "response_rate": sig_tup[4],
                    "response_time": sig_tup[5],
                    "connections": sig_tup[6],
                    "endorsements": sig_tup[7],
                    "notice_period": sig_tup[8],
                    "salary_min": sig_tup[9],
                    "salary_max": sig_tup[10],
                    "github_score": sig_tup[11],
                    "search_appearances": sig_tup[12],
                    "saved_by_recruiters": sig_tup[13],
                    "interview_completion": sig_tup[14],
                    "offer_acceptance": sig_tup[15]
                }
            })

    # Convert signals to DataFrame for summary statistics and correlation
    df_signals = pd.DataFrame(signals_data)
    
    # Exclude candidate_id from numeric calculations
    numeric_cols = [c for c in df_signals.columns if c not in ["candidate_id", "preferred_work_mode"]]
    
    # Calculate stats
    stats_summary = df_signals[numeric_cols].describe().to_dict()
    
    # Calculate correlations
    corr_matrix = df_signals[numeric_cols].corr().to_dict()
    
    # Categorical distributions
    preferred_work_mode_counts = df_signals["preferred_work_mode"].value_counts().to_dict()
    
    # Text lengths percentile stats
    def get_percentiles(arr):
        return {
            "min": int(np.min(arr)),
            "p10": int(np.percentile(arr, 10)),
            "p25": int(np.percentile(arr, 25)),
            "p50": int(np.percentile(arr, 50)),
            "p75": int(np.percentile(arr, 75)),
            "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
            "max": int(np.max(arr)),
            "mean": float(np.mean(arr))
        }
    
    summary_len_stats = get_percentiles(summary_lengths)
    headline_len_stats = get_percentiles(headline_lengths)
    description_len_stats = get_percentiles(description_lengths)
    
    # Save output
    output_data = {
        "total_candidates": total_candidates,
        "missingness": missing_counts,
        "text_lengths": {
            "summary": summary_len_stats,
            "headline": headline_len_stats,
            "description": description_len_stats
        },
        "signals_summary": stats_summary,
        "signals_correlation": corr_matrix,
        "categorical_counts": {
            "preferred_work_mode": preferred_work_mode_counts
        },
        "honeypot_count": len(honeypot_candidates),
        "honeypot_examples": honeypot_candidates[:20],  # Save first 20 as examples
        "keyword_stuffer_count": len(keyword_stuffers),
        "keyword_stuffer_examples": keyword_stuffers[:20],
        "twins_count": sum(tw.get("count") for tw in twin_candidates),
        "twins_group_count": len(twin_candidates),
        "twins_examples": twin_candidates[:10]
    }
    
    output_path = "e:/LogisChainAI/data/profiling_output.json"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, indent=2)
        
    print(f"Profiling complete. Output saved to {output_path}")

if __name__ == "__main__":
    run_profiling()
