import csv
import json
from datetime import datetime

SUBMISSION_PATH = "e:/AI_Resume_Ranker/submission.csv"
CANDIDATES_PATH = "e:/AI_Resume_Ranker/data/candidates.jsonl"

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def is_honeypot(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    yoe = profile.get("years_of_experience", 0)
    
    # 1. Check YoE exceeds career timeline span
    earliest_start = None
    latest_end = None
    for job in career:
        start_dt = parse_date(job.get("start_date"))
        end_dt = parse_date(job.get("end_date"))
        if start_dt:
            if earliest_start is None or start_dt < earliest_start:
                earliest_start = start_dt
        if job.get("is_current"):
            latest_end = datetime(2026, 6, 18)
        elif end_dt:
            if latest_end is None or end_dt > latest_end:
                latest_end = end_dt
                
    if earliest_start and latest_end:
        span_years = (latest_end - earliest_start).days / 365.25
        if yoe > span_years + 2.0:
            return True, f"YoE ({yoe}) exceeds span ({span_years:.2f} yrs)"
            
    # 2. Check Expert/Advanced skills with 0 duration
    expert_with_zero = sum(1 for sk in skills if sk.get("proficiency") in ["expert", "advanced"] and sk.get("duration_months") == 0)
    if expert_with_zero >= 3:
        return True, f"Expert skills with 0 duration >= 3 (count: {expert_with_zero})"
        
    # 3. Check Skill duration exceeds YoE
    for sk in skills:
        sk_years = sk.get("duration_months", 0) / 12.0
        if sk_years > yoe + 3.0:
            return True, f"Skill {sk.get('name')} duration ({sk_years:.2f} yrs) exceeds YoE ({yoe} yrs)"
            
    return False, ""

def main():
    # 1. Load candidate IDs from submission
    top_100_ids = []
    with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top_100_ids.append(row["candidate_id"])
            
    assert len(top_100_ids) == 100, f"Expected 100 rows, got {len(top_100_ids)}"
    top_100_set = set(top_100_ids)
    
    # 2. Scan candidates to check if any of top 100 are honeypots
    cands_found = {}
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if cid in top_100_set:
                cands_found[cid] = cand
                
    print(f"Loaded profiles for {len(cands_found)} of the top 100 candidates.")
    
    honeypot_in_top_100 = 0
    for cid in top_100_ids:
        cand = cands_found.get(cid)
        if not cand:
            print(f"WARNING: Candidate {cid} not found in candidates.jsonl!")
            continue
        flag, reason = is_honeypot(cand)
        if flag:
            print(f"CRITICAL ERROR: Candidate {cid} (ranked) is a honeypot! Reason: {reason}")
            honeypot_in_top_100 += 1
            
    if honeypot_in_top_100 == 0:
        print("SUCCESS: Zero honeypots found in the top 100 candidates!")
    else:
        print(f"FAILED: Found {honeypot_in_top_100} honeypots in top 100.")

if __name__ == "__main__":
    main()
