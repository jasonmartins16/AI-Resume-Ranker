import json
from datetime import datetime

CANDIDATES_PATH = "e:/AI_Resume_Ranker/data/candidates.jsonl"

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def analyze():
    anomalous_candidates = {}
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            education = cand.get("education", [])
            skills = cand.get("skills", [])
            signals = cand.get("redrob_signals", {})
            
            yoe = profile.get("years_of_experience", 0)
            
            # Check 1: YoE vs career history span
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
                    if cid not in anomalous_candidates:
                        anomalous_candidates[cid] = []
                    anomalous_candidates[cid].append(f"YoE exceeds career span (yoe={yoe}, span={span_years:.2f} yrs)")
            
            # Check 2: Expert/Advanced skills with 0 duration
            expert_with_zero = [sk.get("name") for sk in skills if sk.get("proficiency") in ["expert", "advanced"] and sk.get("duration_months") == 0]
            if len(expert_with_zero) >= 3:
                if cid not in anomalous_candidates:
                    anomalous_candidates[cid] = []
                anomalous_candidates[cid].append(f"Expert skills with 0 duration: {', '.join(expert_with_zero)}")
                
            # Check 3: Education anomalies
            for edu in education:
                sy = edu.get("start_year")
                ey = edu.get("end_year")
                if sy and ey and sy > ey:
                    if cid not in anomalous_candidates:
                        anomalous_candidates[cid] = []
                    anomalous_candidates[cid].append(f"Education start year {sy} > end year {ey}")
            
            # Check 4: Date sequence anomalies
            signup_dt = parse_date(signals.get("signup_date"))
            last_act_dt = parse_date(signals.get("last_active_date"))
            if signup_dt and last_act_dt and signup_dt > last_act_dt:
                # We have 7496 of these, so maybe we don't treat it as a hard honeypot unless other things fail?
                pass

    print(f"Total anomalous candidates (excluding signup > active): {len(anomalous_candidates)}")
    # Print the first 30 anomalous candidates and all their reasons
    for cid, reasons in list(anomalous_candidates.items())[:50]:
        print(f"  {cid}: {reasons}")

if __name__ == "__main__":
    analyze()
