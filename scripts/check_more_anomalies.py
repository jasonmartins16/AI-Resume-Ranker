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

def check_more():
    anomalous = {}
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            profile = cand.get("profile", {})
            career = cand.get("career_history", [])
            skills = cand.get("skills", [])
            
            yoe = profile.get("years_of_experience", 0)
            
            # Check 1: Skill duration > total YoE (with margin, e.g. skill duration > YoE + 1 year)
            for sk in skills:
                sk_years = sk.get("duration_months", 0) / 12.0
                if sk_years > yoe + 1.0:
                    if cid not in anomalous:
                        anomalous[cid] = []
                    anomalous[cid].append(f"Skill {sk.get('name')} duration ({sk_years:.1f} yrs) exceeds YoE ({yoe:.1f} yrs)")
            
            # Check 2: Overlapping jobs check (Full-time or regular jobs that overlap significantly)
            # Let's see if there are jobs that overlap by more than 1 month
            job_intervals = []
            for job in career:
                start_dt = parse_date(job.get("start_date"))
                end_dt = parse_date(job.get("end_date"))
                if job.get("is_current"):
                    end_dt = datetime(2026, 6, 18)
                if start_dt and end_dt:
                    job_intervals.append((start_dt, end_dt, job.get("company")))
            
            job_intervals.sort(key=lambda x: x[0])
            for i in range(len(job_intervals) - 1):
                end_i = job_intervals[i][1]
                start_next = job_intervals[i+1][0]
                # If the next job starts before the current one ends, and they are both full-time (or just check any overlap > 3 months)
                if start_next < end_i:
                    overlap_days = (end_i - start_next).days
                    if overlap_days > 90: # Overlap of more than 3 months
                        # Is it a honeypot? Let's check how common this is
                        if cid not in anomalous:
                            anomalous[cid] = []
                        anomalous[cid].append(f"Overlapping jobs: {job_intervals[i][2]} and {job_intervals[i+1][2]} overlap by {overlap_days} days")
                        
    print(f"Total anomalous candidates (more checks): {len(anomalous)}")
    for cid, reasons in list(anomalous.items())[:30]:
        print(f"  {cid}: {reasons}")

if __name__ == "__main__":
    check_more()
