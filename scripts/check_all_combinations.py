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

def check_combinations():
    stats = {
        "yoe_less_than_single_job": 0,
        "yoe_exceeds_career_span_by_1": 0,
        "yoe_exceeds_career_span_by_05": 0,
        "expert_skill_zero_dur_count": {},
        "job_start_before_edu_start_by_5yrs": 0,
        "overlapping_jobs_current": 0,
    }
    
    examples = {
        "yoe_less_than_single_job": [],
        "yoe_exceeds_career_span_by_1": [],
        "expert_skill_zero_dur_count": []
    }
    
    # For finding duplicates / twins
    summaries = {}
    skills_lists = {}
    profiles_hash = {}
    
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
            
            # Twins check: hash candidate fields to see if there are exact duplicates
            # Let's hash by anonymized name, current title, and current company
            name = profile.get("anonymized_name")
            title = profile.get("current_title")
            company = profile.get("current_company")
            yoe_val = profile.get("years_of_experience")
            
            prof_key = (name, title, company, yoe_val)
            if prof_key not in profiles_hash:
                profiles_hash[prof_key] = []
            profiles_hash[prof_key].append(cid)
            
            # Check 1: single job duration > total YoE
            for job in career:
                job_yrs = job.get("duration_months", 0) / 12.0
                if job_yrs > yoe + 0.1: # small epsilon
                    stats["yoe_less_than_single_job"] += 1
                    if len(examples["yoe_less_than_single_job"]) < 5:
                        examples["yoe_less_than_single_job"].append((cid, f"job_yrs={job_yrs:.2f}, yoe={yoe}"))
            
            # Check 2: YoE exceeds career span
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
                if yoe > span_years + 1.0:
                    stats["yoe_exceeds_career_span_by_1"] += 1
                if yoe > span_years + 0.5:
                    stats["yoe_exceeds_career_span_by_05"] += 1
                    
            # Check 3: expert skills with 0 duration
            expert_with_zero = sum(1 for sk in skills if sk.get("proficiency") == "expert" and sk.get("duration_months") == 0)
            if expert_with_zero > 0:
                stats["expert_skill_zero_dur_count"][expert_with_zero] = stats["expert_skill_zero_dur_count"].get(expert_with_zero, 0) + 1
                if expert_with_zero >= 3 and len(examples["expert_skill_zero_dur_count"]) < 5:
                    examples["expert_skill_zero_dur_count"].append(cid)
                    
            # Check 4: Job start before education start by > 5 years
            edu_start_year = min((edu.get("start_year") for edu in education if edu.get("start_year")), default=None)
            job_start_year = min((parse_date(job.get("start_date")).year for job in career if parse_date(job.get("start_date"))), default=None)
            if edu_start_year and job_start_year and job_start_year < edu_start_year - 5:
                stats["job_start_before_edu_start_by_5yrs"] += 1
                
            # Check 5: Overlapping jobs where both are marked is_current
            current_jobs = sum(1 for job in career if job.get("is_current"))
            if current_jobs > 1:
                stats["overlapping_jobs_current"] += 1

    print("STATS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\nEXAMPLES:")
    for k, v in examples.items():
        print(f"  {k}: {v}")
        
    # Find duplicates
    duplicate_profiles = {k: v for k, v in profiles_hash.items() if len(v) > 1}
    print(f"\nDuplicate profile names/titles count: {len(duplicate_profiles)}")
    for k, v in list(duplicate_profiles.items())[:10]:
        print(f"  Profile {k}: Candidates: {v}")

if __name__ == "__main__":
    check_combinations()
