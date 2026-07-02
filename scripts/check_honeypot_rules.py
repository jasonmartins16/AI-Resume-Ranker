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

def check_rules():
    counts = {
        "signup_after_active": 0,
        "edu_start_greater_end": 0,
        "yoe_exceeds_timeline": 0,
        "job_duration_exceeds_dates": 0,
        "founded_date_inconsistency": 0,
        "expert_skills_zero_duration": {}
    }
    
    examples = {
        "signup_after_active": [],
        "edu_start_greater_end": [],
        "yoe_exceeds_timeline": [],
        "job_duration_exceeds_dates": [],
        "founded_date_inconsistency": [],
        "expert_skills_zero_duration": []
    }
    
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
            
            # Check 1: signup date vs last active date
            signup_dt = parse_date(signals.get("signup_date"))
            last_act_dt = parse_date(signals.get("last_active_date"))
            if signup_dt and last_act_dt and signup_dt > last_act_dt:
                counts["signup_after_active"] += 1
                if len(examples["signup_after_active"]) < 5:
                    examples["signup_after_active"].append(cid)
            
            # Check 2: education start_year > end_year
            for edu in education:
                sy = edu.get("start_year")
                ey = edu.get("end_year")
                if sy and ey and sy > ey:
                    counts["edu_start_greater_end"] += 1
                    if len(examples["edu_start_greater_end"]) < 5:
                        examples["edu_start_greater_end"].append((cid, f"{sy} > {ey}"))
                    break
                    
            # Check 3: yoe vs timeline
            total_yoe = profile.get("years_of_experience", 0)
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
                if total_yoe > span_years + 2.0:
                    counts["yoe_exceeds_timeline"] += 1
                    if len(examples["yoe_exceeds_timeline"]) < 5:
                        examples["yoe_exceeds_timeline"].append((cid, f"yoe={total_yoe}, span={span_years:.2f}"))
            
            # Check 4: job duration vs dates
            for job in career:
                dur_months = job.get("duration_months", 0)
                s_dt = parse_date(job.get("start_date"))
                e_dt = parse_date(job.get("end_date"))
                if s_dt and e_dt:
                    calc_months = (e_dt - s_dt).days / 30.4375
                    if dur_months > calc_months + 12:
                        counts["job_duration_exceeds_dates"] += 1
                        if len(examples["job_duration_exceeds_dates"]) < 5:
                            examples["job_duration_exceeds_dates"].append((cid, f"dur={dur_months}, calc={calc_months:.1f}"))
                        break
            
            # Check 5: founded date vs work duration in description
            for job in career:
                desc = job.get("description", "").lower()
                dur_months = job.get("duration_months", 0)
                import re
                match = re.search(r"founded (\d+) years ago", desc)
                if match:
                    founded_years_ago = int(match.group(1))
                    if dur_months / 12 > founded_years_ago + 0.5:
                        counts["founded_date_inconsistency"] += 1
                        if len(examples["founded_date_inconsistency"]) < 5:
                            examples["founded_date_inconsistency"].append((cid, f"worked={dur_months/12:.1f} yrs, founded={founded_years_ago} yrs ago"))
                        break
                
                match_yr = re.search(r"founded in (\d{4})", desc)
                if match_yr and s_dt:
                    founded_year = int(match_yr.group(1))
                    if s_dt.year < founded_year:
                        counts["founded_date_inconsistency"] += 1
                        if len(examples["founded_date_inconsistency"]) < 5:
                            examples["founded_date_inconsistency"].append((cid, f"started={s_dt.year}, founded={founded_year}"))
                        break

            # Check 6: expert/advanced skills with 0 duration
            expert_with_zero = sum(1 for sk in skills if sk.get("proficiency") in ["expert", "advanced"] and sk.get("duration_months") == 0)
            if expert_with_zero > 0:
                counts["expert_skills_zero_duration"][expert_with_zero] = counts["expert_skills_zero_duration"].get(expert_with_zero, 0) + 1
                if expert_with_zero >= 5 and len(examples["expert_skills_zero_duration"]) < 5:
                    examples["expert_skills_zero_duration"].append((cid, f"count={expert_with_zero}"))
                    
    print("COUNTS:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("\nEXAMPLES:")
    for k, v in examples.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    check_rules()
