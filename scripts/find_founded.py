import json

CANDIDATES_PATH = "e:/AI_Resume_Ranker/data/candidates.jsonl"

def find_founded_desc():
    count = 0
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            career = cand.get("career_history", [])
            for job in career:
                desc = job.get("description", "")
                if "founded" in desc.lower() or "establish" in desc.lower():
                    count += 1
                    if count <= 10:
                        print(f"Candidate: {cand.get('candidate_id')}")
                        print(f"Company: {job.get('company')}, Duration: {job.get('duration_months')} months")
                        print(f"Description: {desc}\n")
    print(f"Total jobs with 'founded' or 'establish': {count}")

if __name__ == "__main__":
    find_founded_desc()
