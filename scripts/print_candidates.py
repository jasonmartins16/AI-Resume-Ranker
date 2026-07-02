import json

CANDIDATES_PATH = "e:/AI_Resume_Ranker/data/candidates.jsonl"

def print_candidate(target_cid):
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            if cand.get("candidate_id") == target_cid:
                print(json.dumps(cand, indent=2))
                return

print("--- CAND_0003430 ---")
print_candidate("CAND_0003430")

print("\n--- CAND_0016000 ---")
print_candidate("CAND_0016000")
