import json

CANDIDATES_PATH = "e:/AI_Resume_Ranker/data/candidates.jsonl"

def print_candidates(ids):
    candidates = {}
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if cid in ids:
                candidates[cid] = cand
    for cid in ids:
        if cid in candidates:
            print(f"=== {cid} ===")
            print(json.dumps(candidates[cid], indent=2))

if __name__ == "__main__":
    print_candidates(["CAND_0000189", "CAND_0003352"])
