import os
import sys
import json
import argparse
import numpy as np

# Windows DLL directory registration to prevent PyTorch DLL initialization failures
if os.name == "nt":
    possible_torch_libs = [
        os.path.join(os.path.dirname(__file__), "env", "Lib", "site-packages", "torch", "lib"),
        os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
    ]
    for lib_path in possible_torch_libs:
        if os.path.exists(lib_path):
            try:
                os.add_dll_directory(lib_path)
            except AttributeError:
                pass

from sentence_transformers import SentenceTransformer

def format_profile_for_embedding(cand):
    profile = cand.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    
    # Skills list
    skills = cand.get("skills", [])
    skills_names = [s.get("name") for s in skills if s.get("name")]
    skills_str = ", ".join(skills_names)
    
    # Career history descriptions
    career = cand.get("career_history", [])
    recent_jobs = []
    for job in career[:2]: # Get first 2 jobs
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        recent_jobs.append(f"{title} at {company}: {desc}")
    experience_str = "\n".join(recent_jobs)
    
    text = f"Headline: {headline}\nSummary: {summary}\nSkills: {skills_str}\nExperience:\n{experience_str}"
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--jd", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_candidates", required=True)
    parser.add_argument("--out_jd", required=True)
    args = parser.parse_args()
    
    # 1. Load model
    model = SentenceTransformer(args.model_dir, device="cpu")
    
    # 2. Load candidates
    with open(args.candidates, "r", encoding="utf-8") as f:
        if args.candidates.endswith(".jsonl"):
            candidates = [json.loads(line) for line in f if line.strip()]
        else:
            candidates = json.load(f)
            
    # 3. Format profiles
    texts = [format_profile_for_embedding(c) for c in candidates]
    
    # 4. Generate embeddings
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, device="cpu")
    jd_embedding = model.encode(args.jd, convert_to_numpy=True)
    
    # 5. Save results
    np.save(args.out_candidates, embeddings)
    np.save(args.out_jd, jd_embedding)
    print("SUCCESS")

if __name__ == "__main__":
    main()
