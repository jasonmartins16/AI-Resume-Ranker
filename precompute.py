import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

CANDIDATES_PATH = "e:/AI_Resume_Ranker/data/candidates.jsonl"
EMBEDDINGS_OUT_PATH = "e:/AI_Resume_Ranker/data/candidate_embeddings.npy"
IDS_OUT_PATH = "e:/AI_Resume_Ranker/data/candidate_ids.json"
MODEL_DIR = "e:/AI_Resume_Ranker/models/all-MiniLM-L6-v2"

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
    print("Pre-computation started...")
    
    # 1. Load or download the model
    print(f"Downloading/loading model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Save the model locally
    print(f"Saving model locally to {MODEL_DIR}...")
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    model.save(MODEL_DIR)
    print("Model saved successfully.")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # 2. Load candidates and extract text representation and IDs
    print(f"Loading candidates from {CANDIDATES_PATH}...")
    candidate_ids = []
    texts = []
    
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            cand = json.loads(line)
            candidate_ids.append(cand.get("candidate_id"))
            texts.append(format_profile_for_embedding(cand))
            
            if (idx + 1) % 20000 == 0:
                print(f"Loaded {idx + 1} candidates...")
                
    total = len(texts)
    print(f"Total candidates loaded: {total}")
    
    # 3. Compute embeddings in batches
    print("Generating embeddings (this may take a minute on GPU)...")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    
    # 4. Save results
    print(f"Saving embeddings (shape: {embeddings.shape}) to {EMBEDDINGS_OUT_PATH}...")
    np.save(EMBEDDINGS_OUT_PATH, embeddings)
    
    print(f"Saving candidate IDs to {IDS_OUT_PATH}...")
    with open(IDS_OUT_PATH, "w", encoding="utf-8") as out_f:
        json.dump(candidate_ids, out_f)
        
    print("Pre-computation completed successfully!")

if __name__ == "__main__":
    main()
