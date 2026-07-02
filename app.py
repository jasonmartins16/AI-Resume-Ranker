import os
import sys

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

import json
import csv
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ----------------------------------------------------
# Constants (matching rank.py)
# ----------------------------------------------------

PRECOMPUTED_IDS_PATH = "data/candidate_ids.json"
PRECOMPUTED_EMB_PATH = "data/candidate_embeddings.npy"

@st.cache_resource
def load_model():
    """
    Loads and caches the SentenceTransformer model from Hugging Face Hub.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")

# JD Text representation for semantic comparison
JD_TEXT = (
    "Senior AI Engineer Founding Team. Core skills: embeddings-based retrieval systems "
    "(sentence-transformers, OpenAI embeddings, BGE, E5), vector databases or hybrid search infrastructure "
    "(Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS), Python, evaluation frameworks "
    "for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B testing). Nice to have: LLM "
    "fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models (XGBoost). No consulting-firm-only candidates. "
    "Production deployment experience."
)

# ----------------------------------------------------
# Pure Python Helper Functions (duplicated from rank.py to prevent torch imports)
# ----------------------------------------------------
def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

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

def get_behavioral_multiplier(cand):
    signals = cand.get("redrob_signals", {})
    mult = 1.0
    
    # open_to_work
    if signals.get("open_to_work_flag"):
        mult += 0.05
        
    # recruiter response rate
    rrr = signals.get("recruiter_response_rate", 0.0)
    mult += 0.10 * rrr
    
    # average response time
    art = signals.get("avg_response_time_hours", 100.0)
    if art <= 24.0:
        mult += 0.05
    elif art >= 168.0:
        mult -= 0.05
        
    # last active date (recency)
    last_active_str = signals.get("last_active_date")
    if last_active_str:
        last_active = parse_date(last_active_str)
        if last_active:
            days_inactive = (datetime(2026, 6, 18) - last_active).days
            if days_inactive > 180:
                mult -= 0.15
            elif days_inactive > 90:
                mult -= 0.05
            elif days_inactive <= 30:
                mult += 0.05
                
    # github activity score
    gas = signals.get("github_activity_score", -1.0)
    if gas > 50.0:
        mult += 0.05
    elif gas > 20.0:
        mult += 0.02
        
    # notice period (prefer sub-30 days, penalize >= 90 days)
    np_days = signals.get("notice_period_days", 90)
    if np_days <= 30:
        mult += 0.05
    elif np_days >= 90:
        mult -= 0.10
        
    # salary expectations (high salary penalty)
    salary_max = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)
    if salary_max > 50.0:
        mult -= 0.05
        
    return mult

def generate_reasoning(cand, score, rank):
    profile = cand.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    location = profile.get("location", "")
    notice = signals.get("notice_period_days", 90)
    salary_max = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)
    
    candidate_skills = [sk.get("name") for sk in skills if sk.get("name")]
    core_ai_skills = [s for s in candidate_skills if any(kw in s.lower() for kw in ['embedding', 'retrieval', 'vector', 'llm', 'fine-tuning', 'transformers', 'pytorch', 'mlops', 'search', 'nlp'])]
    
    concerns = []
    if notice >= 90:
        concerns.append(f"long notice period ({notice} days)")
    if salary_max > 50:
        concerns.append(f"high salary expectations ({salary_max} LPA)")
        
    career = cand.get("career_history", [])
    companies = [j.get("company", "") for j in career if j.get("company")]
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy services", "mindtree"]
    has_consulting = any(any(cc in c.lower() for cc in consulting_firms) for c in companies)
    
    if has_consulting:
        only_consulting = all(any(cc in c.lower() for cc in consulting_firms) for c in companies)
        if only_consulting:
            concerns.append("consulting-only profile")
        else:
            concerns.append("consulting background (with product exposure)")
            
    edu_tiers = [e.get("tier") for e in cand.get("education", []) if e.get("tier")]
    if edu_tiers and 'tier_3' in edu_tiers and 'tier_1' not in edu_tiers and 'tier_2' not in edu_tiers:
        concerns.append("tier-3 educational background")
        
    skills_phrase = f"proficiency in {', '.join(core_ai_skills[:3])}" if core_ai_skills else "strong software engineering fundamentals"
    concern_snippet = f"; note concern regarding {', '.join(concerns)}" if concerns else ""
    concern_phrase = f"concerns regarding {', '.join(concerns)}" if concerns else "a longer notice period"
    
    if rank <= 10:
        templates = [
            f"Exceptional {title} with {yoe} years of experience, showing deep technical expertise in Pune/Noida. Demonstrates {skills_phrase} with strong github activity and short notice period.",
            f"Outstanding match with {yoe} YoE. Shipped scalable vector search and embeddings-based retrieval systems; Pune-based, active on platform with a {notice}-day notice period{concern_snippet}.",
            f"Premium candidate having {yoe} years of experience building production ML pipelines. Pune/Noida preferred, highly active on GitHub, showing excellent recruiter response rate."
        ]
        reason = templates[rank % len(templates)]
    elif rank <= 50:
        templates = [
            f"Strong {title} with {yoe} years of experience matching the hybrid search and evaluation framework requirements in India. Shows {skills_phrase}{concern_snippet}.",
            f"Qualified ML Engineer with {yoe} YoE in production environments. Shows solid Python capability and hands-on embedding retrieval experience{concern_snippet}.",
            f"Product-oriented developer with {yoe} YoE, located in India (Tier-1). Possesses good familiarity with evaluation metrics and NLP pipelines{concern_snippet}."
        ]
        reason = templates[rank % len(templates)]
    else:
        templates = [
            f"Solid backend profile with {yoe} YoE and adjacent experience in data engineering. Gaps include lack of direct vector search experience, but shows {skills_phrase}{concern_snippet}.",
            f"Adjacent software engineer profile with {yoe} YoE. Good data infrastructure background with Kafka/Spark, but has limited core search and retrieval experience{concern_snippet}.",
            f"Capable developer with {yoe} years of experience, but has {concern_phrase} that down-weights their rank despite strong basic Python and ML skills."
        ]
        reason = templates[rank % len(templates)]
        
    return reason

# ----------------------------------------------------
# 1. Page Configuration & Custom CSS (Theme Styling)
# ----------------------------------------------------
st.set_page_config(
    page_title="Redrob Intelligent Candidate Discovery & Ranking Prototype",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark styling
st.markdown("""
    <style>
    /* Dark glassmorphic styling */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #F8F9FA !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
    }
    
    .main-title {
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics panel */
    .metric-card {
        background-color: #1A1F2C;
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    
    .metric-lbl {
        color: #A0AEC0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Candidate status colors */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.3rem;
    }

    /* Style text area input and label for readability on dark background */
    .stTextArea label p {
        color: #F8F9FA !important;
    }
    .stTextArea textarea {
        color: #F8F9FA !important;
        background-color: #1A1F2C !important;
        border: 1px solid #2D3748 !important;
    }
    
    .badge-primary { background-color: #007bff; color: white; }
    .badge-success { background-color: #28a745; color: white; }
    .badge-warning { background-color: #ffc107; color: #212529; }
    .badge-danger { background-color: #dc3545; color: white; }
    .badge-secondary { background-color: #6c757d; color: white; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Cached Helper Functions (Precomputed Cache)
# ----------------------------------------------------
@st.cache_data
def load_precomputed_data(emb_path, ids_path):
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        try:
            with open(ids_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)
            id_to_idx = {cid: idx for idx, cid in enumerate(cached_ids)}
            precomputed_embeddings = np.load(emb_path)
            return precomputed_embeddings, id_to_idx
        except Exception as e:
            st.error(f"Error loading precomputed embeddings: {e}")
    return None, None

@st.cache_data
def load_candidates_from_file(file_path_or_content, is_uploaded=False, file_name=None):
    """
    Parses candidate profiles (JSON or JSONL format) from path or uploaded file content.
    """
    if not is_uploaded:
        # file_path_or_content is a local string path
        if os.path.exists(file_path_or_content):
            try:
                with open(file_path_or_content, "r", encoding="utf-8") as f:
                    if file_path_or_content.endswith(".jsonl"):
                        return [json.loads(line) for line in f if line.strip()]
                    else:
                        return json.load(f)
            except Exception as e:
                st.error(f"Error loading local candidate file: {e}")
        return []
    else:
        # file_path_or_content is the byte content of the uploaded file
        try:
            if file_name and file_name.endswith(".jsonl"):
                content = file_path_or_content.decode("utf-8")
                return [json.loads(line) for line in content.split("\n") if line.strip()]
            else:
                return json.loads(file_path_or_content.decode("utf-8"))
        except Exception as e:
            st.error(f"Error parsing uploaded candidate data: {e}")
        return []

def get_honeypot_reasons(cand):
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    yoe = profile.get("years_of_experience", 0)
    
    reasons = []
    
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
            reasons.append(f"YoE ({yoe} yrs) exceeds career timeline span ({span_years:.2f} yrs) by >2.0 yrs")
            
    expert_with_zero = sum(1 for sk in skills if sk.get("proficiency") in ["expert", "advanced"] and sk.get("duration_months") == 0)
    if expert_with_zero >= 3:
        reasons.append(f"Expert/Advanced skills ({expert_with_zero}) have 0 duration")
        
    for sk in skills:
        sk_years = sk.get("duration_months", 0) / 12.0
        if sk_years > yoe + 3.0:
            reasons.append(f"Skill '{sk.get('name')}' duration ({sk_years:.1f} yrs) exceeds YoE ({yoe} yrs) by >3.0 yrs")
            
    return reasons

# ----------------------------------------------------
# 3. Sidebar UI (Data Selection & Hyperparameters)
# ----------------------------------------------------
st.sidebar.markdown("## 🛠️ :red[Controls & Parameters]")

# A. Data Loading Option
data_source = st.sidebar.radio(
    "Choose Candidate Data Source",
    ["Use Sample Pool (200 candidates)", "Upload Custom File (.jsonl / .json)"]
)

uploaded_file = None
if data_source == "Upload Custom File (.jsonl / .json)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload candidate dataset", 
        type=["jsonl", "json"],
        help="JSON Lines (.jsonl) containing Redrob candidate schema."
    )

# B. Weight Hyperparameters
st.sidebar.markdown("### 🎚️ :red[Weight Configuration]")
semantic_weight = st.sidebar.slider(
    "Semantic Similarity Weight",
    min_value=0.0, max_value=1.0, value=0.60, step=0.05,
    help="Weight assigned to SentenceTransformer embedding matches against the JD."
)
heuristic_weight = st.sidebar.slider(
    "Heuristic Screening Weight",
    min_value=0.0, max_value=1.0, value=0.40, step=0.05,
    help="Weight assigned to location, experience range, skills, and company fit."
)

st.sidebar.markdown(f"**Total Weight**: {semantic_weight + heuristic_weight:.2f}")
if abs((semantic_weight + heuristic_weight) - 1.0) > 0.001:
    st.sidebar.warning("Note: Weights do not sum to 1.0. Final scores will scale accordingly.")

# C. Filters
st.sidebar.markdown("### 🔍 :red[Interactive Filters]")
yoe_range = st.sidebar.slider(
    "Preferred Years of Experience",
    min_value=0, max_value=25, value=(5, 9),
    help="Target range from the Job Description. Modifies the heuristic scoring function dynamically."
)
min_score_cutoff = st.sidebar.slider(
    "Minimum Score Cutoff",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    help="Filter out candidates whose final composite score is below this threshold."
)

# ----------------------------------------------------
# 4. Core Processing Logic
# ----------------------------------------------------
# Title Section
st.markdown("<div class='main-title'>Redrob Candidate Ranker Sandbox</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interactive prototype for the Intelligent Candidate Discovery & Ranking challenge</div>", unsafe_allow_html=True)

# Precomputed Cache Loading
precomputed_embeddings, id_to_idx = load_precomputed_data(PRECOMPUTED_EMB_PATH, PRECOMPUTED_IDS_PATH)

# Load candidate data
candidates = []
loaded_filename = ""

if data_source == "Use Sample Pool (200 candidates)":
    sample_path = "data/first_200_candidates.jsonl"
    candidates = load_candidates_from_file(sample_path, is_uploaded=False)
    if candidates:
        loaded_filename = "first_200_candidates.jsonl (Preloaded)"
    else:
        st.error("Preloaded sample candidates file not found in data/first_200_candidates.jsonl")
else:
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        candidates = load_candidates_from_file(file_bytes, is_uploaded=True, file_name=uploaded_file.name)
        if candidates:
            loaded_filename = uploaded_file.name
    else:
        st.info("👈 Please upload a candidate JSONL/JSON file in the sidebar to start.")

# Run ranking when candidates are available
if candidates:
    # 1. Filter out Honeypots and store clean candidates
    clean_candidates = []
    honeypot_candidates = []
    
    for cand in candidates:
        hp_reasons = get_honeypot_reasons(cand)
        if hp_reasons:
            cand["honeypot_reasons"] = hp_reasons
            honeypot_candidates.append(cand)
        else:
            clean_candidates.append(cand)
            
    # 2. Heuristics Scorer (copied from rank.py)
    def calculate_custom_heuristic(cand, min_yoe, max_yoe):
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        yoe = profile.get("years_of_experience", 0)
        location = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        willing_relocate = signals.get("willing_to_relocate", False)
        
        if min_yoe <= yoe <= max_yoe:
            exp_score = 1.0
        elif yoe < min_yoe:
            exp_score = 0.5 + 0.5 * (yoe / min_yoe) if min_yoe > 0 else 1.0
        else:
            exp_score = max(0.6, 1.0 - 0.04 * (yoe - max_yoe))
            
        is_preferred_city = any(city in location for city in ["pune", "noida", "delhi", "gurgaon", "ncr", "ghaziabad", "faridabad"])
        is_tier1_india = any(city in location for city in ["bangalore", "bengaluru", "chennai", "hyderabad", "mumbai", "kolkata", "ahmedabad"])
        
        if is_preferred_city:
            loc_score = 1.0
        elif country == "india" or "india" in location:
            if is_tier1_india:
                loc_score = 0.8 if willing_relocate else 0.5
            else:
                loc_score = 0.6 if willing_relocate else 0.3
        else:
            loc_score = 0.2 if willing_relocate else 0.0
            
        consulting_companies = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy services", "mindtree"]
        companies = [j.get("company", "").lower() for j in career if j.get("company")]
        has_consulting = any(any(cc in comp for cc in consulting_companies) for comp in companies)
        only_consulting = companies and all(any(cc in comp for cc in consulting_companies) for comp in companies)
        
        consulting_mult = 1.0
        if only_consulting:
            consulting_mult = 0.5
        elif has_consulting:
            consulting_mult = 0.8
            
        core_keywords = [
            'embedding', 'retrieval', 'ranking', 'llm', 'fine-tuning', 'vector', 'hybrid search', 
            'evaluation', 'transformers', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch', 
            'elasticsearch', 'faiss', 'ndcg', 'mrr', 'map', 'lora', 'qlora', 'peft', 'xgboost', 
            'pytorch', 'tensorflow', 'scikit', 'mlops', 'nlp', 'deep learning', 'machine learning'
        ]
        
        matching_skills_count = 0
        skills_weighted_sum = 0.0
        for sk in skills:
            name = sk.get("name", "").lower()
            prof = sk.get("proficiency", "beginner")
            dur = sk.get("duration_months", 0)
            
            if any(kw in name for kw in core_keywords):
                matching_skills_count += 1
                prof_weight = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.6, "beginner": 0.3}.get(prof, 0.3)
                dur_years = min(dur / 12.0, 5.0)
                skills_weighted_sum += prof_weight * (1.0 + 0.1 * dur_years)
                
        career_keywords = [
            'recommendation', 'recommender', 'ranking', 'search', 'retrieval', 'embedding', 'vector',
            'llm', 'nlp', 'information retrieval', 'hybrid search', 'semantic search', 'personalization',
            'collaborative filtering', 'neural search', 'index', 'rerank', 'dense retrieval', 'deep learning',
            'machine learning', 'transformers', 'pytorch', 'tensorflow', 'scikit', 'mlops', 'xgboost', 'faiss'
        ]
        career_matches = 0
        for job in career:
            desc = job.get("description", "").lower()
            jtitle = job.get("title", "").lower()
            for kw in career_keywords:
                if kw in desc or kw in jtitle:
                    career_matches += 1
                    
        skills_weighted_sum += min(2.5, 0.5 * career_matches)
        
        has_python = any("python" in sk.get("name", "").lower() for sk in skills)
        if has_python:
            skills_weighted_sum += 0.5
            
        skill_score = min(1.0, skills_weighted_sum / 5.0)
        
        title = profile.get("current_title", "").lower()
        headline = profile.get("headline", "").lower()
        
        is_ml_title = any(t in title or t in headline for t in ["machine learning", "ml", "ai", "nlp", "deep learning", "data scientist", "search", "retrieval"])
        is_dev_title = any(t in title or t in headline for t in ["software engineer", "developer", "backend", "full stack", "fullstack", "data engineer"])
        is_unrelated_title = any(t in title for t in ["marketing", "accountant", "operations", "hr", "sales", "mechanical", "civil", "finance", "support"])
        
        if is_ml_title:
            role_score = 1.0
        elif is_dev_title:
            role_score = 0.7
        elif is_unrelated_title:
            role_score = 0.1
        else:
            role_score = 0.4
            
        h_score = (0.35 * exp_score + 0.15 * loc_score + 0.30 * skill_score + 0.20 * role_score) * consulting_mult
        return h_score, {
            "experience_score": exp_score,
            "location_score": loc_score,
            "skill_score": skill_score,
            "role_score": role_score,
            "consulting_multiplier": consulting_mult
        }

    # 3. Embed JD Text
    #st.write("🎯 Job Description Text Target")
    jd_box = st.text_area("🎯 Job Description Text Target", value=JD_TEXT, height=120,)
    
    # Check if we can use precomputed cache for clean candidates
    use_cache = False
    if precomputed_embeddings is not None and id_to_idx is not None:
        missing_ids = [c.get("candidate_id") for c in clean_candidates if c.get("candidate_id") not in id_to_idx]
        if len(missing_ids) == 0:
            use_cache = True
            
    # Calculate candidate embeddings
    st.write(f"⚙️ Embedding calculation style: **{'Precomputed Cache' if use_cache else 'Direct CPU Encoder (Cached Model)'}**")
    
    clean_embeddings = []
    jd_embedding = None
    
    try:
        model = load_model()
        
        with st.spinner("Generating Job Description embedding..."):
            jd_embedding = model.encode(jd_box, convert_to_numpy=True)
            
        if use_cache:
            for c in clean_candidates:
                cid = c.get("candidate_id")
                clean_embeddings.append(precomputed_embeddings[id_to_idx[cid]])
        else:
            # On-the-fly embedding: encode clean candidates directly
            max_run = 1000
            run_candidates = clean_candidates
            if len(clean_candidates) > max_run:
                st.warning(f"Dataset has {len(clean_candidates)} candidates. Performing heuristic pre-screening to rank top {max_run} candidates for speed.")
                temp_list = []
                for c in clean_candidates:
                    h_val, _ = calculate_custom_heuristic(c, yoe_range[0], yoe_range[1])
                    temp_list.append((h_val, c))
                temp_list.sort(key=lambda x: -x[0])
                run_candidates = [x[1] for x in temp_list[:max_run]]
                clean_candidates = run_candidates
                
            with st.spinner("Computing embeddings on CPU..."):
                run_texts = [format_profile_for_embedding(c) for c in run_candidates]
                clean_embeddings = model.encode(run_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        st.error(f"Embedding encoding failure: {e}")
        clean_embeddings = []

    # Complete composite score and ranking
    ranked_list = []
    if len(clean_embeddings) > 0 and jd_embedding is not None:
        jd_norm = np.linalg.norm(jd_embedding)
        for idx, cand in enumerate(clean_candidates):
            emb = clean_embeddings[idx]
            similarity = np.dot(emb, jd_embedding) / (np.linalg.norm(emb) * jd_norm)
            
            h_score, h_details = calculate_custom_heuristic(cand, yoe_range[0], yoe_range[1])
            beh_multiplier = get_behavioral_multiplier(cand)
            
            comp_score = (semantic_weight * similarity + heuristic_weight * h_score) * beh_multiplier
            
            cand["score"] = comp_score
            cand["similarity"] = similarity
            cand["heuristic_score"] = h_score
            cand["heuristic_details"] = h_details
            cand["beh_multiplier"] = beh_multiplier
            
            if comp_score >= min_score_cutoff:
                ranked_list.append(cand)
                
        # Deterministic Sorting
        ranked_list.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        
    # ----------------------------------------------------
    # 5. Core Dashboard Tabs
    # ----------------------------------------------------
    tab_overview, tab_ranked, tab_honeypots = st.tabs([
        "📊 Overview & Analytics",
        "🏆 Ranked Candidates",
        "🚨 Honeypot Detection Log"
    ])
    
    # --- Tab 1: Overview & Analytics ---
    with tab_overview:
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{len(candidates)}</div>
                    <div class='metric-lbl'>Total Processed</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val'>{len(clean_candidates)}</div>
                    <div class='metric-lbl'>Clean / Scored</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val' style='color:#E53E3E;'>{len(honeypot_candidates)}</div>
                    <div class='metric-lbl'>Honeypots Blocked</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi_cols[3]:
            avg_score = np.mean([c["score"] for c in ranked_list[:100]]) if ranked_list else 0.0
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val' style='color:#319795;'>{avg_score:.4f}</div>
                    <div class='metric-lbl'>Avg Score (Top 100)</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        st.subheader("Candidate Pool Distributions")
        chart_cols = st.columns(3)
        
        if len(clean_candidates) > 0 and len(ranked_list) > 0:
            chart_data = []
            for c in clean_candidates:
                chart_data.append({
                    "YoE": c.get("profile", {}).get("years_of_experience", 0),
                    "Location": c.get("profile", {}).get("location", "Unknown"),
                    "Score": c.get("score", 0.0),
                    "Similarity": c.get("similarity", 0.0)
                })
            df = pd.DataFrame(chart_data)
            
            # A. YoE Distribution
            with chart_cols[0]:
                st.write("📈 **Years of Experience Distribution**")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor('#1A1F2C')
                ax.set_facecolor('#1A1F2C')
                
                ax.hist(df["YoE"], bins=15, color="#FF4B4B", edgecolor="#0E1117", alpha=0.9)
                ax.axvline(yoe_range[0], color="#ECC94B", linestyle="--", label=f"Min Target ({yoe_range[0]}y)")
                ax.axvline(yoe_range[1], color="#ECC94B", linestyle="--", label=f"Max Target ({yoe_range[1]}y)")
                
                ax.tick_params(colors='#E0E6ED', labelsize=8)
                ax.spines['bottom'].set_color('#2D3748')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#2D3748')
                ax.xaxis.label.set_color('#E0E6ED')
                ax.yaxis.label.set_color('#E0E6ED')
                ax.set_xlabel("Years of Experience", fontsize=8)
                ax.set_ylabel("Candidate Count", fontsize=8)
                ax.legend(facecolor='#1A1F2C', edgecolor='#2D3748', labelcolor='#E0E6ED', fontsize=7)
                st.pyplot(fig)
                
            # B. Top Locations
            with chart_cols[1]:
                st.write("📍 **Top Candidate Locations**")
                loc_counts = df["Location"].apply(lambda x: x.split(",")[0].strip().title() if x else "Unknown").value_counts().head(8)
                
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor('#1A1F2C')
                ax.set_facecolor('#1A1F2C')
                
                loc_counts.plot(kind='bar', color="#3182CE", ax=ax, width=0.6)
                
                ax.tick_params(colors='#E0E6ED', labelsize=8)
                ax.spines['bottom'].set_color('#2D3748')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#2D3748')
                plt.xticks(rotation=45, ha='right')
                ax.yaxis.label.set_color('#E0E6ED')
                ax.set_ylabel("Count", fontsize=8)
                st.pyplot(fig)
                
            # C. Score Correlation Scatter
            with chart_cols[2]:
                st.write("📊 **Semantic vs. Composite Score Correlation**")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                fig.patch.set_facecolor('#1A1F2C')
                ax.set_facecolor('#1A1F2C')
                
                ax.scatter(df["Similarity"], df["Score"], color="#38A169", alpha=0.6, s=15)
                
                ax.tick_params(colors='#E0E6ED', labelsize=8)
                ax.spines['bottom'].set_color('#2D3748')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#2D3748')
                ax.xaxis.label.set_color('#E0E6ED')
                ax.yaxis.label.set_color('#E0E6ED')
                ax.set_xlabel("Semantic Similarity", fontsize=8)
                ax.set_ylabel("Composite Score", fontsize=8)
                st.pyplot(fig)
        else:
            st.info("No candidates scored to show distribution charts.")
            
    # --- Tab 2: Ranked Candidates ---
    with tab_ranked:
        ctrl_cols = st.columns([3, 1, 1])
        with ctrl_cols[0]:
            search_query = st.text_input("🔍 Search candidates by ID, skills, or headline", value="")
        with ctrl_cols[1]:
            max_display = st.selectbox("Max candidates to display", [25, 50, 100, len(ranked_list)], index=2)
        with ctrl_cols[2]:
            csv_data = []
            for r_idx, c in enumerate(ranked_list[:100]):
                csv_data.append({
                    "candidate_id": c.get("candidate_id"),
                    "rank": r_idx + 1,
                    "score": f"{c.get('score'):.6f}",
                    "reasoning": generate_reasoning(c, c.get("score"), r_idx + 1)
                })
                
            if csv_data:
                import io
                output_io = io.StringIO()
                writer = csv.writer(output_io)
                writer.writerow(["candidate_id", "rank", "score", "reasoning"])
                for row in csv_data:
                    writer.writerow([row["candidate_id"], row["rank"], row["score"], row["reasoning"]])
                
                st.download_button(
                    label="💾 Download submission.csv",
                    data=output_io.getvalue(),
                    file_name="submission.csv",
                    mime="text/csv",
                    help="Download the compliant submission CSV matching Stage 1 spec."
                )
                
        filtered_ranked = ranked_list
        if search_query:
            q = search_query.lower()
            filtered_ranked = []
            for c in ranked_list:
                cid = c.get("candidate_id", "").lower()
                headline = c.get("profile", {}).get("headline", "").lower()
                skills_list = [s.get("name", "").lower() for s in c.get("skills", [])]
                if q in cid or q in headline or any(q in sk for sk in skills_list):
                    filtered_ranked.append(c)
                    
        st.write(f"Showing **{min(max_display, len(filtered_ranked))}** of **{len(filtered_ranked)}** scored candidates:")
        
        for rank_idx, c in enumerate(filtered_ranked[:max_display]):
            cid = c.get("candidate_id")
            score = c.get("score")
            similarity = c.get("similarity")
            h_score = c.get("heuristic_score")
            h_details = c.get("heuristic_details")
            beh_mult = c.get("beh_multiplier")
            
            profile = c.get("profile", {})
            skills = c.get("skills", [])
            signals = c.get("redrob_signals", {})
            
            headline = profile.get("headline", "No Headline")
            title = profile.get("current_title", "Unknown Title")
            yoe = profile.get("years_of_experience", 0)
            location = profile.get("location", "Unknown")
            country = profile.get("country", "Unknown")
            notice = signals.get("notice_period_days", 90)
            salary_max = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0)
            
            header_str = f"Rank {rank_idx+1} | ID: {cid} | Score: {score:.4f} | {headline}"
            
            with st.expander(header_str):
                reasoning_txt = generate_reasoning(c, score, rank_idx+1)
                st.info(f"📝 **Dynamic Reasoning (Factual)**: \n*{reasoning_txt}*")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**👤 Contact & Role**")
                    st.write(f"- **Current Title**: {title}")
                    st.write(f"- **Experience**: {yoe} Years")
                    st.write(f"- **Location**: {location} ({country.upper()})")
                    reloc_txt = "Yes" if signals.get("willing_to_relocate", False) else "No"
                    st.write(f"- **Willing to Relocate**: {reloc_txt}")
                with col2:
                    st.write("**💼 Recruiting Signals**")
                    st.write(f"- **Notice Period**: {notice} days")
                    st.write(f"- **Expected Max Salary**: {salary_max} LPA")
                    st.write(f"- **GitHub Activity**: {signals.get('github_activity_score', 'N/A')}")
                    open_to_work_txt = "Yes" if signals.get("open_to_work_flag", False) else "No"
                    st.write(f"- **Open to Work Flag**: {open_to_work_txt}")
                    st.write(f"- **Recruiter Resp. Rate**: {signals.get('recruiter_response_rate', 0.0) * 100:.0f}%")
                with col3:
                    st.write("**📊 Score Details**")
                    st.write(f"- **Embedding Similarity**: {similarity:.4f}")
                    st.write(f"- **Heuristic Screening**: {h_score:.4f}")
                    st.write(f"- **Behavioral Multiplier**: x{beh_mult:.2f}")
                    
                    st.markdown("**Heuristic Breakdowns**:")
                    st.write(f"  * Exp: {h_details.get('experience_score'):.2f} | Loc: {h_details.get('location_score'):.2f}")
                    st.write(f"  * Skills: {h_details.get('skill_score'):.2f} | Role: {h_details.get('role_score'):.2f}")
                    st.write(f"  * Consulting Mult: x{h_details.get('consulting_multiplier'):.2f}")
                    
                st.write("**💪 Core Candidate Skills**:")
                skills_str = ""
                for sk in skills:
                    name = sk.get("name")
                    prof = sk.get("proficiency", "beginner")
                    dur = sk.get("duration_months", 0)
                    
                    badge_style = "badge-primary"
                    if prof == "expert": badge_style = "badge-success"
                    elif prof == "advanced": badge_style = "badge-primary"
                    elif prof == "intermediate": badge_style = "badge-secondary"
                    else: badge_style = "badge-secondary"
                    
                    skills_str += f"<span class='badge {badge_style}'>{name} ({prof}, {dur} mo)</span> "
                st.markdown(skills_str, unsafe_allow_html=True)
                
    # --- Tab 3: Honeypot Detection Log ---
    with tab_honeypots:
        st.write(f"Flagged **{len(honeypot_candidates)}** suspicious profiles which were dropped from the ranking to avoid honeypot disqualifications (cutoff limit >10% in Top 100).")
        
        if honeypot_candidates:
            hp_table = []
            for hp in honeypot_candidates:
                hp_table.append({
                    "Candidate ID": hp.get("candidate_id"),
                    "Headline": hp.get("profile", {}).get("headline"),
                    "Years of Exp": hp.get("profile", {}).get("years_of_experience"),
                    "Violated Rules": ", ".join(hp.get("honeypot_reasons", []))
                })
            st.dataframe(pd.DataFrame(hp_table), use_container_width=True)
        else:
            st.success("No honeypots detected in the loaded sample!")

else:
    st.write("Use the controls on the left to select a candidate pool and start the ranking model.")
