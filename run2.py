import json
import subprocess
import os
from typing import List, Dict, Any

import mysql.connector
from mysql.connector import errorcode
import requests
import spacy
import wikipediaapi

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer, util

import math
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score
)
from transformers import pipeline

# --- Download NLTK data ---
nltk.download('punkt')
nltk.download('punkt_tab')

# --- Load models ---
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")
entailer = pipeline("text-classification", model="roberta-large-mnli")

# --- OMDb & Wikipedia setup ---
OMDB_API_KEY = "369e1ac0"
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="MovieDescBot/1.0 (tanishk.singh@example.com)",
)

# --- LLaMA invocation helper ---
def run_llama(prompt: str) -> str:
    if not prompt:
        return ""
    proc = subprocess.Popen(
        ["ollama", "run", "llama3.2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, _ = proc.communicate(input=prompt.encode())
    return out.decode().strip()

# --- Generate scene-by-scene summary ---
def generate_scene_summary(plot_text: str) -> str:
    if not plot_text:
        return ""
    prompt = (
        "Provide a detailed, scene-by-scene breakdown of the following movie plot,"
        f"\n\nPlot:\n{plot_text}\n\nScene Breakdown:"
    )
    return run_llama(prompt)

# --- IR & ranking metrics ---
def dcg_at_k(rels: List[float], k: int) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2)
               for idx, rel in enumerate(rels[:k]))

def ndcg_at_k(rels: List[float], k: int) -> float:
    dcg_val = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg_val = dcg_at_k(ideal, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def average_precision_at_k(gt_binary: List[int], scores: List[float], k: int = None) -> float:
    if k is not None:
        gt_binary = gt_binary[:k]
        scores = scores[:k]
    if not any(gt_binary):
        return 0.0
    return average_precision_score(gt_binary, scores)

def precision_recall_f1_at_k(gt_binary: List[int], scores: List[float], k: int = 10):
    order = np.argsort(-np.array(scores))[:k]
    pred_binary = [1 if i in order else 0 for i in range(len(scores))]
    p = precision_score(gt_binary, pred_binary, zero_division=0)
    r = recall_score(gt_binary, pred_binary, zero_division=0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def mrr_at_k(gt_binary: List[int], scores: List[float], k: int = 10) -> float:
    order = np.argsort(-np.array(scores))[:k]
    for rank, idx in enumerate(order, start=1):
        if gt_binary[idx] == 1:
            return 1.0 / rank
    return 0.0

# --- Factuality check via entailment ---
def entailment_score(source: str, generated: str) -> float:
    out = entailer(f"{source} </s></s> {generated}")[0]
    return out["score"] if out["label"] == "ENTAILMENT" else 0.0

# --- Load ground-truth judgments ---
def load_relevance_judgments(path: str) -> Dict[str, List[float]]:
    with open(path) as f:
        return json.load(f)

# --- Fetch from OMDb ---
def get_omdb_data(title: str) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"http://www.omdbapi.com/?t={title}&plot=full&apikey={OMDB_API_KEY}"
        )
        data = resp.json()
        return data if data.get("Response") == "True" else {}
    except requests.RequestException:
        return {}

# --- Prompt constructor ---
def construct_prompt(movie: Dict[str, Any]) -> str:
    title = movie.get("title", "")
    outline = movie.get("plot outline", "")
    return (
        f"Write a concise movie description for '{title}'.\n"
        f"Plot outline: {outline}\n"
    )

# --- Ensure DB table exists ---
def ensure_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS movie_descriptions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            description TEXT,
            scenes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
    )

# --- Main pipeline ---
def main() -> None:
    # Load movie data
    with open("sample_movies_copy.json", "r") as f:
        movies = json.load(f)

    # Connect to MySQL (create DB if missing)
    try:
        cnx = mysql.connector.connect(
            host="localhost", user="root", password="tanishk07", database="movie"
        )
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            cnx = mysql.connector.connect(
                host="localhost", user="root", password="tanishk07"
            )
            tmp = cnx.cursor()
            tmp.execute("CREATE DATABASE movie CHARACTER SET utf8mb4;")
            cnx.database = "movie"
        else:
            raise

    cur = cnx.cursor()
    ensure_table(cur)

    embeddings: Dict[str, np.ndarray] = {}
    all_scores: Dict[str, List[float]] = {}

    # Process each movie
    for mv in movies:
        title = mv.get("title", "Unknown")
        print(f"Processing: {title}")

        # Build full plot text
        cinema = mv.get("description", {}).get("cinema", [])
        tvplus = mv.get("description", {}).get("tvplus", [])
        merged = " ".join(cinema + tvplus)
        outline = mv.get("plot outline", "")
        omdb_plot = get_omdb_data(title).get("Plot", "")
        full_plot = " ".join(filter(None, [merged, outline, omdb_plot]))

        # Scene breakdown & embedding
        scenes = generate_scene_summary(full_plot)
        emb = model.encode(scenes)
        embeddings[title] = emb

        # Generate and store description
        desc = run_llama(construct_prompt(mv))
        cur.execute(
            "INSERT INTO movie_descriptions (title, description, scenes) VALUES (%s, %s, %s)",
            (title, desc, scenes)
        )
        cnx.commit()

    # Compute pairwise similarity
    titles = list(embeddings.keys())
    for t1 in titles:
        all_scores[t1] = []
        for t2 in titles:
            if t1 == t2:
                continue
            sim = util.cos_sim(embeddings[t1], embeddings[t2]).item()
            all_scores[t1].append(sim)

    # Print top similarities
    print("\n=== Embedding-based Similarities ===")
    for t in titles:
        ranked = sorted(all_scores[t], reverse=True)[:5]
        print(f"{t}: {ranked}")
    print("\n=== Top Similar Movies ===")
    for t in titles:
        sims = all_scores[t]
        others = [o for o in titles if o != t]
        ranked = sorted(zip(others, sims), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nTop similar movies for '{t}':")
        for other, score in ranked:
            print(f"  {other}: {score:.4f}")
    # Optional: Evaluate with ground-truth
    gt_path = "relevance_judgments.json"
    if os.path.exists(gt_path):
        gt = load_relevance_judgments(gt_path)
        metrics = {
            "NDCG@10": [], "MAP@10": [],
            "P@10": [], "R@10": [], "F1@10": [], "MRR@10": []
        }
        for q in titles:
            scores = all_scores[q]
            rels = gt.get(q, [0] * len(scores))
            gt_bin = [1 if r > 0 else 0 for r in rels]

            metrics["NDCG@10"].append(ndcg_at_k(rels, 10))
            metrics["MAP@10"].append(average_precision_at_k(gt_bin, scores, 10))
            p, r, f1 = precision_recall_f1_at_k(gt_bin, scores, 10)
            metrics["P@10"].append(p)
            metrics["R@10"].append(r)
            metrics["F1@10"].append(f1)
            metrics["MRR@10"].append(mrr_at_k(gt_bin, scores, 10))

        # Print averaged metrics
        avg = {k: float(np.mean(v)) for k, v in metrics.items()}
        print("\n=== Evaluation Metrics ===")
        for name, val in avg.items():
            print(f"{name}: {val:.4f}")

    cur.close()
    cnx.close()

if __name__ == "__main__":
    main()
