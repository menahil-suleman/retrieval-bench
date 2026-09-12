"""
run_experiment.py — Full RetrievalBench experiment runner.

Usage:
    py run_experiment.py

What it does:
    1. Loads corpus.json and test_set.json from data/
    2. Builds the VectorStore (with embedding cache)
    3. Instantiates all three retrieval configurations
    4. Calls evaluate() across all configurations and all queries
    5. Saves results to results/ and plots to plots/
    6. Prints a summary table and highlights example queries
       where configurations disagreed
"""

import json
import sys
import os

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    EMBEDDING_MODEL, CROSS_ENCODER_MODEL,
    TOP_K, RERANK_TOP_N, HYBRID_ALPHA,
    K_VALUES, DATA_DIR, RESULTS_DIR, PLOTS_DIR,
    CORPUS_FILE, TEST_SET_FILE, RESULTS_FILE, RESULTS_CSV,
)
from retrievalbench.store import VectorStore
from retrievalbench.retrieval import (
    DenseRetriever, HybridRetriever, HybridRerankedRetriever,
)
from retrievalbench.evaluate import evaluate, save_results
from retrievalbench.visualise import print_summary_table, save_all_plots


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  RetrievalBench — Comparative RAG Retrieval Evaluation")
print("=" * 60)

print("\n[1/5] Loading corpus and test set…")
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    corpus = json.load(f)
with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
    test_set = json.load(f)

print(f"  Corpus  : {len(corpus)} chunks")
print(f"  Test set: {len(test_set)} queries")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build vector store
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Building vector store…")
store = VectorStore(model_name=EMBEDDING_MODEL, cache_dir=DATA_DIR)
store.build(corpus, show_progress=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Instantiate configurations
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Instantiating retrieval configurations…")

dense   = DenseRetriever(store=store)
hybrid  = HybridRetriever(store=store, alpha=HYBRID_ALPHA)
reranked = HybridRerankedRetriever(
    hybrid=hybrid,
    cross_encoder_model=CROSS_ENCODER_MODEL,
    rerank_candidates=max(TOP_K * 2, 20),
)
configurations = [dense, hybrid, reranked]
print(f"  Configurations: {[c.name for c in configurations]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run evaluation
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/5] Running evaluation…")
results = evaluate(
    configurations=configurations,
    test_set=test_set,
    top_k=TOP_K,
    k_values=K_VALUES,
    verbose=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save results and plots
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/5] Saving results and generating plots…")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

save_results(results, RESULTS_FILE, RESULTS_CSV)
save_all_plots(results, PLOTS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary table
# ─────────────────────────────────────────────────────────────────────────────

print_summary_table(results, k_values=K_VALUES)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Illustrative disagreement examples
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Illustrative Examples — Where Configurations Disagreed")
print("=" * 60)

dense_results   = results.get("dense")
hybrid_results  = results.get("hybrid")
reranked_results = results.get("hybrid_reranked")

if dense_results and reranked_results:
    dense_by_qid    = {qr.query_id: qr for qr in dense_results.per_query}
    reranked_by_qid = {qr.query_id: qr for qr in reranked_results.per_query}

    disagreements = []
    for qid, dense_qr in dense_by_qid.items():
        reranked_qr = reranked_by_qid.get(qid)
        if reranked_qr is None:
            continue
        dense_hit    = dense_qr.metrics.get("recall@5", 0.0)
        reranked_hit = reranked_qr.metrics.get("recall@5", 0.0)
        # Cases where reranking fixed a dense failure
        if reranked_hit > dense_hit:
            disagreements.append((qid, dense_qr, reranked_qr))

    print(f"\nQueries where Hybrid+Reranked found the answer but Dense did not "
          f"(Recall@5): {len(disagreements)} / {len(test_set)}\n")

    for qid, d_qr, r_qr in disagreements[:5]:
        print(f"  Query ID  : {qid}")
        print(f"  Query     : {d_qr.query}")
        print(f"  Relevant  : {d_qr.relevant_ids}")
        print(f"  Dense top3: {d_qr.retrieved_ids[:3]}")
        print(f"  Rerank top3: {r_qr.retrieved_ids[:3]}")
        print()

print("\nDone. All results in results/ and plots in plots/")
