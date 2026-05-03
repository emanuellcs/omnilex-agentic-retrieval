# Omnilex LLM Agentic Legal Information Retrieval

This repository contains the solution for the **[LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)** Kaggle competition. The goal of this system is to retrieve the most relevant Swiss legal sources (statutes, court decisions) for an open-ended legal question in English, optimizing for the citation-level Macro F1 score on a hidden test set.

## Overview & Core Philosophy

Our solution tackles the inherent challenges of cross-lingual legal retrieval (English queries to German/French/Italian texts) and the strict formatting requirements of legal citations by employing a robust, multi-stage pipeline.

The system is built on three core pillars:
1.  **Hybridity**: Combining lexical exact-matching (BM25) with multilingual semantic search (Dense Retrieval).
2.  **Structural Prior**: Utilizing the Swiss legal citation network as a graph-based prior to retrieve "forgotten" but highly relevant co-citations.
3.  **Adversarial Decision-Making**: Deploying Large Language Models (LLMs) as expert agents debating the final inclusion set to maximize the set-based F1 score.

## Multi-Stage Retrieval Pipeline

The architecture is divided into four main stages, ensuring high recall initially and high precision at the end.

### Stage 1: Candidate Generation (Hybrid Retrieval)
To ensure no relevant documents are missed, we generate an initial candidate pool using dual strategies:
* **Lexical (BM25)**: We query both the original English question and a German translation (powered by `Helsinki-NLP/opus-mt-en-de`). This catches specific German legal terminology.
* **Dense (Multilingual-E5)**: We embed the original English query using `intfloat/multilingual-e5-large` and perform a fast cosine similarity search via FAISS against all legal text chunks.
* **Fusion**: We use Reciprocal Rank Fusion (RRF) with `k=60` to seamlessly merge the retrieved candidate lists.

### Stage 2: Graph Expansion (The Citation Prior)
Swiss court decisions form a highly interconnected web. We parse 30 years of the `court_considerations` corpus to build a **Citation Co-occurrence Graph**.
* Using the top 5 candidates from Stage 1 as "teleportation" seeds, we run a **Personalized PageRank** random walk.
* This identifies citations frequently co-cited with our top hits, successfully adding the top 10 neighbors to our candidate pool.

### Stage 3: High-Precision Reranking
We score all candidates (lexical, dense, and graph-added) using a multilingual Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). Unlike bi-encoders, the cross-encoder applies full attention between the query and text snippet, granting a massive boost to relevance scoring.

### Stage 4: Grounding & Final Selection
To ensure 100% exact-match compliance and avoid zero-scoring hallucinations, every predicted citation goes through a **Hard Grounding Oracle**. We cross-reference outputs against the pre-computed canonical citation set (`self.corpus_citation_set`). 

We implemented two parallel mechanisms for final citation selection:

#### A. Primary Track (Fully Offline)
Designed for the strict <12-hour offline Kaggle constraint. It uses an **F1-Optimized Thresholding** strategy. The `tune_threshold` function sweeps a range of cross-encoder scores against the validation set to dynamically lock in the threshold that empirically maximizes the Macro F1 metric.

#### B. Creative Track (LLM Agentic Debate)
Designed for the "Most Creative" prize category, this approach utilizes the `google-genai` SDK and `gemini-3-flash-preview` to simulate a mock trial:
1.  **The Advocate**: Argues *why* a candidate is legally relevant.
2.  **The Devil's Advocate**: Challenges the reasoning and argues for exclusion.
3.  **The Arbiter**: Simulating a strict Swiss law exam grader, it reads the debate and makes the final `INCLUDE`/`EXCLUDE` decision, penalizing both omissions and tangential additions.

## Technical Stack

* **Retrieval**: `bm25s`, `faiss-cpu/gpu`
* **NLP**: `sentence-transformers`, `transformers` (MarianMT), `torch`
* **Graph Processing**: `networkx`
* **Agentic Framework**: Gemini 3.0 Flash Preview (`google-genai`)
* **Data Parsing**: Chunked `pandas` loading to handle the 2.5GB+ judicial corpora

## Configuration & Usage

The pipeline parameters can be modified via `config/pipeline_config.json`. The default settings are:

```json
{
  "embedder_model": "intfloat/multilingual-e5-large",
  "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
  "translator_model": "Helsinki-NLP/opus-mt-en-de",
  "bm25_top_k": 50,
  "dense_top_k": 50,
  "graph_expansion_seeds": 5,
  "graph_expansion_k": 10,
  "reranker_top_k": 150,
  "threshold": 0.0,
  "rrf_k": 60,
  "text_excerpt_max_chars": 512
}
```
### Validating the Submission Format
To ensure no pipeline errors compromise the Kaggle evaluation, run the validation script over your final output:

```bash
python utils/validate_submission.py submission.csv --verbose
```

This guarantees the output conforms to the `query_id,predicted_citations` requirement.
