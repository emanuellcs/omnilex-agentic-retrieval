# Solution Architecture: Omnilex Legal Retrieval

This document provides a detailed technical overview of the retrieval architecture implemented for the Swiss legal information retrieval task.

## 1. Core Philosophy

The system is built on three pillars:
1.  **Hybridity**: Combining lexical exact-matching (BM25) with multilingual semantic search (Dense).
2.  **Structural Prior**: Using the citation network as a graph-based prior to identify "forgotten" but relevant citations.
3.  **Adversarial Decision-Making**: Using LLMs not just for search, but as expert agents debating the final inclusion set to optimize set-based F1.

## 2. Multi-Stage Retrieval Pipeline

### Stage 1: Candidate Generation (Hybrid Retrieval)
- **Lexical (BM25)**: We use two query variants: the original English query and a German translation (via MarianMT). This ensures we catch German legal terms that semantic models might miss.
- **Dense (Multilingual-E5)**: We embed the original English query and perform cosine similarity search using FAISS against the multilingual passage embeddings.
- **Merge (RRF)**: We use Reciprocal Rank Fusion (k=60) to merge these four lists (EN-BM25, DE-BM25, EN-Dense-Laws, EN-Dense-Courts).

### Stage 2: Graph Expansion (The Citation Prior)
Swiss court decisions are a highly interconnected web. We extract all citation mentions from 30 years of the `court_considerations` corpus to build a **Citation Co-occurrence Graph**.
- **Personalized PageRank**: We use the top-5 candidates from Stage 1 as "teleportation" seeds in the graph. The random walk identifies citations that are frequently co-cited with our top candidates, adding the top-10 neighbors to the candidate pool.

### Stage 3: High-Precision Reranking
We score all candidates (lexical, dense, and graph-added) using a multilingual **Cross-Encoder** (`mmarco-mMiniLMv2-L12-H384-v1`). Unlike bi-encoders (Stage 1), the cross-encoder performs full attention between the query and the text snippet, providing a significantly more accurate relevance signal.

## 3. Final Selection Mechanisms

### Primary: F1-Optimized Thresholding
For the fully offline submission, we sweep a range of scores on the validation set to find the threshold that maximizes Macro F1. This global threshold provides a balanced trade-off between precision and recall.

### Creative: Three-Agent Adversarial Debate (Gemini)
The creative submission uses an LLM-based decision engine powered by the `google-genai` SDK and `gemini-3-flash-preview`:
1.  **The Advocate**: Instructed to find reasons *why* each of the top-30 candidates is relevant.
2.  **The Devil's Advocate**: Instructed to challenge those reasons and argue for exclusion.
3.  **The Arbiter**: Reads the debate and makes a binary INCLUDE/EXCLUDE decision. The prompt explicitly instructs the Arbiter to mimic a Swiss law exam grader, penalizing both the omission of critical authorities and the inclusion of tangentially related ones.

## 4. Hard Grounding & Safety
A critical component of the architecture is the **Hard Grounding Oracle**. Every pipeline output is filtered through a pre-computed set of all canonical citations in the retrieval corpus. This guarantees that no hallucinated or malformed citations are ever submitted, ensuring zero precision loss from formatting errors.

## 5. Technical Stack
- **Retrieval**: `bm25s`, `faiss-cpu/gpu`
- **NLP**: `sentence-transformers`, `transformers` (MarianMT), `torch`
- **Graph**: `networkx`
- **Inference**: Gemini 3.0 Flash Preview (`google-genai`), Local CPU/GPU (Primary)
- **Data Handling**: `pandas` (with chunked loading for 2.5GB+ files)
