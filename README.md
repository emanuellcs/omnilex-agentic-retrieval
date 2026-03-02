# Omnilex: Advanced Agentic Legal Retrieval for Swiss Law

Project for the Kaggle LLM Agentic Legal Information Retrieval competition. This project implements a sophisticated, multi-stage hybrid retrieval pipeline designed to handle complex English legal queries and retrieve exact canonical citations from Swiss federal law and court decision corpora.

## Key Features

- **Multi-Stage Hybrid Retrieval**: Combines BM25 (lexical) and Dense (semantic) search with Cross-Encoder reranking.
- **Structural Priors**: Leverages 30 years of Swiss Federal Court co-occurrence data via a Citation Graph.
- **Adversarial LLM Debate**: Uses a three-agent debate mechanism (Gemini API) to maximize F1 score by disambiguating complex cases.
- **Hard Grounding**: Ensures 100% validity of output citations via strict corpus validation.
- **Kaggle Optimized**: Fully supports offline inference with memory-efficient chunked loading for large corpora (2.5M+ rows).

## 🛠 Project Structure

```
├── src/omnilex/           # Core Package
│   ├── retrieval/         # Search engines (BM25, Dense), Reranker, Graph, Translator
│   ├── pipeline/          # Orchestrators (Hybrid & Creative pipelines)
│   ├── citations/         # Swiss-specific citation normalizer and abbreviations
│   ├── evaluation/        # Macro F1 and secondary retrieval metrics
│   └── llm/               # Model loading and prompting utilities
├── notebooks/             # Implementation & Analysis
│   ├── 03_hybrid_dense_retrieval.ipynb  # Primary Kaggle Submission
│   ├── 04_creative_debate_pipeline.ipynb # Creative Prize Submission (Gemini)
│   └── 05_citation_graph_analysis.ipynb  # Structural Exploratory Analysis
├── utils/                 # Offline Build & Tuning Scripts
├── tests/                 # Comprehensive Unit Test Suite
└── config/                # Pipeline and Path Configurations
```

## 🚀 Getting Started

### 1. Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Data

Organize the competition data into `data/raw/`:
- `laws_de.csv`
- `court_considerations.csv`
- `val.csv`
- `test.csv`
- `train.csv`
- `sample_submission`

### 3. Build Offline Indices

Run the build scripts to generate searchable indices and the citation graph:

```bash
# 1. BM25 Indices
python utils/build_bm25_indices.py --laws-csv data/raw/laws_de.csv --courts-csv data/raw/court_considerations.csv

# 2. Dense FAISS Indices (requires ~2GB VRAM or CPU)
python utils/build_faiss_indices.py --laws-csv data/raw/laws_de.csv --courts-csv data/raw/court_considerations.csv

# 3. Citation Co-occurrence Graph
python utils/build_citation_graph.py --courts-csv data/raw/court_considerations.csv
```

## 🤖 Retrieval Pipelines

### Primary Pipeline (Offline)
Located in `notebooks/03_hybrid_dense_retrieval.ipynb`. It uses:
1. **Query Translation**: English --> German (MarianMT).
2. **Retrieval**: BM25 + Multilingual-E5-Large.
3. **Fusion**: Reciprocal Rank Fusion (RRF).
4. **Graph Expansion**: Personalized PageRank expansion.
5. **Reranking**: Cross-Encoder (mMiniLMv2).
6. **Selection**: F1-optimized thresholding.

### Creative Pipeline (Gemini API)
Located in `notebooks/04_creative_debate_pipeline.ipynb`. It replaces the thresholding stage with a **Three-Agent Adversarial Debate** powered by Gemini 3.0 Flash Preview via the `google-genai` SDK:
- **Advocate**: Argues for inclusion.
- **Devil's Advocate**: Argues for exclusion.
- **Arbiter**: Makes the final binary decision based on the debate, calibrated for Swiss law school standards (omission penalty vs. over-inclusion penalty).

## 🧪 Testing

Run the full test suite to verify module integrity:

```bash
pytest tests/ -v
```

## 📜 License

Apache 2.0. See [LICENSE](LICENSE) for details.
