# Linear Algebra Hybrid Search Engine

A semantic search and ranking engine built from scratch using only **Python** and **NumPy**—without relying on high-level machine learning or NLP libraries (`scikit-learn`, `nltk`, `gensim`, etc.).

This project implements the core mathematical foundations of modern information retrieval: sparse vector space modeling, dimensionality reduction via Singular Value Decomposition, and eigenvector centrality on directed graphs.

---

## Mathematical Architecture

The retrieval pipeline processes raw text and hyperlink graphs through four linear algebra stages:

1. **TF-IDF & Cosine Similarity (`tfidf_engine.py`)**
   Documents and queries are mapped into an $N \times V$ term-document matrix weighted by Term Frequency–Inverse Document Frequency. To eliminate document-length bias, each row vector is $L_2$-normalized to unit length, reducing cosine similarity across the entire corpus to a single matrix-vector product:
   $$\text{sim}(D, \vec{q}) = \hat{A} \cdot \hat{q}$$

2. **Semantic Retrieval via Truncated SVD / LSA (`svd_engine.py`)**
   The high-dimensional term matrix is factorized via Singular Value Decomposition ($A = U \Sigma V^T$) and truncated to the top $k$ latent semantic axes. Queries are projected into the $k$-dimensional concept space ($\vec{q}_k = V_k^T \vec{q}$), enabling conceptual matching even when exact keywords do not overlap.

3. **Eigenvector Authority via Power Iteration (`pagerank.py`)**
   Internal hyperlinks are modeled as a column-stochastic transition matrix $M$. Dead-end pages (*dangling nodes*) are handled via uniform teleportation, and the dominant eigenvector is computed iteratively with damping factor $d = 0.85$:
   $$v_{t+1} = d M v_t + \frac{1 - d}{N}\mathbf{1}$$

4. **Hybrid Multiplicative Ranking (`hybrid_search.py`)**
   To prevent structural template/boilerplate pages from dominating unrelated queries, PageRank scores are log-normalized to $[0, 1]$ and applied as a multiplicative authority boost over content relevance:
   $$\text{Score} = \big((1 - w)\cdot \text{TF-IDF} + w \cdot \text{SVD}\big) \times \big(1 + \alpha \cdot \text{PR}_{\text{norm}}\big)$$

---

## Experimental Highlights

Evaluated on an autonomously crawled 150-article Wikipedia subgraph (`1,866` internal links):
* **Matrix Sparsity:** `150 × 28,008` vocabulary matrix with **98.33% sparsity** (`70,183` non-zero entries).
* **Dimensionality Reduction:** Truncating from `28,008` terms down to **$k = 40$ latent concepts** (700x compression) preserves **39.62%** of total spectral energy.
* **PageRank Convergence:** Power Iteration converges to an $L_1$ residual below $10^{-6}$ in **25 iterations**.
* **Semantic Recall Improvement:** On narrow queries where exact-match TF-IDF returns only a single document (`0.4272`), Truncated SVD surfaces conceptually related articles with `0.8115` cosine similarity.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `crawler.py` | Breadth-First Search (BFS) Wikipedia API crawler for building linked article datasets. |
| `analyze_dataset.py` | Autonomous corpus profiler reporting vocabulary size, matrix sparsity, and dangling nodes. |
| `tfidf_engine.py` | Vocabulary indexing, TF-IDF weighting, $L_2$ normalization, and exact-match cosine search. |
| `svd_engine.py` | Truncated SVD decomposition and side-by-side TF-IDF vs. LSA semantic search comparison. |
| `pagerank.py` | Column-stochastic transition matrix builder and Power Iteration authority solver. |
| `hybrid_search.py` | Interactive CLI search engine combining lexical, semantic, and graph-authority signals. |

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/EmreErsoy5602/linear-search-engine.git
cd linear-search-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Crawl dataset and inspect matrix statistics
python crawler.py
python analyze_dataset.py

# 4. Launch the hybrid search engine
python hybrid_search.py
```

> **Note:** The default tokenizer and stop-word list in `tfidf_engine.py` are configured for Turkish Wikipedia articles, and can be swapped out for any language without modifying the linear algebra core.

---

## License

This project is open-source under the [MIT License](LICENSE).
