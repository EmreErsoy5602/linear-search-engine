import json
from collections import Counter
import numpy as np

from tfidf_engine import build_index, tokenize
from svd_engine import build_svd_index
from pagerank import build_transition_matrix, compute_pagerank


class HybridSearchEngine:
    def __init__(self, dataset_path: str = "dataset.json", svd_k: int = 40):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        print("1/3 TF-IDF indeksi kuruluyor...")
        self.vocab, self.idf_vector, self.tfidf_norm = build_index(self.dataset)

        print("2/3 SVD anlamsal uzayı hesaplanıyor...")
        self.doc_concepts_norm, self.Vt_k = build_svd_index(self.tfidf_norm, k=svd_k)

        print("3/3 PageRank özvektörü yakınsatılıyor...")
        M = build_transition_matrix(self.dataset)
        raw_pr = compute_pagerank(M)

        # Aykırı şablon sayfalarının (örn. Wayback Machine) skoru ezmemesi için log + min-max [0, 1]
        log_pr = np.log(raw_pr)
        pr_range = log_pr.max() - log_pr.min()
        self.pr_norm = (log_pr - log_pr.min()) / pr_range if pr_range > 0 else np.zeros_like(raw_pr)

    def _vectorize_query(self, query: str) -> np.ndarray | None:
        tokens = tokenize(query)
        if not tokens:
            return None

        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        counts = Counter(tokens)
        for word, count in counts.items():
            if word in self.vocab:
                col_idx = self.vocab[word]
                q_vec[col_idx] = (count / len(tokens)) * self.idf_vector[col_idx]

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None
        return q_vec / q_norm

    def search(self, query: str, top_k: int = 5, svd_weight: float = 0.5, pr_alpha: float = 0.35):
        q_unit = self._vectorize_query(query)
        if q_unit is None:
            return []

        # 1. Kelime tabanlı benzerlik (TF-IDF)
        tfidf_scores = self.tfidf_norm @ q_unit

        # 2. Kavram tabanlı anlamsal benzerlik (SVD)
        q_concept = self.Vt_k @ q_unit
        q_concept_norm = np.linalg.norm(q_concept)
        if q_concept_norm > 0:
            svd_scores = self.doc_concepts_norm @ (q_concept / q_concept_norm)
            svd_scores = np.maximum(0.0, svd_scores)  # Negatif kosinüs açılarını sıfırla
        else:
            svd_scores = np.zeros(len(self.dataset), dtype=np.float32)

        # 3. İçerik alakası + Çarpımsal PageRank desteği
        relevance = (1.0 - svd_weight) * tfidf_scores + svd_weight * svd_scores
        final_scores = relevance * (1.0 + pr_alpha * self.pr_norm)

        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if final_scores[idx] > 0:
                results.append({
                    "title": self.dataset[idx]["title"],
                    "final": float(final_scores[idx]),
                    "tfidf": float(tfidf_scores[idx]),
                    "svd": float(svd_scores[idx]),
                    "pr": float(self.pr_norm[idx]),
                })
        return results


if __name__ == "__main__":
    engine = HybridSearchEngine("dataset.json", svd_k=40)
    print("\n=== HİBRİT ARAMA MOTORU HAZIR (TF-IDF + SVD + PageRank) ===")

    while True:
        q = input("\nAra (çıkmak için q) > ").strip()
        if q.lower() == "q":
            break

        results = engine.search(q, top_k=5)
        if not results:
            print("  Sonuç bulunamadı.")
            continue

        for rank, r in enumerate(results, 1):
            print(
                f"  {rank}. [{r['final']:.4f}] {r['title']} "
                f"(TF-IDF: {r['tfidf']:.2f} | SVD: {r['svd']:.2f} | PR: {r['pr']:.2f})"
            )