import json
from collections import Counter
import numpy as np

# Önceki yazdığın modülü doğrudan yeniden kullanıyoruz
from tfidf_engine import build_index, tokenize, search as search_tfidf


def build_svd_index(tfidf_norm: np.ndarray, k: int = 40):
    # ------------------------------------------------------------------
    # TODO 1: Thin SVD hesapla -> A = U @ diag(S) @ Vt
    # İpucu: np.linalg.svd(tfidf_norm, full_matrices=False)
    # ------------------------------------------------------------------
    U, S, Vt = np.linalg.svd(tfidf_norm, full_matrices=False)

    # ------------------------------------------------------------------
    # TODO 2: İlk k boyutu kes (Truncated SVD)
    # - U_k  -> şekli (150, k)
    # - S_k  -> şekli (k,)
    # - Vt_k -> şekli (k, 28008)
    # ------------------------------------------------------------------
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Bilgi koruma oranı: İlk k tekil değerin kareleri toplamı / tüm kareler toplamı
    energy_ratio = np.sum(S_k ** 2) / np.sum(S ** 2)
    print(f"SVD tamamlandı! Boyut: {tfidf_norm.shape[1]} -> {k} kavram")
    print(f"Korunan toplam bilgi (enerji) oranı: %{energy_ratio * 100:.2f}")

    # ------------------------------------------------------------------
    # TODO 3: Belgelerin k-boyutlu kavram matrisini oluştur ve satırları normalize et
    # 1) doc_concepts = U_k * S_k  (broadcasting ile her sütunu kendi tekil değeriyle ölçekle)
    # 2) Her satırı kendi L2 normuna böl (sıfır norm korumasıyla)
    # ------------------------------------------------------------------
    doc_concepts = U_k * S_k
    norms = np.linalg.norm(doc_concepts, axis = 1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_concepts_norm = doc_concepts / norms

    return doc_concepts_norm, Vt_k


def search_svd(query: str, dataset: list[dict], vocab: dict, idf_vector: np.ndarray,
               doc_concepts_norm: np.ndarray, Vt_k: np.ndarray, top_k: int = 5):
    tokens = tokenize(query)
    if not tokens:
        return []

    q_vec = np.zeros(len(vocab), dtype=np.float32)
    counts = Counter(tokens)
    for word, count in counts.items():
        if word in vocab:
            col_idx = vocab[word]
            q_vec[col_idx] = (count / len(tokens)) * idf_vector[col_idx]

    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return []
    q_unit = q_vec / q_norm

    # ------------------------------------------------------------------
    # TODO 4:
    # 1) Sorgu vektörünü k-boyutlu kavram uzayına iz düşür: q_concept = Vt_k @ q_unit
    # 2) q_concept vektörünü kendi L2 normuna böl (q_concept_norm == 0 ise [] dön)
    # 3) doc_concepts_norm ile matris-vektör çarpımı (@) yapıp top_k indeksi sırala
    # ------------------------------------------------------------------
    q_concept = Vt_k @ q_unit
    q_concept_norm = np.linalg.norm(q_concept)
    if q_concept_norm == 0:
        return []
    q_concept_unit = q_concept / q_concept_norm

    scores = doc_concepts_norm @ q_concept_unit
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(float(scores[i]), dataset[i]["title"]) for i in top_indices if scores[i] > 0]


if __name__ == "__main__":
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    vocab, idf_vector, tfidf_norm = build_index(dataset)
    doc_concepts_norm, Vt_k = build_svd_index(tfidf_norm, k=40)

    while True:
        q = input("\nAra (çıkmak için q) > ").strip()
        if q.lower() == "q":
            break

        tfidf_res = search_tfidf(q, dataset, vocab, idf_vector, tfidf_norm, top_k=5)
        svd_res = search_svd(q, dataset, vocab, idf_vector, doc_concepts_norm, Vt_k, top_k=5)

        print("\n  [KLASİK TF-IDF]")
        for rank, (score, title) in enumerate(tfidf_res, 1):
            print(f"    {rank}. [{score:.4f}] {title}")

        print("\n  [ANLAMSAL SVD (k=40)]")
        for rank, (score, title) in enumerate(svd_res, 1):
            print(f"    {rank}. [{score:.4f}] {title}")