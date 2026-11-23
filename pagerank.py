import json
import numpy as np


def build_transition_matrix(dataset: list[dict]) -> np.ndarray:
    """
    Dataset'teki iç bağlantılardan sütun-stokastik Geçiş Matrisi (M) üretir.
    M[i, j] = j sayfasından i sayfasına geçme olasılığı.
    """
    n = len(dataset)
    title_to_id = {doc["title"]: idx for idx, doc in enumerate(dataset)}

    # Komşuluk matrisi (Adjacency Matrix): j -> i linki varsa adj[i, j] = 1.0
    adj = np.zeros((n, n), dtype=np.float64)
    for j, doc in enumerate(dataset):
        for target_title in doc["links"]:
            if target_title in title_to_id:
                i = title_to_id[target_title]
                adj[i, j] = 1.0

    # ------------------------------------------------------------------
    # TODO 1: Sütun-stokastik geçiş matrisi M'yi oluştur
    # 1) Her sütunun toplamını (çıkış link sayısını) bul -> axis=0
    # 2) Toplamı 0 olan sütunları (dangling düğümleri) tamamen 1.0 yap:
    #    İpucu: adj[:, col_sums == 0] = 1.0
    # 3) Güncel sütun toplamlarını tekrar hesapla ve adj matrisini
    #    sütun toplamlarına böl (broadcasting ile tek satırda!)
    # ------------------------------------------------------------------
    col_sums = np.sum(adj, axis=0)
    adj[:, col_sums == 0] = 1.0
    col_sums = np.sum(adj, axis=0)
    M = adj / col_sums

    return M


def compute_pagerank(M: np.ndarray, damping: float = 0.85, tol: float = 1e-6, max_iter: int = 100):
    n = M.shape[0]

    # ------------------------------------------------------------------
    # TODO 2: Başlangıç vektörü v'yi oluştur
    # Her sayfaya eşit 1/n olasılık ver (İpucu: np.ones(n) / n)
    # ------------------------------------------------------------------
    v = np.ones(n, dtype=np.float64) / n;

    for iteration in range(1, max_iter + 1):
        # --------------------------------------------------------------
        # TODO 3: Power Iteration adımı ve yakınsama kontrolü
        # 1) Formül: v_new = damping * (M @ v) + (1.0 - damping) / n
        # 2) Değişim miktarı (L1 normu): diff = np.sum(np.abs(v_new - v))
        # 3) v'yi v_new olarak güncelle
        # 4) Eğer diff < tol ise kaçıncı iterasyonda yakınsadığını yazdırıp break yap
        # --------------------------------------------------------------
        v_new = damping * (M @ v) + (1.0 - damping) / n
        diff = np.sum(np.abs(v_new - v))
        v = v_new
        if diff < tol:
            print(f"PageRank {iteration}. iterasyonda yakınsadı! (son fark: {diff:.2e})")
            break

    return v


if __name__ == "__main__":
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    M = build_transition_matrix(dataset)

    # Doğrulama 1: Her sütunun toplamı 1.0 olmalı (min ve max 1.0 çıkmalı)
    col_check = np.sum(M, axis=0)
    print(f"Sütun toplamları kontrolü -> min: {col_check.min():.4f}, max: {col_check.max():.4f}")

    pr_scores = compute_pagerank(M)

    # Doğrulama 2: Tüm PageRank skorlarının toplamı 1.0 (olasılık dağılımı) olmalı
    print(f"PageRank toplam olasılık kontrolü: {np.sum(pr_scores):.6f}\n")

    print("=== MİNİ-WİKİPEDİA EN OTORİTER 10 SAYFA (PageRank) ===")
    top_10 = np.argsort(pr_scores)[::-1][:10]
    for rank, idx in enumerate(top_10, 1):
        title = dataset[idx]["title"]
        print(f"  {rank:2d}. [{pr_scores[idx]:.5f}] {title}")