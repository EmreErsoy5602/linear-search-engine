import json
import re
import numpy as np


def clean_words(text: str) -> list[str]:
    """Metni küçük harfe çevirir ve en az 2 harfli kelimeleri listeler."""
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return re.findall(r"[a-zçğıöşü]{2,}", text)


def analyze_dataset(filepath: str = "dataset.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    num_docs = len(dataset)
    if num_docs == 0:
        print("Veri seti boş!")
        return

    vocabulary = set()  # set() tekrar eden kelimeleri otomatik eler
    word_counts = []
    unique_per_doc = []
    link_counts = []

    for doc in dataset:
        words = clean_words(doc["title"] + " " + doc["text"])
        unique_words = set(words)

        vocabulary.update(unique_words)
        word_counts.append(len(words))
        unique_per_doc.append(len(unique_words))
        link_counts.append(len(doc["links"]))

    # Standart Python listelerini tek tip (int32) NumPy dizilerine çevir
    word_counts_np = np.array(word_counts, dtype=np.int32)
    unique_per_doc_np = np.array(unique_per_doc, dtype=np.int32)
    link_counts_np = np.array(link_counts, dtype=np.int32)

    vocab_size = len(vocabulary)
    total_matrix_elements = num_docs * vocab_size
    non_zero_elements = int(np.sum(unique_per_doc_np))
    sparsity = (1 - (non_zero_elements / total_matrix_elements)) * 100

    # Her hücreyi 32-bit float (4 byte) saklarsak gerekecek bellek
    estimated_ram_mb = (total_matrix_elements * 4) / (1024 * 1024)

    # argmax -> en büyük değerin kendisini değil, İNDEKSİNİ döndürür
    longest_idx = int(np.argmax(word_counts_np))
    most_linked_idx = int(np.argmax(link_counts_np))

    # argmin -> en küçük değerin kendisini değil, İNDEKSİNİ döndürür
    shortest_idx = int(np.argmin(word_counts_np))
    least_linked_idx = int(np.argmin(link_counts_np))

    dangling_nodes = int(np.sum(link_counts_np == 0))

    #median and std
    median_word_count = float(np.median(word_counts_np))
    standard_deviation_word_count = float(np.std(word_counts_np))

    #1000 kelimeden fazlası olan makaleler
    docs_over_1000_words = np.sum(word_counts_np > 1000)

    print("=== VERİ SETİ OTONOM ANALİZ RAPORU ===")
    print(f"Toplam makale sayısı (N)       : {num_docs}")
    print(f"Toplam kelime sayısı           : {np.sum(word_counts_np):,}")
    print(f"Benzersiz kelime / Sözlük (V)  : {vocab_size:,}")
    print(f"Toplam iç bağlantı (link)      : {np.sum(link_counts_np):,}")
    print()
    print("--- İSTATİSTİKLER (NumPy) ---")
    print(f"Makale başı ortalama kelime    : {np.mean(word_counts_np):.1f}")
    print(f"Makale başı ortalama link      : {np.mean(link_counts_np):.1f}")
    print(f"En uzun makale                 : {dataset[longest_idx]['title']} ({word_counts_np[longest_idx]} kelime)")
    print(f"En çok link veren makale       : {dataset[most_linked_idx]['title']} ({link_counts_np[most_linked_idx]} link)")
    print(f"En kısa makale                 : {dataset[shortest_idx]['title']} ({word_counts_np[shortest_idx]} kelime)")
    print(f"En az link veren makale        : {dataset[least_linked_idx]['title']} ({link_counts_np[least_linked_idx]} link)")
    print(f"1000 kelimeden fazla olan makaleler : {docs_over_1000_words}")
    print(f"Kelime sayısı medyanı          : {median_word_count:.1f}")
    print(f"Kelime sayısı standart sapması  : {standard_deviation_word_count:.2f}")
    print(f"Bağlantı sayısı medyanı          : {np.median(link_counts_np):.1f}")
    print(f"Bağlantı sayısı standart sapması  : {np.std(link_counts_np):.2f}")
    print(f"Dangling düğümler              : {dangling_nodes}")
    print()
    print("--- MATRİS PROFİLİ ---")
    print(f"Matris boyutu (N x V)          : {num_docs} x {vocab_size}")
    print(f"Toplam eleman (hücre) sayısı   : {total_matrix_elements:,}")
    print(f"Sıfır olmayan eleman sayısı    : {non_zero_elements:,}")
    print(f"Seyreklik (Sıfır olma oranı)   : %{sparsity:.2f}")
    print(f"Tahmini RAM (float32)          : {estimated_ram_mb:.2f} MB")


if __name__ == "__main__":
    analyze_dataset()