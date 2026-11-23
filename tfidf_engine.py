import json
import re
from collections import Counter
import numpy as np

STOP_WORDS = {
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "da", "de", "ki", "için",
    "gibi", "daha", "en", "çok", "az", "her", "hiç", "ama", "fakat", "ancak",
    "çünkü", "olarak", "olan", "olduğu", "olanlar", "tarafından", "üzere",
    "kadar", "sonra", "önce", "ise", "hem", "ne", "ya", "mi", "mı", "mu", "mü",
    "var", "yok", "diğer", "aynı", "bazı", "tüm", "bütün", "birçok", "birkaç",
    "ayrıca", "böylece", "dolayı", "göre", "karşı", "rağmen", "şekilde"
}


def tokenize(text: str) -> list[str]:
    text = text.replace("İ", "i").replace("I", "ı").lower()
    words = re.findall(r"[a-zçğıöşü]{2,}", text)
    return [w for w in words if w not in STOP_WORDS]


def build_index(dataset: list[dict]):
    num_docs = len(dataset)
    tokenized_docs = [tokenize(doc["title"] + " " + doc["text"]) for doc in dataset]

    # Sözlük (kelime -> sütun indeksi) ve her kelimenin kaç belgede geçtiği (doc_freq)
    vocab = {}
    doc_freq = Counter()
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] += 1
            if token not in vocab:
                vocab[token] = len(vocab)

    vocab_size = len(vocab)

    # TF Matrisi (num_docs x vocab_size) ve df_array (vocab_size,)
    tf_matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
    for doc_idx, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue
        counts = Counter(tokens)
        total = len(tokens)
        for word, count in counts.items():
            tf_matrix[doc_idx, vocab[word]] = count / total

    df_array = np.zeros(vocab_size, dtype=np.float32)
    for word, col_idx in vocab.items():
        df_array[col_idx] = doc_freq[word]

    # ------------------------------------------------------------------
    # TODO 1: IDF vektörünü hesapla -> shape: (vocab_size,)
    # Formül: ln((N + 1) / (df + 1)) + 1.0  (İpucu: np.log)
    # ------------------------------------------------------------------
    idf_vector = np.log((num_docs + 1) / (df_array + 1)) + 1.0

    # ------------------------------------------------------------------
    # TODO 2: TF-IDF matrisini oluştur -> shape: (num_docs, vocab_size)
    # İpucu: Broadcasting ile tek satırda element-wise çarpım (*)
    # ------------------------------------------------------------------
    tfidf_matrix = tf_matrix * idf_vector

    # ------------------------------------------------------------------
    # TODO 3: Her belge satırını kendi L2 normuna böl (birim vektör yap)
    # İpucu: np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    # Dikkat: Sıfıra bölme (NaN) riskine karşı normu 0 olan satırları koru
    #         (örneğin norms[norms == 0] = 1.0 yapıp sonra bölebilirsin)
    # ------------------------------------------------------------------
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf_norm = tfidf_matrix / norms

    return vocab, idf_vector, tfidf_norm


def search(query: str, dataset: list[dict], vocab: dict, idf_vector: np.ndarray, tfidf_norm: np.ndarray, top_k: int = 5):
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
    # 1) Kosinüs benzerliği skorlarını tek matris-vektör çarpımı (@) ile bul
    # 2) En yüksek skorlu top_k belgenin indekslerini büyükten küçüğe al
    #    (İpucu: np.argsort(scores)[::-1][:top_k])
    # ------------------------------------------------------------------
    
    scores = tfidf_norm @ q_unit
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(float(scores[i]), dataset[i]["title"]) for i in top_indices if scores[i] > 0]


if __name__ == "__main__":
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    vocab, idf_vector, tfidf_norm = build_index(dataset)
    print(f"İndeks hazır! Matris şekli: {tfidf_norm.shape}")

    # Doğrulama testi: İlk 3 belge vektörünün uzunluğu (normu) 1.0 çıkmalı
    print("İlk 3 satırın norm kontrolü:", np.linalg.norm(tfidf_norm[:3], axis=1))

    while True:
        q = input("\nAra (çıkmak için q) > ").strip()
        if q.lower() == "q":
            break
        results = search(q, dataset, vocab, idf_vector, tfidf_norm)
        if not results:
            print("  Sonuç bulunamadı.")
        for rank, (score, title) in enumerate(results, 1):
            print(f"  {rank}. [{score:.4f}] {title}")