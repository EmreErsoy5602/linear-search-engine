import json
import random
import time
from collections import deque
import requests

API_URL = "https://tr.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "LinearAlgebraSearchProject/1.0 (https://github.com/example/linear-project; student@example.com)"
}

session = requests.Session()
session.headers.update(HEADERS)


def fetch_page(title: str, max_retries: int = 3) -> dict | None:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|links",
        "explaintext": 1,
        "pllimit": "max",
        "plnamespace": 0,
        "redirects": 1,
    }

    for attempt in range(max_retries):
        try:
            response = session.get(API_URL, params=params, timeout=10)

            if response.status_code == 429:
                raw_retry = response.headers.get("Retry-After", "")
                retry_after = int(raw_retry) if raw_retry.isdigit() else 5 * (attempt + 1)
                print(f"Hız sınırı (429)! {retry_after} sn bekleniyor... ({title})")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()
            break
        except requests.RequestException as e:
            print(f"İstek hatası ({title}): {e}")
            time.sleep(2)
            return None
    else:
        print(f"Atlandı (çok fazla 429): {title}")
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page_data = next(iter(pages.values()))
    if "missing" in page_data:
        return None

    text = page_data.get("extract", "").strip()
    if len(text) < 300:
        return None

    links = [link["title"] for link in page_data.get("links", [])]

    return {
        "title": page_data["title"],
        "text": text,
        "links": links,
    }


def crawl_wikipedia(seed_title: str, max_pages: int = 150, max_enqueue_per_page: int = 15) -> list[dict]:
    queue = deque([seed_title])
    visited = set()
    collected = {}

    try:
        while queue and len(collected) < max_pages:
            current_title = queue.popleft()
            if current_title in visited:
                continue
            visited.add(current_title)

            page = fetch_page(current_title)
            time.sleep(1.0)  # Hata olsa da olmasa da her zaman 1 saniye bekle

            if not page:
                continue

            canonical_title = page["title"]
            if canonical_title in collected:
                continue

            collected[canonical_title] = page
            visited.add(canonical_title)
            print(f"[{len(collected)}/{max_pages}] İndirildi: {canonical_title} ({len(page['text'])} karakter)")

            # Alfabetik sapmayı önlemek için linklerden rastgele seç
            unvisited_links = [link for link in page["links"] if link not in visited]
            sample_size = min(len(unvisited_links), max_enqueue_per_page)
            for link in random.sample(unvisited_links, sample_size):
                queue.append(link)

    except KeyboardInterrupt:
        print("\nDurduruldu! Şimdiye kadar indirilen sayfalar kaydediliyor...")

    valid_titles = set(collected.keys())
    dataset = []

    for doc_id, (title, page) in enumerate(collected.items()):
        internal_links = [
            link for link in page["links"]
            if link in valid_titles and link != title
        ]
        dataset.append({
            "id": doc_id,
            "title": title,
            "text": page["text"],
            "links": list(set(internal_links)),
        })

    return dataset


if __name__ == "__main__":
    # 150 makale ~2.5 dakika sürer ve TF-IDF + SVD + PageRank için fazlasıyla yeterlidir
    data = crawl_wikipedia(seed_title="Bilgisayar bilimi", max_pages=150)

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_edges = sum(len(doc["links"]) for doc in data)
    print(f"\nTamamlandı! {len(data)} makale ve {total_edges} iç bağlantı 'dataset.json' dosyasına kaydedildi.")