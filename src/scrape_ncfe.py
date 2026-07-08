"""
Scrapes NCFE (National Centre for Financial Education) FAQ pages into data/raw/ncfe_faqs.json.

Licensing: ncfe.org.in/robots.txt allows full crawling (checked 2026-07-03, `Disallow:` empty) and the
footer copyright notice has no reproduction prohibition (unlike Zerodha Varsity, which was rejected for
exactly that reason). See CHECKPOINT.md for the full licensing check.

Scope: 6 FAQ category pages only. Each is a small, fixed, known set of pages on a stable government site
- no retry/backoff/caching infrastructure needed. If a request fails, it's logged and skipped; just rerun
the script.
"""

import json
import time
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup

# Suppress the InsecureRequestWarning that verify=False triggers on every request below —
# we've already made a deliberate, documented decision about it (see fetch_category_page).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CATEGORY_URLS = [
    "https://ncfe.org.in/faqs/financial-planning/",
    "https://ncfe.org.in/faqs/banking/",
    "https://ncfe.org.in/faqs/financial-literacy/",
    "https://ncfe.org.in/faqs/loan-borrowing/",
    "https://ncfe.org.in/faqs/mutual-fund/",
    "https://ncfe.org.in/faqs/gold/",
]

REQUEST_DELAY_SECONDS = 1.5
USER_AGENT = "DomainGPT-student-project/0.1 (personal-finance RAG corpus build; contact via GitHub repo)"

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ncfe_faqs.json"


def fetch_category_page(session, url):
    try:
        # verify=False: ncfe.org.in's server doesn't send its intermediate certificate
        # (confirmed via `openssl s_client` — return code 21, incomplete chain). The leaf
        # cert itself checks out (CN=*.ncfe.org.in, issued by Sectigo) and this is public,
        # non-sensitive government content, not a channel carrying secrets - so skipping
        # strict verification for this specific known site is a reasonable tradeoff here,
        # not a blanket practice.
        response = session.get(url, timeout=15, verify=False)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"  [skip] failed to fetch {url}: {e}")
        return None


def extract_qa_pairs(html, category, source_url):
    soup = BeautifulSoup(html, "html.parser")
    pairs = []
    for item in soup.select(".elementor-toggle-item"):
        title_el = item.select_one("a.elementor-toggle-title")
        content_el = item.select_one(".elementor-tab-content")
        if not title_el or not content_el:
            continue
        question = title_el.get_text(strip=True)
        answer = content_el.get_text(separator=" ", strip=True)
        if not question or not answer:
            continue
        pairs.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "source_url": source_url,
            }
        )
    return pairs


def scrape_all():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_pairs = []
    seen_questions = set()

    for url in CATEGORY_URLS:
        category = url.rstrip("/").rsplit("/", 1)[-1]
        print(f"Fetching {category} ({url})...")
        html = fetch_category_page(session, url)
        if html is None:
            continue

        pairs = extract_qa_pairs(html, category, url)
        new_count = 0
        for pair in pairs:
            key = (pair["category"], pair["question"])
            if key in seen_questions:
                continue
            seen_questions.add(key)
            all_pairs.append(pair)
            new_count += 1

        print(f"  found {len(pairs)} entries, {new_count} new after dedup")
        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_pairs)} Q&A pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    scrape_all()
