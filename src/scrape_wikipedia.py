"""
Fetches Wikipedia articles for core personal-finance terms into data/raw/wikipedia_articles.json.

Licensing: Wikipedia text is CC BY-SA — explicitly reusable with attribution (source_url is kept per
article for exactly this reason). Supplementary/fallback source, used for standard term definitions
NCFE/SEBI don't cover as cleanly (e.g. EMI, compound interest).

Uses the MediaWiki API's `extracts` (explaintext=1) endpoint rather than scraping rendered HTML — it
returns the full article body as clean plain text (headings included as `== Heading ==` markers, no
markup/nav/infobox noise to strip), and `redirects=1` follows redirect titles automatically.
"""

import json
import time
from pathlib import Path

import requests

ARTICLE_TITLES = [
    "Systematic investment plan",
    "Public Provident Fund (India)",
    "National Pension System",
    "Equated monthly installment",
    "Mutual fund",
    "Fixed deposit",
    "Compound interest",
    "Credit card",
    "Emergency fund",
]

API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_DELAY_SECONDS = 1.0
USER_AGENT = "DomainGPT-student-project/0.1 (personal-finance RAG corpus build; contact via GitHub repo)"

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "wikipedia_articles.json"


def fetch_article(session, title):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    try:
        response = session.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"  [skip] failed to fetch '{title}': {e}")
        return None

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            print(f"  [skip] '{title}' not found on Wikipedia")
            return None
        extract = page.get("extract", "").strip()
        if not extract:
            return None
        resolved_title = page.get("title", title)
        return {
            "title": resolved_title,
            "text": extract,
            "source_url": f"https://en.wikipedia.org/wiki/{resolved_title.replace(' ', '_')}",
        }
    return None


def fetch_all():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    articles = []
    for title in ARTICLE_TITLES:
        print(f"Fetching '{title}'...")
        article = fetch_article(session, title)
        if article is None:
            continue
        articles.append(article)
        print(f"  '{article['title']}' — {len(article['text'])} chars")
        time.sleep(REQUEST_DELAY_SECONDS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(articles)} articles to {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_all()
