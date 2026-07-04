"""
Scrapes SEBI Investor Education pages (investor.sebi.gov.in) into data/raw/sebi_articles.json.

Licensing: no robots.txt found at the site root (404, checked 2026-07-03/07-04) and no explicit
copyright/reproduction prohibition found on the pages checked. Same reasonable-effort standard applied
as NCFE: proceed as a secondary source, always cite back to source_url in every answer. See
CHECKPOINT.md for the full licensing check.

Scope: 6 known investor-education article pages (not FAQ-shaped like NCFE — each is a short article).
Each page's real content lives in the second `.container` element on the page (the first `.container`
is just the breadcrumb nav) — confirmed by inspecting all 6 pages before writing this scraper.
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ARTICLE_URLS = [
    "https://investor.sebi.gov.in/moneymatters.html",
    "https://investor.sebi.gov.in/personalinvestments.html",
    "https://investor.sebi.gov.in/personalsecurities.html",
    "https://investor.sebi.gov.in/bullandbeermarket.html",
    "https://investor.sebi.gov.in/dabba_trading.html",
    "https://investor.sebi.gov.in/cautiontoinvestor.html",
]

REQUEST_DELAY_SECONDS = 1.5
USER_AGENT = "DomainGPT-student-project/0.1 (personal-finance RAG corpus build; contact via GitHub repo)"

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "sebi_articles.json"


def fetch_page(session, url):
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"  [skip] failed to fetch {url}: {e}")
        return None


def clean_title(raw_title):
    # Titles look like ":: Topic of Interest | Caution to Investor | SEBI Investor ::" or
    # ":: Personal Finance & Investments : Money matters: Lets Understand |SEBI Investor | ::"
    # Strip the "::" wrapper and the trailing "SEBI Investor" site-name segment.
    text = raw_title.strip().strip(":").strip()
    parts = [p.strip() for p in text.split("|") if p.strip()]
    parts = [p for p in parts if "sebi investor" not in p.lower()]
    return " - ".join(parts) if parts else text


def extract_article(html, source_url):
    soup = BeautifulSoup(html, "html.parser")
    containers = soup.select(".container")
    if len(containers) < 2:
        print(f"  [skip] unexpected structure (found {len(containers)} .container elements): {source_url}")
        return None

    content_container = containers[1]
    text = content_container.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    title = clean_title(soup.title.get_text()) if soup.title else source_url

    return {
        "title": title,
        "text": text,
        "source_url": source_url,
    }


def scrape_all():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    articles = []
    for url in ARTICLE_URLS:
        print(f"Fetching {url}...")
        html = fetch_page(session, url)
        if html is None:
            continue

        article = extract_article(html, url)
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
    scrape_all()
