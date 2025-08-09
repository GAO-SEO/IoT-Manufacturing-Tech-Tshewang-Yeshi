import os, re, json, time, urllib.parse
import requests
from bs4 import BeautifulSoup
from collections import deque

BASE = "https://iotmanufacturingtech.com"
OUT_JSONL = "data/site_content/pages.jsonl"
SEEN, QUEUE = set([BASE]), deque([BASE])

def same_domain(url):
    return urllib.parse.urlparse(url).netloc.endswith("iotmanufacturingtech.com")

def canonicalize(url):
    u = urllib.parse.urlsplit(url)
    u = u._replace(fragment="", query="")  # drop # and ?params
    return urllib.parse.urlunsplit(u)

def extract(url):
    r = requests.get(url, timeout=20, headers={"User-Agent":"KB-bot/1.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # canonical URL if present
    can = soup.find("link", rel="canonical")
    canon = canonicalize(can["href"]) if can and can.get("href") else url

    # remove nav/footers/aside/scripts
    for sel in ["nav","footer","aside","script","style","noscript","form"]:
        for t in soup.select(sel):
            t.decompose()

    title = (soup.title.string or "").strip() if soup.title else ""
    # get main content text
    body = soup.get_text(separator="\n")
    # collapse whitespace
    body = re.sub(r"\n{2,}", "\n\n", body).strip()

    return {"url": canon, "title": title, "text": body}

def enqueue_links(soup, base_url):
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(base_url, a["href"])
        href = canonicalize(href)
        if same_domain(href) and href not in SEEN:
            SEEN.add(href)
            QUEUE.append(href)

def crawl():
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        while QUEUE:
            url = QUEUE.popleft()
            try:
                r = requests.get(url, timeout=20, headers={"User-Agent":"KB-bot/1.0"})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                page = extract(url)
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
                enqueue_links(soup, url)
                time.sleep(0.5)
            except Exception as e:
                print("skip", url, e)

if __name__ == "__main__":
    crawl()
