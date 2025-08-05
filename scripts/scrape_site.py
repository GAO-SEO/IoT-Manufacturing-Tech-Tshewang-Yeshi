# scrape_site.py
import requests
from bs4 import BeautifulSoup
import os

base_url = "https://iotmanufacturingtech.com"
visited = set()
output_dir = "site_content"
os.makedirs(output_dir, exist_ok=True)

def scrape(url):
    if url in visited or not url.startswith(base_url):
        return
    visited.add(url)
    print("Scraping:", url)

    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text()
        file_name = os.path.join(output_dir, url.replace(base_url, "").strip("/").replace("/", "_") + ".txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(text)

        for a_tag in soup.find_all("a", href=True):
            next_url = a_tag["href"]
            if next_url.startswith("/"):
                next_url = base_url + next_url
            scrape(next_url)
    except Exception as e:
        print("Error scraping:", url, e)

scrape(base_url)
