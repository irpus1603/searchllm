import requests
from bs4 import BeautifulSoup
import re
import json
import datetime
import tiktoken
from urllib.parse import urljoin, urlparse
import time

# Initialize tokenizer (for chunking text)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Settings
MAX_PAGES = 10  # Increase if you want to scrape more links
visited_urls = set()  # To prevent duplicate crawling

def get_text_from_url(url):
    """Scrapes and cleans text from a given URL."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract and clean text
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces

        return text.strip()

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def chunk_text(text, max_tokens=512):
    """Chunks text into segments with a max token size."""
    tokens = tokenizer.encode(text)
    chunks = [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks

def create_rag_json(url, text_chunks):
    """Formats scraped and chunked data into RAG-friendly JSON structure."""
    return {
        "source": url,
        "title": f"Scraped Content from {url}",
        "date": str(datetime.datetime.utcnow()),
        "chunks": [
            {"chunk_id": f"{url}_chunk_{i+1}", "content": chunk}
            for i, chunk in enumerate(text_chunks)
        ]
    }

def get_all_links(url, domain):
    """Finds all internal links on a given page."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()

        for a_tag in soup.find_all("a", href=True):
            link = urljoin(url, a_tag["href"])
            parsed_link = urlparse(link)

            # Keep only internal links (same domain)
            if parsed_link.netloc == domain and link not in visited_urls:
                links.add(link)

        return links

    except Exception as e:
        print(f"Error getting links from {url}: {e}")
        return set()

def crawl_website(start_url, max_pages=10):
    """Crawls a website, scraping multiple pages recursively."""
    domain = urlparse(start_url).netloc
    to_visit = {start_url}  # Start with the main page
    crawled_data = []

    while to_visit and len(visited_urls) < max_pages:
        url = to_visit.pop()

        if url in visited_urls:
            continue  # Skip already visited URLs

        print(f"Scraping: {url}")
        visited_urls.add(url)

        # Scrape content
        raw_text = get_text_from_url(url)

        if raw_text:
            text_chunks = chunk_text(raw_text)
            structured_data = create_rag_json(url, text_chunks)
            crawled_data.append(structured_data)

        # Extract all internal links and add to the queue
        new_links = get_all_links(url, domain)
        to_visit.update(new_links - visited_urls)

        time.sleep(1)  # Avoid overwhelming the server

    return crawled_data

def sanitize_filename(url):
    """Converts a URL into a safe filename."""
    parsed_url = urlparse(url)
    filename = parsed_url.netloc + parsed_url.path
    filename = filename.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")
    return filename

def save_json(data, start_url):
    """Saves structured data to a JSON file with a sanitized filename."""
    filename = f"datacrawl/{sanitize_filename(start_url)}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Data saved to {filename}")

# ---- RUN SCRIPT ----
if __name__ == "__main__":
    start_url = "https://ioh.co.id/portal/id/bsbloglanding"  # Change this URL
    data = crawl_website(start_url, MAX_PAGES)

    if data:
        save_json(data, start_url)
