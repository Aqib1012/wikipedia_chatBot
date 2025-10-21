# scraper.py
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document  # for LangChain v0.2+

def scrape_url(url):
    """
    Input: URL (Wikipedia or any web page)
    Output: List of LangChain Document objects
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # Get page content (with headers)
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    # Extract visible text
    text = soup.get_text(separator="\n", strip=True)

    # Split into chunks (1000 chars with 200 overlap)
    chunk_size = 1000
    overlap = 200
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk))
        start += chunk_size - overlap

    return chunks
