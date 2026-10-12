import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY is missing. Add it to your .env file."
    )

tavily = TavilyClient(api_key=api_key)


@tool
def web_search(query: str) -> str:
    """Search the web for recent, reliable information about a topic.
    Returns titles, URLs, and snippets."""

    query = query.strip()

    if not query:
        return "Search failed: the query is empty. Provide a research topic."

    # Reject queries containing only site filters, such as:
    # site:arxiv.org OR site:github.com
    search_terms = [
        word for word in query.split()
        if not word.lower().startswith("site:")
        and word.upper() not in {"OR", "AND"}
    ]

    if not search_terms:
        return (
            "Search failed: the query contains only website filters. "
            "Include meaningful search terms, for example: "
            "'RAG techniques 2026 site:arxiv.org'."
        )

    try:
        response = tavily.search(
            query=query,
            max_results=5,
        )

        results = response.get("results", [])

        if not results:
            return f"No search results found for: {query}"

        output = []

        for result in results:
            output.append(
                f"Title: {result.get('title', 'Untitled')}\n"
                f"URL: {result.get('url', 'No URL')}\n"
                f"Snippet: {result.get('content', '')[:500]}"
            )

        return "\n\n---\n\n".join(output)

    except Exception as e:
        return (
            f"Web search failed: {type(e).__name__}: {e}"
        )


@tool
def scrape_url(url: str) -> str:
    """Scrape a webpage and return clean text for deeper reading."""

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script", "style", "nav", "footer",
            "header", "noscript",
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if not text:
            return f"No readable text found at: {url}"

        return text[:5000]

    except requests.RequestException as e:
        return f"Could not scrape URL: {e}"

    except Exception as e:
        return f"Scraping failed: {type(e).__name__}: {e}"