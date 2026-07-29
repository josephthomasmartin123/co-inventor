"""
Web search tools for Co-Inventor agents.

Three search backends, used in priority order:
1. Exa (exa.ai) — best for recent research papers and scientific advances
2. Tavily — great general web search with AI summarization
3. DuckDuckGo HTML — free fallback, no key required

The WEB_SEARCH_TOOL dict is the Anthropic/OpenRouter tool_use schema.
execute_search() is called by the agentic loop when Claude invokes the tool.
"""
from __future__ import annotations

import re

import httpx

from app.config import settings

# Tool schema passed to Claude in the tools= parameter
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for patents, academic papers, technical articles, or recent advances. "
        "Use this to: (1) check prior art for an invention idea, (2) find recent breakthroughs "
        "in a technical domain, (3) discover cross-domain solutions. "
        "Returns up to 5 results with title, URL, and snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query. Be specific — include technical terms and domain. "
                    "For recent advances, include year ranges like '2022 2023 2024 breakthrough'. "
                    "Example: 'nanobubble rheology modifier concrete workability 2023 patent'."
                ),
            }
        },
        "required": ["query"],
    },
}


async def execute_search(query: str) -> list[dict]:
    """
    Execute a web search using the best available backend.
    Returns list of {title, url, snippet}.
    Search is best-effort — never raises exceptions (returns [] on failure).
    """
    # Exa is best for recent research; try first if key available
    if settings.exa_api_key:
        try:
            results = await _exa_search(query)
            if results:
                return results
        except Exception:
            pass

    # Tavily is good general search
    if settings.tavily_api_key:
        try:
            results = await _tavily_search(query)
            if results:
                return results
        except Exception:
            pass

    # DuckDuckGo HTML fallback — no key required
    return await _ddg_search(query)


async def _exa_search(query: str) -> list[dict]:
    """
    Exa neural search — excels at finding recent research papers and technical advances.
    Uses Exa's /search endpoint with neural mode.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": settings.exa_api_key,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "num_results": 5,
                "type": "neural",               # Neural search better for concepts
                "use_autoprompt": True,         # Exa rewrites query for better results
                "contents": {
                    "text": {"max_characters": 400},
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("text", "")[:400],
                "published_date": r.get("publishedDate", ""),
            })
        return results


async def _tavily_search(query: str) -> list[dict]:
    """Tavily AI search — good general search with answer synthesis."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:400],
            }
            for r in data.get("results", [])
        ]


async def _ddg_search(query: str) -> list[dict]:
    """DuckDuckGo HTML scrape — free fallback, no API key required."""
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; co-inventor-bot/1.0)"},
            )
            html = resp.text

        results = []
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)', html)
        urls = re.findall(r'class="result__url"[^>]*>([^<]+)', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)

        for i in range(min(5, len(titles))):
            title = titles[i].strip() if i < len(titles) else ""
            url = urls[i].strip() if i < len(urls) else ""
            raw_snippet = snippets[i] if i < len(snippets) else ""
            snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()[:400]
            if title or url:
                results.append({"title": title, "url": url, "snippet": snippet})

        return results
    except Exception:
        return []
