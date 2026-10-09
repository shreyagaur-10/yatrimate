from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def tavily_search(query):
    if not TAVILY_API_KEY:
        return "Hotel search unavailable: TAVILY_API_KEY is missing. Please check Booking.com or Hotels.com."

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=5)

        raw_results = response.get("results", [])
        if not raw_results:
            return "No hotel results found. Please check Booking.com or Airbnb for options."

        results = []
        for i, r in enumerate(raw_results, 1):
            title   = r.get("title", "Unknown")
            url     = r.get("url", "")
            snippet = r.get("content", "").strip()
            # Keep only the first 300 characters to avoid wall-of-text
            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

            results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

        return "\n\n".join(results)

    except Exception as e:
        print(f"[tavily_search] Error: {e}")
        return "Hotel search is temporarily unavailable. Please check Booking.com or Airbnb for hotel options."
