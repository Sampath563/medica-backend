# backend/serpapi_helper.py

from serpapi import GoogleSearch
import os

def fetch_search_results(query: str):
    params = {
        "engine": "google",
        "q": query,
        "location": "India",
        "api_key": os.getenv("SERPAPI_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    print(f"DEBUG: Raw search results: {results}")
    top_results = []

    for result in results.get("organic_results", [])[:5]:
        top_results.append({
            "title": result.get("title"),
            "snippet": result.get("snippet")
        })

    return top_results
