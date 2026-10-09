import os
import time
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


# =========================
# Retry Helper
# =========================

def invoke_llm_with_retry(messages, max_retries=3, wait_seconds=5):
    """Invoke LLM with retry logic for rate limit and transient errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "rate_limit" in error_str or "413" in error_str or "429" in error_str
            is_last_attempt = attempt == max_retries - 1

            if is_last_attempt:
                raise  # Re-raise on final attempt

            if is_rate_limit:
                wait = wait_seconds * (attempt + 1)  # Exponential-style backoff
                print(f"[Retry {attempt + 1}/{max_retries}] Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                # For non-rate-limit errors, retry once quickly
                time.sleep(2)


# =========================
# State
# =========================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =========================
# Flight Agent
# =========================

def flight_agent(state: TravelState):
    query = state["user_query"]
    try:
        flight_data = search_flights(query)
        if not flight_data:
            flight_data = "No flight data available at this time. Please check flight booking sites directly."
    except Exception as e:
        print(f"[flight_agent] Error fetching flights: {e}")
        flight_data = "Flight search is temporarily unavailable. Please check Google Flights or Skyscanner for options."

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight results fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Hotel Agent
# =========================

def hotel_agent(state: TravelState):
    try:
        query = f"Best hotels for {state['user_query']}"
        hotel_results = tavily_search(query)
        if not hotel_results:
            hotel_results = "No hotel data available at this time. Please check Booking.com or Hotels.com."
    except Exception as e:
        print(f"[hotel_agent] Error fetching hotels: {e}")
        hotel_results = "Hotel search is temporarily unavailable. Please check Booking.com or Airbnb for options."

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information fetched.")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Itinerary Agent
# =========================

def itinerary_agent(state: TravelState):
    # Truncate inputs to stay within token limits
    flight_results = state['flight_results'][:800]
    hotel_results = state['hotel_results'][:800]

    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{flight_results}

Hotel Results:
{hotel_results}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    try:
        response = invoke_llm_with_retry([
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt)
        ])
        itinerary_content = response.content
        messages = [response]
    except Exception as e:
        print(f"[itinerary_agent] LLM error: {e}")
        itinerary_content = (
            f"Itinerary generation failed temporarily. "
            f"Based on your query '{state['user_query']}', please plan your trip using the flight "
            f"and hotel information provided. Consider visiting local attractions and booking in advance."
        )
        messages = [AIMessage(content=itinerary_content)]

    return {
        "itinerary": itinerary_content,
        "messages": messages,
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Final Response Agent
# =========================

def final_agent(state: TravelState):
    # Truncate large fields to stay within model token limits (free tier: 8K TPM)
    flight_results = state['flight_results'][:1000]
    hotel_results = state['hotel_results'][:1000]
    itinerary = state['itinerary'][:2000]

    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{flight_results}

Hotels:
{hotel_results}

Itinerary:
{itinerary}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    try:
        response = invoke_llm_with_retry([
            SystemMessage(content="You are a professional AI travel booking assistant."),
            HumanMessage(content=final_prompt)
        ])
        messages = [response]
    except Exception as e:
        print(f"[final_agent] LLM error: {e}")
        # Graceful fallback — return partial info instead of crashing
        fallback = (
            f"## ✈️ Trip Summary\n\n"
            f"Here is your travel plan for: **{state['user_query']}**\n\n"
            f"### Flight Information\n{state['flight_results'][:500] or 'Please check Google Flights.'}\n\n"
            f"### Hotel Suggestions\n{state['hotel_results'][:500] or 'Please check Booking.com.'}\n\n"
            f"### Itinerary\n{state['itinerary'][:800] or 'Please plan based on your destination.'}\n\n"
            f"*Note: AI summary is temporarily unavailable. The above is raw data from our sources.*"
        )
        messages = [AIMessage(content=fallback)]

    return {
        "messages": messages,
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        result = travel_graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "flight_results": "",
                "hotel_results": "",
                "itinerary": "",
                "llm_calls": 0
            },
            config=config
        )

        final_answer = result["messages"][-1].content

        return {
            "thread_id": thread_id,
            "answer": final_answer,
            "flight_results": result.get("flight_results", ""),
            "hotel_results": result.get("hotel_results", ""),
            "itinerary": result.get("itinerary", ""),
            "llm_calls": result.get("llm_calls", 0),
        }

    except Exception as e:
        print(f"[run_travel_agent] Unexpected error: {e}")
        return {
            "thread_id": thread_id,
            "answer": (
                f"We encountered a temporary issue while planning your trip for '{user_input}'. "
                f"Please try again in a moment."
            ),
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        }
