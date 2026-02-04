# macrs-ecom-recommender

A MACRS-aligned multi-agent conversational recommender system for e-commerce, with planner/reflection orchestration and RAG search.

## Goals
- Multi-agent candidate generation (ask / recommend / chit-chat) every turn
- Central planner selects a single act and response
- Reflection updates strategy weights between turns
- External retrieval/ranking (RAG) for product candidates
- Full logging and replayable evaluation

## Stack
- Python 3.12
- LangGraph for orchestration
- Groq LLMs via `langchain-groq`
- PostgreSQL + pgvector for product retrieval

## Quick Start
- Configure environment variables in `.env`
- Run ingestion: `python scripts/ingest.py products.csv`
- Run a smoke turn: `python scripts/smoke.py "looking for running shoes"`

## License
TBD
