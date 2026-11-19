"""Backwards-compatible entry point.

The former stand-alone ``/chat`` service duplicated the application (its own app, config and RAG
pipeline). There is now a single backend: ``teamdocs.api.app`` (Main Agent at ``POST /api/v1/chat``).
``uvicorn devmind.api.main:app`` keeps working and serves that same application.
"""
from teamdocs.api.app import create_app

app = create_app()
