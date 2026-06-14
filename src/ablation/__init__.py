"""Ablation-testing harness for the enhanced RAG pipeline.

Disables one enhancement at a time (query-time only — no re-indexing) and
scores the 12 evaluation queries with programmatic metrics plus an LLM judge,
so the marginal contribution of each component can be measured.

See ABLATION.md for the dependency map and how to run it.
"""
