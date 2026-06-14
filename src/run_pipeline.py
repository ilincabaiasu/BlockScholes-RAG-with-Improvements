"""Entry-point script for the Block Scholes RAG system.

Usage:
  python -m src.run_pipeline --ingest
  python -m src.run_pipeline --query "What was BTC IV in November 2024?"
  python -m src.run_pipeline --query "..." --pipeline enhanced
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block Scholes RAG — ingest documents or query the pipeline."
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingest_all() then index_all().",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Run the pipeline(s) on TEXT and print results.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["gemini", "baseline", "enhanced"],
        default=None,
        help="Which pipeline to run (default: all three).",
    )
    args = parser.parse_args()

    if not args.ingest and not args.query:
        parser.print_help()
        sys.exit(0)

    # 1. Verify Qdrant is reachable before doing anything else
    from src.config.qdrant_client import verify_connection
    await verify_connection()

    # ------------------------------------------------------------------
    # --ingest
    # ------------------------------------------------------------------
    if args.ingest:
        from src.ingestion.ingestor import ingest_all
        from src.indexing.indexer import index_all

        print("=== Ingesting PDFs ===")
        ingest_summary = ingest_all()
        print(json.dumps(ingest_summary, indent=2))

        print("\n=== Indexing documents ===")
        index_summary = await index_all()
        print(json.dumps(index_summary, indent=2))

    # ------------------------------------------------------------------
    # --query
    # ------------------------------------------------------------------
    if args.query:
        query = args.query
        pipelines_to_run = (
            [args.pipeline] if args.pipeline else ["gemini", "baseline", "enhanced"]
        )

        for pipeline_name in pipelines_to_run:
            if pipeline_name == "gemini":
                from src.pipelines.gemini_pipeline import run
            elif pipeline_name == "baseline":
                from src.pipelines.baseline_pipeline import run
            else:
                from src.pipelines.enhanced_pipeline import run

            result = await run(query)

            print(f"Pipeline: {result.pipeline}")
            print(f"Answer: {result.response_text}")
            print(f"Sources cited: {result.context_sources}")
            print(f"Visual path used: {result.visual_path_used}")
            print(f"Latency: {result.latency_breakdown}")
            print("---")


if __name__ == "__main__":
    asyncio.run(main())
