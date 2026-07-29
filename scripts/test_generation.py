"""
Quick test script for the Generation agent in isolation.

Usage:
  uv run python scripts/test_generation.py

Tests the generation agent without spinning up the full server.
Set ANTHROPIC_API_KEY (or OPENROUTER_API_KEY) in .env first.
"""
import asyncio
import sys
import os

# Ensure we run from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agents import generation


async def main():
    problem = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Reduce biofouling on ship hulls without using toxic antifouling paints"
    )

    print(f"\nProblem: {problem}\n")
    print("Running generation (no triggers)...\n")

    async def on_progress(event_type: str, data: dict):
        if event_type == "invention_generated":
            print(f"  [{data['strategy']:<20}] {data['title']}")

    result = await generation.run(
        problem_statement=problem,
        session_id="test-session",
        on_progress=on_progress,
        triggers=[],
    )

    print(f"\n── {len(result.inventions)} inventions generated ──\n")
    for i, inv in enumerate(result.inventions, 1):
        print(f"{i}. [{inv.strategy}] {inv.title}")
        print(f"   Summary: {inv.summary[:100]}...")
        print(f"   Mechanism: {inv.mechanism[:100]}...")
        print()


if __name__ == "__main__":
    asyncio.run(main())
