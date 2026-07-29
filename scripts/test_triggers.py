"""
Quick test script for the Trigger Discovery agent.

Usage:
  uv run python scripts/test_triggers.py "your problem statement here"
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agents import trigger_discovery


async def main():
    problem = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Improve workability of high-strength concrete without increasing water/cement ratio"
    )

    print(f"\nProblem: {problem}\n")
    print("Discovering triggers...\n")

    async def on_progress(event_type: str, data: dict):
        if event_type == "trigger_found":
            print(f"  [{data['distance']:<10}] [{data['source_domain']:<20}] {data['advance'][:70]}")

    result = await trigger_discovery.run(
        problem_statement=problem,
        session_id="test-session",
        on_progress=on_progress,
        n_triggers=3,
    )

    print(f"\n── {len(result.triggers)} triggers found ──\n")
    for i, t in enumerate(result.triggers, 1):
        print(f"{i}. [{t.distance}] {t.source_domain}")
        print(f"   Advance: {t.advance}")
        print(f"   Mechanism: {t.mechanism[:120]}...")
        print(f"   Relevance: {t.relevance[:120]}...")
        if t.source_url:
            print(f"   Source: {t.source_url}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
