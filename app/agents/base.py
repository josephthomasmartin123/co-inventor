"""
Core LLM calling infrastructure for Co-Inventor agents.

Supports two LLM providers:
- Anthropic (direct) — set ANTHROPIC_API_KEY
- OpenRouter (openai-compatible) — set OPENROUTER_API_KEY

The key function is run_claude_with_tools() — an async agentic loop
that handles tool_use (web search) automatically across multiple rounds.

on_tool_call: optional async callback fired on every web search:
    await on_tool_call(agent_name, query, results)
Use this to emit SSE events so the UI can show what's being searched.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from app.config import settings
from app.tools.web_search import WEB_SEARCH_TOOL, execute_search

logger = logging.getLogger(__name__)


class AgentCallError(Exception):
    pass


async def run_claude_with_tools(
    *,
    model: str,
    system: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    on_tool_call: Optional[Callable] = None,   # async (agent, query, results) -> None
    agent_name: str = "agent",                  # identifies which agent is searching
    max_tool_rounds: int = 6,
) -> tuple[str, list[dict]]:
    """
    Core agentic loop. Handles tool_use automatically.

    Returns (final_text, search_results_accumulated).
    Fires on_tool_call(agent_name, query, results) on each web search.
    """
    if settings.use_openrouter:
        return await _run_openrouter(
            model=model, system=system, messages=messages, tools=tools,
            on_text=on_text, on_tool_call=on_tool_call, agent_name=agent_name,
            max_tool_rounds=max_tool_rounds,
        )
    else:
        return await _run_anthropic(
            model=model, system=system, messages=messages, tools=tools,
            on_text=on_text, on_tool_call=on_tool_call, agent_name=agent_name,
            max_tool_rounds=max_tool_rounds,
        )


async def _dispatch_search(
    query: str,
    agent_name: str,
    on_tool_call: Optional[Callable],
) -> tuple[list[dict], str]:
    """Execute a web search and fire the on_tool_call callback."""
    results = await execute_search(query)
    if on_tool_call:
        try:
            await on_tool_call(agent_name, query, results)
        except Exception as e:
            logger.debug(f"on_tool_call error: {e}")
    return results, json.dumps(results)


async def _run_anthropic(
    *, model: str, system: str, messages: list[dict],
    tools: Optional[list[dict]], on_text: Optional[Callable],
    on_tool_call: Optional[Callable], agent_name: str,
    max_tool_rounds: int,
) -> tuple[str, list[dict]]:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    active_tools = tools if tools is not None else [WEB_SEARCH_TOOL]
    search_results_accumulated: list[dict] = []
    current_messages = list(messages)

    for _ in range(max_tool_rounds):
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=current_messages,
            tools=active_tools if active_tools else anthropic.NOT_GIVEN,
        )

        text_content = ""
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_content += block.text
                if on_text:
                    on_text(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        if response.stop_reason == "end_turn" or not tool_uses:
            return text_content, search_results_accumulated

        tool_results = []
        for tu in tool_uses:
            if tu.name == "web_search":
                results, result_content = await _dispatch_search(
                    tu.input.get("query", ""), agent_name, on_tool_call
                )
                search_results_accumulated.extend(results)
            else:
                result_content = json.dumps({"error": f"Unknown tool: {tu.name}"})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_content,
            })

        current_messages.append({"role": "assistant", "content": response.content})
        current_messages.append({"role": "user", "content": tool_results})

    raise AgentCallError(f"Exceeded max_tool_rounds={max_tool_rounds}")


async def _run_openrouter(
    *, model: str, system: str, messages: list[dict],
    tools: Optional[list[dict]], on_text: Optional[Callable],
    on_tool_call: Optional[Callable], agent_name: str,
    max_tool_rounds: int,
) -> tuple[str, list[dict]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://co-inventor.app", "X-Title": "Co-Inventor"},
    )

    active_tools = tools if tools is not None else [WEB_SEARCH_TOOL]
    search_results_accumulated: list[dict] = []

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in active_tools
    ] if active_tools else None

    openai_messages = [{"role": "system", "content": system}]
    for msg in messages:
        if isinstance(msg["content"], str):
            openai_messages.append({"role": msg["role"], "content": msg["content"]})
        elif isinstance(msg["content"], list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block["content"],
                    })
                elif hasattr(block, "type") and block.type == "text":
                    openai_messages.append({"role": msg["role"], "content": block.text})

    current_messages = openai_messages

    for _ in range(max_tool_rounds):
        response = await client.chat.completions.create(
            model=model,
            messages=current_messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=4096,
        )

        choice = response.choices[0]
        msg = choice.message
        text_content = msg.content or ""
        if on_text and text_content:
            on_text(text_content)

        tool_calls = msg.tool_calls or []

        if choice.finish_reason in ("stop", "end_turn") or not tool_calls:
            return text_content, search_results_accumulated

        current_messages.append({
            "role": "assistant",
            "content": text_content or None,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            if tc.function.name == "web_search":
                try:
                    args = json.loads(tc.function.arguments)
                    results, result_content = await _dispatch_search(
                        args.get("query", ""), agent_name, on_tool_call
                    )
                    search_results_accumulated.extend(results)
                except Exception as e:
                    result_content = json.dumps({"error": str(e)})
                    results = []
            else:
                result_content = json.dumps({"error": f"Unknown tool: {tc.function.name}"})

            current_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_content,
            })

    raise AgentCallError(f"Exceeded max_tool_rounds={max_tool_rounds}")


def parse_json_response(text: str) -> dict | list:
    """Parse JSON from LLM response, handling markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        cleaned = "\n".join(inner_lines).strip()
    return json.loads(cleaned)
