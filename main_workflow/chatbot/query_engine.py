"""
query_engine.py
Sends the user question + schema context to Gemini and extracts
a DuckDB-compatible SQL query from the response.
Uses the modern Google GenAI SDK.
"""

import re
import os
from datetime import date

from google import genai
from google.genai import types


_SYSTEM_TEMPLATE = """\
You are a data analyst assistant for a restaurant expense tracking system.
The spending data is stored in a DuckDB table called `data`.

{schema}

Today's date is {today}.

RULES:
1. Always answer by writing a SQL SELECT query wrapped in <sql>...</sql> tags.
2. Use only DuckDB-compatible SQL syntax.
3. For relative date filters use INTERVAL, e.g.:
     WHERE Invoice_Date >= CURRENT_DATE - INTERVAL '1 month'
4. Column names are case-sensitive - use them exactly as listed above.
   Wrap column names in double quotes if they contain spaces or special chars.
5. After the <sql> block write one short plain-English sentence explaining the result.
6. If the question cannot be answered from the available data, say so clearly and omit the SQL block.
7. Never generate DROP, DELETE, INSERT, UPDATE, CREATE, ALTER, ATTACH, COPY, or TRUNCATE statements.
"""


def _build_system_prompt(schema_context: str) -> str:
    return _SYSTEM_TEMPLATE.format(
        schema=schema_context,
        today=date.today().isoformat(),
    )


def _history_to_text(history: list[dict]) -> str:
    parts = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        parts.append(f"{label}: {content}")
    return "\n\n".join(parts)


def _build_contents(question: str, history: list[dict]) -> str:
    history_text = _history_to_text(history)
    if history_text:
        return (
            "Conversation history:\n"
            f"{history_text}\n\n"
            "Current user question:\n"
            f"{question}"
        )
    return f"Current user question:\n{question}"


def _response_text(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    collected = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) if content else []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                collected.append(part_text)

    return "\n".join(collected).strip()


def get_model(api_key: str | None = None) -> genai.Client:
    """
    Initialise and return a Google GenAI client.
    api_key defaults to env var GEMINI_API_KEY.
    """
    key = api_key or os.environ["GEMINI_API_KEY"]
    return genai.Client(api_key=key)


def ask(
    question: str,
    schema_context: str,
    history: list[dict],
    model: genai.Client,
) -> tuple[str, str | None]:
    """
    Send the question to Gemini with full conversation history.

    Args:
        question       : The user's natural-language question.
        schema_context : Output of schema_probe.build_context_string().
        history        : List of {"role": "user"/"assistant", "content": str}.
                         Modified in-place - current turn appended.
        model          : A genai.Client instance from get_model().

    Returns:
        (reply_text, sql_or_None)
        reply_text - full Gemini response
        sql_or_None - extracted SQL string, or None if no <sql> block present
    """
    system_prompt = _build_system_prompt(schema_context)

    prior_history = list(history)
    history.append({"role": "user", "content": question})

    contents = _build_contents(question, prior_history)

    response = model.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            max_output_tokens=1024,
        ),
    )

    reply = _response_text(response).strip()
    history.append({"role": "assistant", "content": reply})

    match = re.search(r"<sql>(.*?)</sql>", reply, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else None

    return reply, sql
