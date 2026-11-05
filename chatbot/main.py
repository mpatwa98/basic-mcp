import json
from typing import Any

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from config import get_settings
from github_utils import (
    fetch_commits,
    fetch_contributors,
    fetch_issues,
    fetch_pull_requests,
    fetch_repositories,
)

app = FastAPI()

client = OpenAI(
    api_key=get_settings().groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_repositories",
            "description": "Fetch GitHub repositories for a user/owner or the authenticated user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "GitHub username or organization. If omitted, fetches authenticated user repos.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pull_requests",
            "description": "Fetch pull requests for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization).",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "State of pull requests (default 'open').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_issues",
            "description": "Fetch issues for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization).",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name.",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "State of issues (default 'open').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_commits",
            "description": "Fetch commits for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization).",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name.",
                    },
                    "author": {
                        "type": "string",
                        "description": "GitHub login or email of author to filter commits.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_contributors",
            "description": "Fetch contributors for a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or organization).",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name.",
                    },
                },
            },
        },
    },
]

AVAILABLE_TOOLS = {
    "fetch_repositories": fetch_repositories,
    "fetch_pull_requests": fetch_pull_requests,
    "fetch_issues": fetch_issues,
    "fetch_commits": fetch_commits,
    "fetch_contributors": fetch_contributors,
}


def sanitize_tool_output(tool_name: str, data: Any) -> Any:
    """Format and limit tool data so LLM context stays concise."""
    if not isinstance(data, list):
        return data

    if tool_name == "fetch_pull_requests":
        return [
            {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "state": pr.get("state"),
                "author": pr.get("user", {}).get("login"),
                "created_at": pr.get("created_at"),
            }
            for pr in data[:10]
        ]
    if tool_name == "fetch_issues":
        return [
            {
                "number": issue.get("number"),
                "title": issue.get("title"),
                "state": issue.get("state"),
                "author": issue.get("user", {}).get("login"),
                "created_at": issue.get("created_at"),
            }
            for issue in data[:10]
        ]
    if tool_name == "fetch_commits":
        return [
            {
                "sha": c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name"),
                "date": c.get("commit", {}).get("author", {}).get("date"),
            }
            for c in data[:10]
        ]
    if tool_name == "fetch_contributors":
        return [
            {
                "login": c.get("login"),
                "contributions": c.get("contributions"),
            }
            for c in data[:10]
        ]
    if tool_name == "fetch_repositories":
        return [
            {
                "name": r.get("name"),
                "full_name": r.get("full_name"),
                "description": r.get("description"),
            }
            for r in data[:10]
        ]
    return data[:10]


class Req(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query")
def answer_query(req: Req):
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to GitHub tools. "
                "Use the tools when the user asks questions that require GitHub data. "
                "If the user does not specify owner/repo, you can omit them and the tool will use the default repository."
            ),
        },
        {"role": "user", "content": req.prompt},
    ]

    # Step 1: Send user prompt and tool definitions to the LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    executed_tools: list[dict[str, Any]] = []

    # Step 2: Check if the model decided to call any tools
    if response_message.tool_calls:
        messages.append(response_message)

        # Step 3: Execute each requested tool and collect the output
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            raw_args = tool_call.function.arguments
            args = json.loads(raw_args) if raw_args else {}

            tool_fn = AVAILABLE_TOOLS.get(func_name)
            if tool_fn:
                try:
                    tool_result = tool_fn(**args)
                    formatted_result = sanitize_tool_output(func_name, tool_result)
                except Exception as exc:
                    formatted_result = {"error": str(exc)}
            else:
                formatted_result = {"error": f"Tool '{func_name}' not found."}

            executed_tools.append(
                {
                    "name": func_name,
                    "arguments": args,
                    "result": formatted_result,
                }
            )

            # Step 4: Add tool response message back to the conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(formatted_result),
                }
            )

        # Step 5: Second LLM call to synthesize the final answer using tool results
        follow_up = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        final_answer = follow_up.choices[0].message.content
    else:
        final_answer = response_message.content

    return {
        "prompt": req.prompt,
        "tool_calls": executed_tools,
        "response": final_answer,
    }
