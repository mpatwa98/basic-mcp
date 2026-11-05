import json
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI
from pydantic import BaseModel

from config import get_settings

app = FastAPI(title="MCP-Powered GitHub Assistant")

client = OpenAI(
    api_key=get_settings().groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"
MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"


@asynccontextmanager
async def get_mcp_session() -> AsyncGenerator[ClientSession]:
    """Connect to the running MCP server over HTTP; fallback to stdio subprocess if not running."""
    try:
        async with streamable_http_client(MCP_SERVER_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except Exception:
        # Fallback: spawn mcp_server.py as a subprocess via standard input/output (stdio transport)
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_server.py", "--transport", "stdio"],
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


class Req(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    return {
        "service": "MCP-Powered GitHub Chatbot",
        "mcp_server_url": MCP_SERVER_URL,
        "endpoints": ["/query", "/mcp/tools", "/mcp/resources", "/mcp/prompts"],
    }


@app.get("/mcp/tools")
async def list_mcp_tools():
    """Discover tools exposed dynamically by the MCP Server."""
    async with get_mcp_session() as session:
        tools_response = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools_response.tools
        ]


@app.get("/mcp/resources")
async def list_mcp_resources():
    """Discover resources exposed by the MCP Server."""
    async with get_mcp_session() as session:
        resources_response = await session.list_resources()
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
            }
            for resource in resources_response.resources
        ]


@app.get("/mcp/prompts")
async def list_mcp_prompts():
    """Discover prompt templates exposed by the MCP Server."""
    async with get_mcp_session() as session:
        prompts_response = await session.list_prompts()
        return [
            {
                "name": prompt.name,
                "description": prompt.description,
            }
            for prompt in prompts_response.prompts
        ]


@app.post("/query")
async def answer_query(req: Req):
    async with get_mcp_session() as session:
        # Step 1: Dynamically discover tools from the MCP server
        tools_response = await session.list_tools()
        llm_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools_response.tools
        ]

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to tools via Model Context Protocol (MCP). "
                    "Use tools whenever answering requires GitHub data. "
                    "If the user does not specify owner/repo, leave them empty to use default repository."
                ),
            },
            {"role": "user", "content": req.prompt},
        ]

        # Step 2: Send query + MCP-discovered tools to LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=llm_tools if llm_tools else None,
            tool_choice="auto" if llm_tools else None,
        )

        response_message = response.choices[0].message
        executed_tools: list[dict[str, Any]] = []

        # Step 3: If LLM requested tool execution, invoke tool via MCP session
        if response_message.tool_calls:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                args = json.loads(raw_args) if raw_args else {}

                # Execute tool using the MCP Client session
                call_result = await session.call_tool(func_name, args)
                tool_output_text = "\n".join(
                    c.text for c in call_result.content if hasattr(c, "text")
                )

                executed_tools.append(
                    {
                        "tool": func_name,
                        "arguments": args,
                        "output_preview": tool_output_text[:300],
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_output_text,
                    }
                )

            # Step 4: Synthesize final response using tool results
            follow_up = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            final_answer = follow_up.choices[0].message.content
        else:
            final_answer = response_message.content

        return {
            "prompt": req.prompt,
            "mcp_tools_available": [t.name for t in tools_response.tools],
            "executed_tools": executed_tools,
            "response": final_answer,
        }
