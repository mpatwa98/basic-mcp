import argparse
import json

from mcp.server import MCPServer

from github_utils import (
    _get_default_repo,
    fetch_commits,
    fetch_contributors,
    fetch_issues,
    fetch_pull_requests,
    fetch_repositories,
)

# 1. Initialize the MCP Server
mcp = MCPServer(
    name="github-assistant",
    instructions="Provides tools and resources to interact with GitHub repositories, pull requests, issues, commits, and contributors.",
)


# 2. Expose GitHub Tools to the MCP Server
@mcp.tool()
def get_repositories(owner: str | None = None) -> str:
    """Fetch repositories for the authenticated user or specified GitHub owner."""
    repositories = fetch_repositories(owner=owner)
    summary = [
        {
            "name": repo.get("name"),
            "full_name": repo.get("full_name"),
            "description": repo.get("description"),
            "html_url": repo.get("html_url"),
            "stars": repo.get("stargazers_count"),
        }
        for repo in repositories[:10]
    ]
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_pull_requests(
    owner: str | None = None,
    repo: str | None = None,
    state: str = "open",
) -> str:
    """Fetch pull requests for a repository. Defaults to current repository if owner/repo are omitted."""
    prs = fetch_pull_requests(owner=owner, repo=repo, state=state)
    summary = [
        {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "author": pr.get("user", {}).get("login"),
            "created_at": pr.get("created_at"),
            "html_url": pr.get("html_url"),
        }
        for pr in prs[:10]
    ]
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_issues(
    owner: str | None = None,
    repo: str | None = None,
    state: str = "open",
) -> str:
    """Fetch issues for a repository. Defaults to current repository if owner/repo are omitted."""
    issues = fetch_issues(owner=owner, repo=repo, state=state)
    summary = [
        {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "state": issue.get("state"),
            "author": issue.get("user", {}).get("login"),
            "created_at": issue.get("created_at"),
            "html_url": issue.get("html_url"),
        }
        for issue in issues[:10]
    ]
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_commits(
    owner: str | None = None,
    repo: str | None = None,
    author: str | None = None,
) -> str:
    """Fetch recent commits for a repository. Defaults to current repository if owner/repo are omitted."""
    commits = fetch_commits(owner=owner, repo=repo, author=author)
    summary = [
        {
            "sha": c.get("sha", "")[:7],
            "message": c.get("commit", {}).get("message", "").split("\n")[0],
            "author": c.get("commit", {}).get("author", {}).get("name"),
            "date": c.get("commit", {}).get("author", {}).get("date"),
        }
        for c in commits[:10]
    ]
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_contributors(
    owner: str | None = None,
    repo: str | None = None,
) -> str:
    """Fetch top contributors for a repository. Defaults to current repository if owner/repo are omitted."""
    contributors = fetch_contributors(owner=owner, repo=repo)
    summary = [
        {
            "login": c.get("login"),
            "contributions": c.get("contributions"),
            "html_url": c.get("html_url"),
        }
        for c in contributors[:10]
    ]
    return json.dumps(summary, indent=2)


# 3. Expose an MCP Resource (context data readable by MCP clients)
@mcp.resource("github://current-repo")
def get_current_repo_resource() -> str:
    """Read the current configured default GitHub repository context."""
    owner, repo = _get_default_repo()
    return json.dumps({"owner": owner, "repo": repo})


# 4. Expose an MCP Prompt (reusable prompt template)
@mcp.prompt("analyze-repository")
def analyze_repository_prompt(owner: str, repo: str) -> str:
    """Prompt template to analyze a repository's activity and status."""
    return (
        f"Please analyze the GitHub repository '{owner}/{repo}'. "
        f"Check its recent commits, open pull requests, and issues to summarize its current activity."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio", "sse"],
        default="streamable-http",
        help="MCP transport protocol (default: streamable-http)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for HTTP/SSE transport")
    parser.add_argument("--port", type=int, default=8001, help="Port for HTTP/SSE transport")
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
