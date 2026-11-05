import re
import subprocess
from typing import Any

import httpx
from fastapi import HTTPException

try:
    from .config import get_settings
except ImportError:
    from config import get_settings

GITHUB_API_BASE_URL = "https://api.github.com"


def _get_github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }
    api_key = get_settings().github_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_default_repo() -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            if url.endswith(".git"):
                url = url[:-4]
            parts = url.replace(":", "/").split("/")
            if len(parts) >= 2:
                return parts[-2], parts[-1]
    except Exception:
        pass

    try:
        repos = fetch_repositories(per_page=1)
        if isinstance(repos, list) and len(repos) > 0:
            full_name = repos[0].get("full_name", "")
            if "/" in full_name:
                owner, repo = full_name.split("/", 1)
                return owner, repo
    except Exception:
        pass

    return None, None


def _github_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    url = endpoint if endpoint.startswith("http") else f"{GITHUB_API_BASE_URL}{endpoint}"
    clean_params = {k: v for k, v in (params or {}).items() if v is not None}

    try:
        response = httpx.get(
            url,
            headers=_get_github_headers(),
            params=clean_params,
            timeout=15.0,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Network error connecting to GitHub: {exc}",
        ) from exc

    if response.status_code != 200:
        try:
            detail = response.json().get("message", response.text)
        except Exception:
            detail = response.text
        status_code = response.status_code if response.status_code in {400, 401, 403, 404, 422} else 502
        raise HTTPException(
            status_code=status_code,
            detail=f"GitHub API error: {detail}",
        )

    return response.json()


def fetch_repositories(
    owner: str | None = None,
    visibility: str = "all",
    sort: str = "updated",
    per_page: int = 30,
    page: int = 1,
) -> list[dict[str, Any]]:
    if owner:
        endpoint = f"/users/{owner}/repos"
        params = {"sort": sort, "per_page": per_page, "page": page}
    else:
        endpoint = "/user/repos"
        params = {"visibility": visibility, "sort": sort, "per_page": per_page, "page": page}
    return _github_get(endpoint, params=params)


list_all_repos = fetch_repositories


def fetch_pull_requests(
    owner: str | None = None,
    repo: str | None = None,
    state: str = "open",
    sort: str = "created",
    direction: str = "desc",
    per_page: int = 30,
    page: int = 1,
) -> list[dict[str, Any]]:
    if not owner or not repo:
        def_owner, def_repo = _get_default_repo()
        owner = owner or def_owner
        repo = repo or def_repo

    if not owner or not repo:
        raise HTTPException(
            status_code=400,
            detail="Both 'owner' and 'repo' are required to fetch pull requests.",
        )

    endpoint = f"/repos/{owner}/{repo}/pulls"
    params = {
        "state": state,
        "sort": sort,
        "direction": direction,
        "per_page": per_page,
        "page": page,
    }
    return _github_get(endpoint, params=params)


def fetch_issues(
    owner: str | None = None,
    repo: str | None = None,
    state: str = "open",
    sort: str = "created",
    direction: str = "desc",
    per_page: int = 30,
    page: int = 1,
) -> list[dict[str, Any]]:
    if owner and repo:
        endpoint = f"/repos/{owner}/{repo}/issues"
    elif not owner and not repo:
        def_owner, def_repo = _get_default_repo()
        if def_owner and def_repo:
            endpoint = f"/repos/{def_owner}/{def_repo}/issues"
        else:
            endpoint = "/user/issues"
    else:
        def_owner, def_repo = _get_default_repo()
        owner = owner or def_owner
        repo = repo or def_repo
        if owner and repo:
            endpoint = f"/repos/{owner}/{repo}/issues"
        else:
            raise HTTPException(
                status_code=400,
                detail="Both 'owner' and 'repo' must be provided when querying a specific repository.",
            )

    params = {
        "state": state,
        "sort": sort,
        "direction": direction,
        "per_page": per_page,
        "page": page,
    }
    return _github_get(endpoint, params=params)


def fetch_commits(
    owner: str | None = None,
    repo: str | None = None,
    sha: str | None = None,
    path: str | None = None,
    author: str | None = None,
    since: str | None = None,
    until: str | None = None,
    per_page: int = 30,
    page: int = 1,
) -> list[dict[str, Any]]:
    if not owner or not repo:
        def_owner, def_repo = _get_default_repo()
        owner = owner or def_owner
        repo = repo or def_repo

    if not owner or not repo:
        raise HTTPException(
            status_code=400,
            detail="Both 'owner' and 'repo' are required to fetch commits.",
        )

    endpoint = f"/repos/{owner}/{repo}/commits"
    params = {
        "sha": sha,
        "path": path,
        "author": author,
        "since": since,
        "until": until,
        "per_page": per_page,
        "page": page,
    }
    return _github_get(endpoint, params=params)


def fetch_contributors(
    owner: str | None = None,
    repo: str | None = None,
    anon: bool = False,
    per_page: int = 30,
    page: int = 1,
) -> list[dict[str, Any]]:
    if not owner or not repo:
        def_owner, def_repo = _get_default_repo()
        owner = owner or def_owner
        repo = repo or def_repo

    if not owner or not repo:
        raise HTTPException(
            status_code=400,
            detail="Both 'owner' and 'repo' are required to fetch contributors.",
        )

    endpoint = f"/repos/{owner}/{repo}/contributors"
    params = {
        "anon": "true" if anon else "false",
        "per_page": per_page,
        "page": page,
    }
    return _github_get(endpoint, params=params)


def fetch_github_context(prompt: str) -> dict[str, Any]:
    """Check for GitHub keywords in prompt and fetch relevant data."""
    prompt_lower = prompt.lower()
    context: dict[str, Any] = {}

    # Check for pull requests
    if re.search(r"\b(pr|prs|pull request|pull requests|pulls)\b", prompt_lower):
        try:
            prs = fetch_pull_requests(per_page=5)
            context["pull_requests"] = [
                {
                    "number": pr.get("number"),
                    "title": pr.get("title"),
                    "state": pr.get("state"),
                    "author": pr.get("user", {}).get("login"),
                    "created_at": pr.get("created_at"),
                }
                for pr in prs[:5]
            ]
        except Exception as exc:
            context["pull_requests_error"] = str(exc)

    # Check for issues
    if re.search(r"\b(issue|issues|bug|bugs)\b", prompt_lower):
        try:
            issues = fetch_issues(per_page=5)
            context["issues"] = [
                {
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "state": issue.get("state"),
                    "author": issue.get("user", {}).get("login"),
                    "created_at": issue.get("created_at"),
                }
                for issue in issues[:5]
            ]
        except Exception as exc:
            context["issues_error"] = str(exc)

    # Check for commits
    if re.search(r"\b(commit|commits)\b", prompt_lower):
        try:
            commits = fetch_commits(per_page=5)
            context["commits"] = [
                {
                    "sha": c.get("sha", "")[:7],
                    "message": c.get("commit", {}).get("message", "").split("\n")[0],
                    "author": c.get("commit", {}).get("author", {}).get("name"),
                    "date": c.get("commit", {}).get("author", {}).get("date"),
                }
                for c in commits[:5]
            ]
        except Exception as exc:
            context["commits_error"] = str(exc)

    # Check for contributors
    if re.search(r"\b(contributor|contributors|author|authors)\b", prompt_lower):
        try:
            contributors = fetch_contributors(per_page=5)
            context["contributors"] = [
                {
                    "login": contributor.get("login"),
                    "contributions": contributor.get("contributions"),
                }
                for contributor in contributors[:5]
            ]
        except Exception as exc:
            context["contributors_error"] = str(exc)

    # Check for repositories / general GitHub keywords if no specific entity matched
    if not context and re.search(r"\b(repo|repos|repository|repositories|github)\b", prompt_lower):
        try:
            repos = fetch_repositories(per_page=5)
            context["repositories"] = [
                {
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                }
                for repo in repos[:5]
            ]
        except Exception as exc:
            context["repositories_error"] = str(exc)

    return context
