import json

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from config import get_settings
from github_utils import fetch_github_context

app = FastAPI()

client = OpenAI(
    api_key=get_settings().groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)


class Req(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query")
def answer_query(req: Req):
    github_context = fetch_github_context(req.prompt)

    if github_context:
        prompt_with_context = (
            f"Context from GitHub:\n"
            f"{json.dumps(github_context, indent=2)}\n\n"
            f"User Question:\n{req.prompt}"
        )
    else:
        prompt_with_context = req.prompt

    response = client.responses.create(
        input=prompt_with_context,
        model="openai/gpt-oss-20b",
    )
    return {
        "prompt": req.prompt,
        "github_data": github_context,
        "response": response.output_text,
    }
