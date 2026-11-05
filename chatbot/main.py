from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from config import get_settings

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
    response = client.responses.create(
        input=req.prompt,
        model="openai/gpt-oss-20b",
    )
    return {
        "Hello": "World",
        "prompt": req.prompt,
        "api_key": get_settings().groq_api_key,
        "response": response.output_text,
    }
