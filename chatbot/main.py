from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Req(BaseModel):
    prompt: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query")
def answer_query(req: Req):
    return {"Hello": "World", "prompt": req.prompt}
