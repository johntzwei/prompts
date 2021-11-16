from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import Model
from pydantic import BaseModel


class PromptData(BaseModel):
    prompt: str


app = FastAPI()

model = Model("distilgpt2")

origins = [
    "http://localhost:3000",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "HOLLAAAAA"}


@app.post("/evaluate-prompt/")
def evaluate_prompt(data: PromptData):
    prompt = data.prompt
    # TODO: switch out dummy model and result
    res = model.evaluate(prompt)
    return {"prompt": prompt,
            "result": res}
