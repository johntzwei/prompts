from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import Model

app = FastAPI()

model = Model("distilgpt2")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "HOLLAAAAA"}


@app.get("/evaluate-prompt/{prompt}")
async def evaluate_prompt(prompt):
    res = model.evaluate(prompt)
    return {"Prompt": prompt,
            "Result": res}
