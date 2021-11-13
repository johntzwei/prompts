from fastapi import FastAPI
from model import Model

app = FastAPI()

model = Model("distilgpt2")


@app.get("/")
async def root():
    return {"message": "HOLLAAAAA"}


@app.get("/evaluate-prompt/{prompt}")
async def evaluate_prompt(prompt):
    res = model.evaluate(prompt)
    return {"Prompt": prompt,
            "Result": res}
