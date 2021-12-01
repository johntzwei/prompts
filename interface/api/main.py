import json
from typing import List
import time
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from redis import Redis
from rq import Queue

from worker_helper import inference

TIMEOUT = 5.

# api
app = FastAPI()
q = Queue(connection=Redis())

@app.get("/")
async def main(inputs: str, choices: str):
    choices = json.loads(choices)
    job = q.enqueue(inference, inputs)

    start = time.time()
    while time.time() - start < TIMEOUT:
        time.sleep(0.01)

        if job.result is not None:
            return job.result

    return 'timeout'
