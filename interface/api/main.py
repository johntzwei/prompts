from typing import List
import time
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from redis import Redis
from rq import Queue

from worker_helper import inference

TIMEOUT = 10.

# api
app = FastAPI()
q = Queue(connection=Redis())

@app.get("/")
async def main(inputs: str):
    job = q.enqueue(inference, inputs)

    start = time.time()
    while time.time() - start < TIMEOUT:
        time.sleep(1)

        if job.result is not None:
            return job.result

    return 'timeout'
