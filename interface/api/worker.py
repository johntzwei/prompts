import sys
from rq import Connection, SimpleWorker
from worker_helper import inference, init

model_card = 'EleutherAI/gpt-neo-2.7B'
model_card = 't5-large'

with Connection():
    qs = sys.argv[1:] or ['default']
    init(model_card)#I initialize model before workers start
    w = SimpleWorker(qs)#I use SimpleWorker because it does not fork
    w.work()
