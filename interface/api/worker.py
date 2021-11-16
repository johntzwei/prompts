import sys
from rq import Connection, SimpleWorker
from worker_helper import inference, init

with Connection():
    qs = sys.argv[1:] or ['default']
    init()#I initialize model before workers start
    w = SimpleWorker(qs)#I use SimpleWorker because it does not fork
    w.work()
