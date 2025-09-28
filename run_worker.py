from rq import SimpleWorker, Queue
from redis import Redis

redis_conn = Redis()

q = Queue(connection=redis_conn)
worker = SimpleWorker(queues=[q], connection=redis_conn)
worker.work()
