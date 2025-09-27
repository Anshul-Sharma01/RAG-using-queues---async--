from fastapi import FastAPI, Query
from clients.rq_client import queue 
from queues.worker import process_query



app = FastAPI()


@app.get("/")
def root():
    return { "status" : "Server is up and running !!" }


@app.post("/chat")
def chat( query : str = Query(..., description="the chat queryof user")):
    job = queue.enqueue(process_query, query)

    return { "status" : "queued", "job_id" : job.id  }