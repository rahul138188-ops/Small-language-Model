"""Start with STORYGPT_CHECKPOINT set to a checkpoint path."""
import os, threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field
from generate import load, complete

@asynccontextmanager
async def lifespan(app):
    app.state.model,app.state.tok,app.state.meta=load(os.getenv('STORYGPT_CHECKPOINT','checkpoints/tiny/best.pt'),os.getenv('STORYGPT_DEVICE','cpu'))
    app.state.lock=threading.Lock()
    yield
app=FastAPI(title='StoryForge generation API',lifespan=lifespan)
class Request(BaseModel):
    prompt: str=Field(min_length=1,max_length=4000)
    max_tokens: int=Field(default=100,ge=1,le=512)
    temperature: float=Field(default=.8,gt=0,le=3)
    top_k: int=Field(default=40,ge=0)
    top_p: float=Field(default=.95,gt=0,le=1)
@app.get('/health')
def health(): return {'status':'ready','checkpoint':app.state.meta['label'],'step':app.state.meta['step']}
@app.post('/generate')
def generate(r:Request):
    with app.state.lock:
        text=complete(app.state.model,app.state.tok,r.prompt,max_new_tokens=r.max_tokens,temperature=r.temperature,top_k=r.top_k,top_p=r.top_p)
    return {'text':text}
