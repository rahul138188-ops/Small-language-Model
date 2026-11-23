"""Serve the story generator page and its model API together."""
import os
import threading
from pathlib import Path
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from generate import load, complete

ROOT = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(max(1, int(os.getenv('TORCH_NUM_THREADS', '2'))))
    checkpoint = Path(os.getenv('STORYGPT_CHECKPOINT', 'checkpoints/tiny/best.pt'))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise RuntimeError(f'Model missing: {checkpoint}. Copy checkpoints/tiny/best.pt from your original project.')
    app.state.model, app.state.tokenizer, app.state.meta = load(checkpoint, os.getenv('STORYGPT_DEVICE', 'cpu'))
    app.state.lock = threading.Lock()
    yield

app = FastAPI(title='Small Language Model — Story Generator', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')

class StoryRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_tokens: int = Field(default=100, ge=16, le=256)
    temperature: float = Field(default=0.8, ge=0.2, le=1.2)

    @field_validator('prompt')
    @classmethod
    def strip_prompt(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('Please enter a story beginning.')
        return value

@app.get('/', include_in_schema=False)
def homepage():
    return FileResponse(ROOT / 'static' / 'index.html')

@app.get('/health')
def health():
    return {'status': 'ready', 'parameters': app.state.meta['parameters']}

@app.post('/generate')
def generate_story(request: StoryRequest):
    model, tokenizer = app.state.model, app.state.tokenizer
    if len(tokenizer.encode(request.prompt).ids) > model.config.context_length:
        raise HTTPException(422, 'This prompt is too long for the model. Try a shorter story beginning.')
    if not app.state.lock.acquire(blocking=False):
        raise HTTPException(429, 'The model is writing another story. Please try again in a moment.', headers={'Retry-After': '5'})
    try:
        text = complete(model, tokenizer, request.prompt, max_new_tokens=request.max_tokens,
                        temperature=request.temperature, top_k=40, top_p=0.95)
    finally:
        app.state.lock.release()
    return {'text': text}
