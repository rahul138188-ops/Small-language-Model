FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 TORCH_NUM_THREADS=2
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-web.txt
COPY model.py generate.py web_app.py ./
COPY static ./static
COPY checkpoints/tiny/best.pt ./checkpoints/tiny/best.pt
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["sh", "-c", "exec python -m uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
