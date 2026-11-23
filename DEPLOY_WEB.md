# Run and publish the story generator

This adds a website to your existing model project. It does not retrain or replace your model.
The frontend is HTML/CSS/JavaScript; FastAPI serves it and loads the existing best.pt once
on startup. The browser posts JSON to /generate on the same host. No separate frontend
hosting, API keys, Node installation or CORS setup is needed.

## Add to your existing repository

Copy these files/folders from this ZIP into the folder that contains model.py:

- web_app.py
- static/ (all three files)
- requirements-web.txt
- Dockerfile
- .dockerignore
- render.yaml
- DEPLOY_WEB.md
- tests/test_web_app.py (optional)

Do not replace your edited README.md or the whole tests folder. This ZIP deliberately
contains only the new files. Keep model.py, generate.py and checkpoints/tiny/best.pt
from your existing repository.

## Test on Windows

Open PowerShell in your repository root:

```powershell
cd "C:\Users\rahul\Downloads\StoryForge_Complete_Repo"
# Only if the virtual environment does not already exist:
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 in your browser. Do not double-click static/index.html;
it must be served by Python so /generate reaches the API.
Click a suggested prompt and then Generate story. Leave the terminal running.
Use Ctrl+C to stop. If port 8000 is already in use, replace it with 8001 and open
http://127.0.0.1:8001 instead.

Optional automated test (needs pytest and httpx):

```powershell
.\.venv\Scripts\python.exe -m pip install pytest httpx
.\.venv\Scripts\python.exe -m pytest tests/test_web_app.py -q
```

## Push the website files

If you recently edited files on GitHub, first run git pull origin main. If Git reports
conflicts, resolve them before proceeding; do not force-push.

```powershell
git add web_app.py static requirements-web.txt Dockerfile .dockerignore render.yaml DEPLOY_WEB.md tests/test_web_app.py
git add -f checkpoints/tiny/best.pt
git commit -m "Add story generator frontend and deployment setup"
git push origin main
```

Verify Dockerfile, web_app.py and static are at the repository root on GitHub.
Verify checkpoints/tiny/best.pt is present; do not upload last.pt or your .venv.

## Deploy on Render

1. Sign in at https://dashboard.render.com and connect your GitHub account.
2. Select New > Web Service (not Static Site).
3. Select rahul138188-ops/Small-language-Model.
4. Choose branch main and Docker as the runtime/language.
5. Leave Root Directory empty if Dockerfile is beside model.py at repository root.
6. Dockerfile path: ./Dockerfile. Keep Docker command at its default (Dockerfile CMD).
7. Choose the Free instance if available to your account. Do not select a paid plan
   unless you want its charges.
8. If shown, set health check path to /health. No secret or API key is required.
9. Select Deploy Web Service. Wait for the build and startup logs to show ready.
10. Open the actual onrender.com URL returned by Render and generate a story.

render.yaml is an alternative Blueprint configuration; you do not need a Blueprint
when following the manual Web Service flow above. Docker uses Render's PORT variable.
The container uses CPU-only PyTorch, one worker and two compute threads.

The deployment has not been created for you: your Render account must authorize the
repository and deployment. Free web services sleep after inactivity and can take time
to wake. See https://render.com/docs/free for current limits. If logs show an out-of-memory
error, the instance memory is insufficient; use a host with more memory or explicitly
choose a larger plan. The Docker build itself was not executed in the preparation
environment; API integration was tested with the actual model on CPU.

## What connects to what?

1. GET / returns static/index.html.
2. The page loads /static/styles.css and /static/app.js.
3. app.js sends POST /generate with prompt, max_tokens and temperature.
4. web_app.py calls complete() from generate.py using your loaded model.
5. The API returns {"text": "..."}; JavaScript shows it using textContent.

The API allows one active generation per worker and returns 429 when busy instead of
building an unbounded queue. It bounds prompt length, token output and temperature.
It is a public demo, without accounts or persistent story history. Your current model
can make grammatical mistakes; deploying the app does not improve the model's training.

## Replace the model after training

Replace checkpoints/tiny/best.pt with your new compatible checkpoint, commit it and push.
Redeploy if your host does not deploy automatically. The API and page do not need changes.
If paths changed elsewhere in the repository, update your old SHA256SUMS.json or remove
that old archive inventory; it describes the original ZIP, not the updated web app.
