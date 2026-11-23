"""Run from the project root: python -m pytest tests/test_web_app.py -q"""
from fastapi.testclient import TestClient
from web_app import app

def test_page_and_real_model():
    with TestClient(app) as client:
        page = client.get('/')
        assert page.status_code == 200 and 'Start your story' in page.text
        assert client.get('/static/app.js').status_code == 200
        assert client.get('/static/styles.css').status_code == 200
        assert client.get('/health').json()['parameters'] == 1433280
        response = client.post('/generate', json={'prompt': 'A little rabbit', 'max_tokens': 16, 'temperature': .8})
        assert response.status_code == 200
        assert isinstance(response.json()['text'], str) and response.json()['text']
        assert client.post('/generate', json={'prompt': '   '}).status_code == 422
        assert client.post('/generate', json={'prompt': 'Hello', 'max_tokens': 10000}).status_code == 422
        assert client.post('/generate', json={'prompt': 'Hello', 'temperature': 0}).status_code == 422
        assert client.post('/generate', json={'prompt': 'x'*501}).status_code == 422
        app.state.lock.acquire()
        try:
            assert client.post('/generate', json={'prompt': 'Hello'}).status_code == 429
        finally:
            app.state.lock.release()
