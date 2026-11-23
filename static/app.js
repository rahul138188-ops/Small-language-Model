'use strict';
const form = document.querySelector('#story-form');
const promptInput = document.querySelector('#prompt');
const temperature = document.querySelector('#temperature');
const generateButton = document.querySelector('#generate');
const generateLabel = document.querySelector('#generate-label');
const errorBox = document.querySelector('#error');
const story = document.querySelector('#story');
const status = document.querySelector('#status');
const copyButton = document.querySelector('#copy');
let busy = false;
function updateCount() { document.querySelector('#count').textContent = `${promptInput.value.length} / 500`; }
promptInput.addEventListener('input', updateCount);
temperature.addEventListener('input', () => { document.querySelector('#temperature-value').textContent = Number(temperature.value).toFixed(1); });
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { promptInput.value = button.dataset.prompt; updateCount(); promptInput.focus(); }));
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  const prompt = promptInput.value.trim();
  errorBox.hidden = true;
  if (!prompt) { errorBox.textContent = 'Write a story beginning or choose a suggestion.'; errorBox.hidden = false; return; }
  busy = true;
  generateButton.disabled = true;
  copyButton.disabled = true;
  generateLabel.textContent = 'Writing your story…';
  document.querySelector('.result').setAttribute('aria-busy', 'true');
  status.textContent = 'The model is writing. This can take a moment on a shared CPU.';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    // Relative URL: the page and Python API run on the same server.
    const response = await fetch('/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, signal: controller.signal,
      body: JSON.stringify({prompt, max_tokens: Number(document.querySelector('#length').value), temperature: Number(temperature.value)})
    });
    let data;
    try { data = await response.json(); } catch { throw new Error('The server is starting or unavailable. Please try again shortly.'); }
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : 'Please check your prompt and settings, then try again.';
      throw new Error(detail);
    }
    if (typeof data.text !== 'string' || !data.text.trim()) throw new Error('No story was returned. Try another prompt.');
    story.textContent = data.text;
    story.hidden = false;
    document.querySelector('#empty').hidden = true;
    copyButton.textContent = 'Copy text';
    status.textContent = 'Your story is ready. Try another opening for something different.';
  } catch (error) {
    errorBox.textContent = error.name === 'AbortError' ? 'The request took too long. Wait a moment before trying again; the server may still be finishing.' : (error instanceof TypeError ? 'Could not reach the server. Check your connection and try again.' : error.message);
    errorBox.hidden = false;
    status.textContent = story.hidden ? 'No story generated yet.' : 'Your previous story is still shown.';
  } finally {
    clearTimeout(timeout); busy = false; generateButton.disabled = false;
    copyButton.disabled = story.hidden;
    generateLabel.textContent = 'Generate story';
    document.querySelector('.result').setAttribute('aria-busy', 'false');
  }
});
copyButton.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(story.textContent); copyButton.textContent = 'Copied!'; status.textContent = 'Story copied to your clipboard.'; }
  catch { status.textContent = 'Clipboard access is unavailable. Select the story text and copy it manually.'; }
});
