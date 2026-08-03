"""
SAE Feature Labeling Tool
Run on GPU: python label_features.py
Then open: http://localhost:8080
"""

import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import urllib.parse

FEATURE_FILE = Path("checkpoints/sae_raca/feature_interpretations.json")

def load_features():
    with open(FEATURE_FILE) as f:
        return json.load(f)

def save_features(data):
    with open(FEATURE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def render_page(features, current_id=None):
    ids = list(features.keys())
    unlabeled = [i for i in ids if features[i]['label'] is None]
    labeled = [i for i in ids if features[i]['label'] is not None]
    
    if current_id is None:
        current_id = unlabeled[0] if unlabeled else ids[0]
    
    f = features[current_id]
    docs_html = ""
    for doc in f.get('top_activating_docs', []):
        snippet = doc['text_snippet'].replace('<', '&lt;').replace('>', '&gt;')
        docs_html += f"""
        <div class="doc">
            <div class="activation">activation: {doc['activation']:.3f}</div>
            <div class="snippet" dir="rtl">{snippet}</div>
        </div>"""

    label_val = f['label'] or ''
    hal_true = 'checked' if f['is_hallucination_feature'] == True else ''
    hal_false = 'checked' if f['is_hallucination_feature'] == False else ''

    nav_html = ""
    for fid in ids:
        is_current = fid == current_id
        is_done = features[fid]['label'] is not None
        style = "current" if is_current else ("done" if is_done else "")
        nav_html += f'<a href="/?id={fid}" class="nav-item {style}">{fid}</a>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SAE Feature Labeler</title>
<style>
  body {{ font-family: sans-serif; margin: 0; display: flex; height: 100vh; background: #0d1117; color: #e6edf3; }}
  .sidebar {{ width: 200px; overflow-y: auto; background: #161b22; padding: 10px; flex-shrink: 0; }}
  .sidebar h3 {{ color: #58a6ff; margin-top: 0; }}
  .progress {{ font-size: 12px; color: #8b949e; margin-bottom: 10px; }}
  .nav-item {{ display: block; padding: 4px 8px; margin: 2px 0; border-radius: 4px; text-decoration: none; color: #8b949e; font-size: 13px; }}
  .nav-item:hover {{ background: #21262d; color: #e6edf3; }}
  .nav-item.current {{ background: #1f6feb; color: white; }}
  .nav-item.done {{ color: #3fb950; }}
  .main {{ flex: 1; padding: 24px; overflow-y: auto; }}
  h1 {{ color: #58a6ff; margin-top: 0; }}
  .stats {{ color: #8b949e; font-size: 13px; margin-bottom: 20px; }}
  .doc {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
  .activation {{ color: #f78166; font-size: 12px; margin-bottom: 6px; }}
  .snippet {{ font-size: 14px; line-height: 1.6; color: #e6edf3; }}
  .form {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin-top: 20px; }}
  label {{ display: block; margin-bottom: 8px; font-weight: bold; color: #58a6ff; }}
  input[type=text] {{ width: 100%; padding: 8px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; color: #e6edf3; font-size: 14px; box-sizing: border-box; }}
  .radio-group {{ display: flex; gap: 20px; margin: 12px 0; }}
  .radio-group label {{ font-weight: normal; color: #e6edf3; display: flex; align-items: center; gap: 6px; }}
  button {{ background: #1f6feb; color: white; border: none; padding: 10px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; margin-top: 8px; }}
  button:hover {{ background: #388bfd; }}
  .skip {{ background: #21262d; margin-left: 8px; }}
  .skip:hover {{ background: #30363d; }}
</style>
</head>
<body>
<div class="sidebar">
  <h3>Features</h3>
  <div class="progress">{len(labeled)}/{len(ids)} labeled</div>
  {nav_html}
</div>
<div class="main">
  <h1>Feature {current_id}</h1>
  <div class="stats">
    Max activation: {f['max_activation']:.3f} &nbsp;|&nbsp;
    Frequency: {f['activation_frequency']:.3f} &nbsp;|&nbsp;
    Mean: {f['mean_activation']:.4f}
  </div>
  <h3>Top Activating Documents</h3>
  {docs_html}
  <div class="form">
    <form method="POST" action="/save">
      <input type="hidden" name="id" value="{current_id}">
      <input type="hidden" name="next_id" value="{unlabeled[1] if len(unlabeled) > 1 else (unlabeled[0] if unlabeled else ids[0])}">
      <label>Label (what concept does this feature encode?)</label>
      <input type="text" name="label" value="{label_val}" placeholder="e.g. anti-money laundering regulations" required>
      <label style="margin-top:14px">Is this a hallucination feature?</label>
      <div class="radio-group">
        <label><input type="radio" name="is_hal" value="false" {hal_false}> No — real legal concept</label>
        <label><input type="radio" name="is_hal" value="true" {hal_true}> Yes — hallucination pattern</label>
      </div>
      <button type="submit">Save &amp; Next →</button>
      <button type="button" class="skip" onclick="window.location='/?id={ids[(ids.index(current_id)+1) % len(ids)]}'">Skip</button>
    </form>
  </div>
</div>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        current_id = params.get('id', [None])[0]
        features = load_features()
        if current_id not in features:
            current_id = None
        html = render_page(features, current_id)
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = self.rfile.read(length).decode('utf-8')
        params = parse_qs(body)
        fid = params['id'][0]
        label = urllib.parse.unquote_plus(params['label'][0])
        is_hal = params['is_hal'][0] == 'true'
        next_id = params.get('next_id', [fid])[0]

        features = load_features()
        features[fid]['label'] = label
        features[fid]['is_hallucination_feature'] = is_hal
        save_features(features)

        self.send_response(302)
        self.send_header('Location', f'/?id={next_id}')
        self.end_headers()

if __name__ == '__main__':
    print("Labeling tool running at http://localhost:8080")
    print("Open this in your browser (SSH tunnel must be active)")
    HTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
