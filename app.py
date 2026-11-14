"""Run the Mother-Tongue Detector web demo on your own computer.

    python app.py

Your browser opens http://localhost:8000. The trained model is in docs/model.json (the notebook rebuilds it) and runs
inside the browser; this script only serves the files in docs/.
"""
import functools
import http.server
import os
import webbrowser

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
PORT = 8000

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DOCS)
with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as server:
    print(f"Mother-Tongue Detector: http://localhost:{PORT}  (press Ctrl+C to stop)")
    webbrowser.open(f"http://localhost:{PORT}")
    server.serve_forever()
