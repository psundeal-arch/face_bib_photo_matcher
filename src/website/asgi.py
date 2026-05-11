"""ASGI entrypoint for running the Flask app with Uvicorn."""

from pathlib import Path
import sys

from asgiref.wsgi import WsgiToAsgi

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from app import app as flask_app

app = WsgiToAsgi(flask_app)

__all__ = ["app"]
