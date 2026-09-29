"""
app.py — Root WSGI Application Entrypoint for Vercel Python Runtime.
Exports the Flask application instance from api.index.
"""

from api.index import app

# Top-level WSGI export expected by Vercel
__all__ = ["app"]

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
