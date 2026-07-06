"""
Production WSGI entry point.

Gunicorn (or any WSGI server) imports the `app` object from this module
instead of running app.py directly, so Flask's built-in dev server
(app.run(debug=True, ...)) is never used in production.

Usage:
    gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
"""

from app import app  # noqa: F401  (imported for the WSGI server to find)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
