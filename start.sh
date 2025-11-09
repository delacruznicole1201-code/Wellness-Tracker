#!/usr/bin/env bash
# Start Flask app via gunicorn
gunicorn api.predict:app --bind 0.0.0.0:$PORT
