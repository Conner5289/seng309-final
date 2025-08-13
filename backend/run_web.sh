#!/bin/bash

source .venv/bin/activate
uvicorn ai_api:app --host 192.168.10.3 --port 5000 --workers 4
