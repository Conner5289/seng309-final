#!/bin/bash

uvicorn ai_api:app --host 0.0.0.0 --port 5000 --reload
