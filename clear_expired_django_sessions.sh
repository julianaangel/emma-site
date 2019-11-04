#!/bin/bash

# This script should be run periodically to clear out expired
# sessions that Django stores

# Activate the virtual environment with Django
source "$DJANGO_MANAGEMENT_PATH"/venv/bin/activate

# Call the management command that clears expired sessions
python "$DJANGO_MANAGEMENT_PATH"/manage.py clearsessions
