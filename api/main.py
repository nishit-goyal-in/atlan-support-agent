"""Vercel deployment entry point for Atlan Support Agent v2."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.app.main import app

# Export the app for Vercel
handler = app