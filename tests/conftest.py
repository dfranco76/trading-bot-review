# tests/conftest.py
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))