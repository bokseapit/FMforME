"""
FMforME Fusion 360 Add-in
=========================
Receives validated ModelSpec JSON and drives Fusion 360 parametric modeling.

Installation in Fusion 360:
  1. Copy this entire fusion_addin/ folder to:
     %APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\FMforME\
  2. In Fusion 360: Tools → Add-Ins → FMforME → Run

This file is a thin wrapper that imports and runs the main add-in module.
"""

import sys
import os

# Ensure the addin directory is on sys.path
ADDIN_DIR = os.path.dirname(os.path.abspath(__file__))
if ADDIN_DIR not in sys.path:
    sys.path.insert(0, ADDIN_DIR)

# Import and initialize the main add-in
from layer3_fusion_addin import run

def run(context):
    """Fusion 360 add-in entry point."""
    run(context)

def stop(context):
    """Fusion 360 add-in stop callback."""
    pass
