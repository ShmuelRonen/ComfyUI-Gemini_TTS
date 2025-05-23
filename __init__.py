# __init__.py
"""
ComfyUI Gemini Text-to-Speech Node Package

This package provides a ComfyUI node for generating speech using Google's Gemini 2.5 
Flash and Pro models with native TTS capabilities.

Features:
- Native audio generation using Gemini 2.5 Flash/Pro Preview
- Multi-speaker conversation support
- Customizable voice instructions and styling
- Graceful fallbacks when native audio isn't available
- Compatible with ComfyUI audio workflow

Requirements:
- google-generativeai
- torch
- torchaudio
- PIL
- numpy

Author: Based on existing Gemini Flash node architecture
Version: 1.0.0
"""

import os
import sys

# Add the current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the main node class
try:
    from .gemini_tts_node import GeminiTTS
    
    # Define the node mappings for ComfyUI
    NODE_CLASS_MAPPINGS = {
        "GeminiTTS": GeminiTTS,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "GeminiTTS": "🎙️ Gemini Text-to-Speech",
    }
    
    # Optional: Add version info
    __version__ = "1.0.0"
    
    # Export what should be available when importing this package
    __all__ = [
        "GeminiTTS",
        "NODE_CLASS_MAPPINGS", 
        "NODE_DISPLAY_NAME_MAPPINGS",
        "__version__"
    ]
    
    print(f"✅ Gemini TTS Node v{__version__} loaded successfully")
    
except ImportError as e:
    print(f"❌ Error loading Gemini TTS Node: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install google-generativeai torch torchaudio pillow numpy")
    
    # Provide empty mappings to prevent ComfyUI from crashing
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
except Exception as e:
    print(f"❌ Unexpected error loading Gemini TTS Node: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Optional: Add some helper functions or constants
SUPPORTED_MODELS = [
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro-preview"
]

SAMPLE_RATES = [
    "16000",
    "22050", 
    "24000",
    "44100",
    "48000"
]

# Optional: Package metadata
PACKAGE_INFO = {
    "name": "ComfyUI-Gemini-TTS",
    "description": "Text-to-Speech node using Google Gemini 2.5 models",
    "author": "Your Name",
    "version": __version__ if '__version__' in locals() else "unknown",
    "homepage": "https://github.com/yourusername/ComfyUI-Gemini-TTS",
    "requirements": [
        "google-generativeai>=0.8.0",
        "torch>=1.9.0",
        "torchaudio>=0.9.0", 
        "pillow>=8.0.0",
        "numpy>=1.21.0"
    ]
}