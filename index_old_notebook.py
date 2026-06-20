"""
Ahadu Bank AI Platform

Streamlit entrypoint wrapper.
This file loads the actual app from streamlit_app.py.
"""

try:
    import streamlit as st
    import streamlit_app
except Exception as e:
    raise RuntimeError(
        "Failed to load Streamlit app.\n"
        "Ensure 'streamlit_app.py' exists in the same directory and that dependencies are installed.\n"
        f"Original error: {e}"
    )
