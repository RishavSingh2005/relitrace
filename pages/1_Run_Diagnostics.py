"""Shortcut to the primary v0.4 dashboard upload workflow."""
import streamlit as st
st.set_page_config(page_title="ReliTrace | Run diagnostics", page_icon="🔬")
st.title("Run data-shift diagnostics")
st.info("CSV comparison and automatic threshold calibration now live on the **main ReliTrace page**. Choose **Analyze CSV datasets** there.")
st.page_link("app.py", label="Open the main ReliTrace dashboard", icon="🔎")
