"""Shared sidebar navigation for the Streamlit application."""
from pathlib import Path

import streamlit as st


PAGES = {
    "📊 Executive Dashboard": "pages/01_Executive_Dashboard.py",
    "🔍 Customer Explorer": "pages/02_Customer_Explorer.py",
    "🎲 Churn Prediction": "pages/03_Churn_Prediction.py",
    "🧪 Model Lab": "pages/04_Model_Lab.py",
    "🔬 Explainable AI": "pages/05_Explainable_AI.py",
    "💾 SQL Analytics": "pages/06_SQL_Analytics.py",
    "💡 Retention Simulator": "pages/07_Retention_Simulator.py",
}


def render_sidebar_navigation(project_root: Path) -> None:
    st.sidebar.markdown(
        """
# 🎯 Customer Intelligence AI
**Understand customers. Predict churn. Improve retention.**
"""
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    for page_name, page_path in PAGES.items():
        if (project_root / page_path).exists():
            st.sidebar.page_link(page_path, label=page_name)
        else:
            st.sidebar.markdown(f"~{page_name}~ (Coming soon)")

    st.sidebar.markdown("---")
