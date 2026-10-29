"""Shared visual styling and page headers for the Streamlit app."""
import html

import streamlit as st


def configure_dashboard_ui() -> None:
    st.markdown(
        """
<style>
:root {
    --app-bg: #080d1b;
    --surface: #111a2d;
    --surface-raised: #17233a;
    --border: rgba(148, 163, 184, 0.15);
    --text: #e8eef8;
    --muted: #91a0b7;
    --cyan: #37d7e8;
    --blue: #638bff;
}

.stApp {
    background:
        radial-gradient(ellipse at 75% -15%, rgba(41, 106, 170, 0.2), transparent 38%),
        var(--app-bg);
    color: var(--text);
    font-family: 'Avenir Next', 'Segoe UI', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(8, 13, 27, 0.72);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, [data-testid="stMetricLabel"] {
    font-family: 'Avenir Next', 'Segoe UI', sans-serif;
    color: var(--text);
}

h2 {
    margin-top: 1.65rem;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--border);
    font-size: 1.35rem;
    letter-spacing: -0.025em;
}

h3 {
    font-size: 1.05rem;
    letter-spacing: -0.015em;
}

p, label, [data-testid="stMarkdownContainer"] {
    color: #c4cede;
}

.page-hero {
    position: relative;
    overflow: hidden;
    margin: 0.25rem 0 1.75rem;
    padding: clamp(1.5rem, 3vw, 2.5rem);
    border: 1px solid rgba(112, 165, 255, 0.21);
    border-radius: 22px;
    background:
        radial-gradient(ellipse at 90% 15%, rgba(55, 215, 232, 0.15), transparent 34%),
        linear-gradient(120deg, rgba(20, 34, 59, 0.98), rgba(13, 24, 43, 0.92));
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.2);
}

.page-hero__topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.page-hero::after {
    position: absolute;
    right: -48px;
    bottom: -96px;
    width: 230px;
    height: 230px;
    border: 1px solid rgba(55, 215, 232, 0.12);
    border-radius: 50%;
    content: "";
    box-shadow: 0 0 0 26px rgba(55, 215, 232, 0.025), 0 0 0 54px rgba(55, 215, 232, 0.02);
    pointer-events: none;
}

.page-hero__eyebrow {
    display: block;
    color: var(--cyan);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.page-hero__status {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.38rem 0.65rem;
    border: 1px solid rgba(55, 215, 232, 0.2);
    border-radius: 999px;
    background: rgba(55, 215, 232, 0.07);
    color: #b7edf1;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    white-space: nowrap;
}

.page-hero__status-dot {
    width: 0.42rem;
    height: 0.42rem;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 12px rgba(55, 215, 232, 0.8);
}

.page-hero__title-row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.page-hero__icon {
    display: grid;
    width: 3.2rem;
    height: 3.2rem;
    flex: 0 0 3.2rem;
    place-items: center;
    border: 1px solid rgba(112, 165, 255, 0.22);
    border-radius: 15px;
    background: linear-gradient(145deg, rgba(55, 215, 232, 0.13), rgba(99, 139, 255, 0.13));
    font-size: clamp(1.8rem, 4vw, 2.5rem);
    line-height: 1;
}

.page-hero h1 {
    margin: 0;
    font-size: clamp(1.8rem, 3.5vw, 2.65rem);
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.14;
}

.page-hero__description {
    margin: 0.8rem 0 0;
    color: #aab8cc;
    font-size: 1rem;
}

[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    min-height: 126px;
    padding: 1.1rem 1.2rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(22, 35, 57, 0.96), rgba(15, 25, 43, 0.96));
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    transition: transform 160ms ease, border-color 160ms ease;
}

[data-testid="stMetric"]::before {
    position: absolute;
    top: 0;
    left: 1.2rem;
    width: 3rem;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--blue), transparent);
    content: "";
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(55, 215, 232, 0.36);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 600;
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
}

[data-testid="stMetricLabel"] p {
    white-space: normal;
    overflow-wrap: anywhere;
}

[data-testid="stMetricValue"] {
    color: #f3f7ff;
    font-family: 'Avenir Next', 'Segoe UI', sans-serif;
    font-size: clamp(1.15rem, 1.65vw, 1.65rem);
    font-weight: 700;
    white-space: nowrap;
    overflow: visible;
    text-overflow: clip;
}

[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stTable"] {
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: rgba(17, 26, 45, 0.78);
}

[data-testid="stDataFrame"] {
    padding: 0.35rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101a2d 0%, #0c1425 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-size: 1.25rem;
    line-height: 1.35;
    letter-spacing: -0.035em;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
    min-height: 2.7rem;
    margin: 0.16rem 0;
    padding: 0.55rem 0.75rem;
    border: 1px solid transparent;
    border-radius: 10px;
    color: #c2cde0;
    transition: background 150ms ease, border-color 150ms ease, color 150ms ease;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
    border-color: rgba(55, 215, 232, 0.2);
    background: rgba(55, 215, 232, 0.08);
    color: #f3f8ff;
}

[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"] {
    border-color: rgba(55, 215, 232, 0.25);
    background: linear-gradient(100deg, rgba(55, 215, 232, 0.13), rgba(99, 139, 255, 0.09));
    color: #f3f8ff;
    font-weight: 700;
}

.stButton > button, .stDownloadButton > button {
    min-height: 2.65rem;
    border: 1px solid rgba(99, 139, 255, 0.38);
    border-radius: 10px;
    background: linear-gradient(110deg, #395fca, #3151a5);
    color: #fff;
    font-weight: 700;
    transition: transform 150ms ease, filter 150ms ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(55, 215, 232, 0.55);
    filter: brightness(1.12);
    color: #fff;
}

[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="textarea"] > div, [data-testid="stNumberInput"] input {
    border-color: var(--border);
    border-radius: 10px;
    background-color: #111a2d;
}

[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1.25rem 1rem 2rem;
    }

    .page-hero {
        border-radius: 16px;
    }

    .page-hero__icon {
        width: 2.7rem;
        height: 2.7rem;
        flex-basis: 2.7rem;
        border-radius: 12px;
    }

    .page-hero__status {
        padding: 0.32rem 0.48rem;
        font-size: 0.58rem;
    }

    [data-testid="stMetric"] {
        min-height: 108px;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
<section class="page-hero">
    <div class="page-hero__topline">
        <span class="page-hero__eyebrow">{html.escape(eyebrow)}</span>
        <span class="page-hero__status"><span class="page-hero__status-dot"></span>INTERACTIVE WORKSPACE</span>
    </div>
    <div class="page-hero__title-row">
        <span class="page-hero__icon" aria-hidden="true">{html.escape(icon)}</span>
        <h1>{html.escape(title)}</h1>
    </div>
    <p class="page-hero__description">{html.escape(description)}</p>
</section>
""",
        unsafe_allow_html=True,
    )
