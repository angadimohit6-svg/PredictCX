"""Shared chart functions and themes for the dashboard"""
import logging
from typing import Optional, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Color scheme
COLORS = {
    "primary": "#37D7E8",
    "secondary": "#638BFF",
    "accent": "#A78BFA",
    "success": "#42D6A4",
    "warning": "#F6B85B",
    "danger": "#FF7085",
    "background": "#080D1B",
    "card": "#111A2D",
    "text": "#E8EEF8",
    "text_muted": "#91A0B7",
}


def get_chart_template() -> dict:
    """Get the chart template for consistent styling.

    Returns:
        Plotly template dictionary
    """
    template = {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "colorway": [
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent"],
                COLORS["success"],
                COLORS["warning"],
                COLORS["danger"],
            ],
            "font": {
                "color": COLORS["text"],
                "family": "Avenir Next, Segoe UI, sans-serif",
                "size": 12,
            },
            "title": {
                "font": {"size": 16, "color": COLORS["text"]},
                "x": 0.02,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
            },
            "xaxis": {
                "gridcolor": COLORS["card"],
                "zerolinecolor": COLORS["card"],
                "linecolor": "rgba(145,160,183,0.2)",
                "tickfont": {"color": COLORS["text_muted"]},
                "title_font": {"color": COLORS["text_muted"]},
                "automargin": True,
            },
            "yaxis": {
                "gridcolor": COLORS["card"],
                "zerolinecolor": COLORS["card"],
                "linecolor": "rgba(145,160,183,0.2)",
                "tickfont": {"color": COLORS["text_muted"]},
                "title_font": {"color": COLORS["text_muted"]},
                "automargin": True,
            },
            "legend": {
                "bgcolor": "rgba(17,26,45,0.82)",
                "bordercolor": "rgba(148,163,184,0.14)",
                "borderwidth": 1,
                "font": {"color": COLORS["text_muted"]},
                "orientation": "h",
                "y": -0.22,
                "x": 0,
            },
            "hoverlabel": {
                "bgcolor": "#17233A",
                "bordercolor": COLORS["primary"],
                "font": {"color": COLORS["text"], "size": 12},
                "align": "left",
            },
            "hovermode": "x unified",
            "margin": {"l": 48, "r": 24, "t": 58, "b": 52},
        }
    }
    return template


def create_kpi_card(
    title: str,
    value: float,
    subtitle: Optional[str] = None,
    delta: Optional[float] = None,
    color: str = COLORS["primary"]
) -> go.Figure:
    """Create a KPI card visualization.

    Args:
        title: KPI title
        value: KPI value
        subtitle: Optional subtitle
        delta: Optional change from previous period
        color: Color for the value

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Add value
    fig.add_trace(go.Indicator(
        mode="number+delta" if delta is not None else "number",
        value=value,
        delta={"reference": value - delta} if delta is not None else None,
        title={"text": title},
        number={"font": {"size": 32, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]}
    ))

    fig.update_layout(
        template=get_chart_template(),
        height=150,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    orientation: str = "v"
) -> go.Figure:
    """Create a bar chart.

    Args:
        df: DataFrame with data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        color: Column to color by
        orientation: 'v' or 'h'

    Returns:
        Plotly figure
    """
    if orientation == "h":
        fig = px.bar(df, x=y, y=x, title=title, color=color, orientation="h")
    else:
        fig = px.bar(df, x=x, y=y, title=title, color=color)

    fig.update_layout(
        template=get_chart_template(),
        hovermode="x unified",
        bargap=0.28,
    )
    fig.update_traces(marker_line_width=0, opacity=0.92)

    return fig


def create_histogram(
    df: pd.DataFrame,
    x: str,
    title: str,
    color: Optional[str] = None,
    nbins: int = 30
) -> go.Figure:
    """Create a histogram.

    Args:
        df: DataFrame with data
        x: Column to plot
        title: Chart title
        color: Column to color by
        nbins: Number of bins

    Returns:
        Plotly figure
    """
    fig = px.histogram(df, x=x, title=title, color=color, nbins=nbins)

    fig.update_layout(
        template=get_chart_template(),
        hovermode="x unified"
    )

    return fig


def create_box_plot(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    title: str,
    color: Optional[str] = None
) -> go.Figure:
    """Create a box plot.

    Args:
        df: DataFrame with data
        x: X-axis column (optional)
        y: Y-axis column
        title: Chart title
        color: Column to color by

    Returns:
        Plotly figure
    """
    fig = px.box(df, x=x, y=y, title=title, color=color)

    fig.update_layout(
        template=get_chart_template()
    )

    return fig


def create_pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str
) -> go.Figure:
    """Create a pie chart.

    Args:
        df: DataFrame with data
        names: Column for pie slice names
        values: Column for pie slice values
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = px.pie(
        df,
        names=names,
        values=values,
        title=title,
        hole=0.58,
        color_discrete_sequence=[
            COLORS["primary"],
            COLORS["danger"],
            COLORS["secondary"],
            COLORS["accent"],
        ],
    )
    fig.update_traces(
        textposition="inside",
        textinfo="label+percent",
        insidetextfont={"color": COLORS["text"], "size": 12},
        marker={"line": {"color": COLORS["card"], "width": 3}},
        hovertemplate="%{label}<br>%{value:,} customers · %{percent}<extra></extra>",
    )

    fig.update_layout(
        template=get_chart_template()
    )

    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: Optional[str] = None,
    size: Optional[str] = None
) -> go.Figure:
    """Create a scatter plot.

    Args:
        df: DataFrame with data
        x: X-axis column
        y: Y-axis column
        title: Chart title
        color: Column to color by
        size: Column to size points by

    Returns:
        Plotly figure
    """
    fig = px.scatter(df, x=x, y=y, title=title, color=color, size=size)

    fig.update_layout(
        template=get_chart_template()
    )

    return fig


def create_gauge_chart(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    thresholds: Optional[List[tuple]] = None
) -> go.Figure:
    """Create a gauge chart for probabilities or percentages.

    Args:
        value: Value to display
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        thresholds: List of (value, color) tuples for threshold markers

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [min_val, max_val * 0.33], "color": COLORS["success"]},
                {"range": [max_val * 0.33, max_val * 0.66], "color": COLORS["warning"]},
                {"range": [max_val * 0.66, max_val], "color": COLORS["danger"]}
            ],
            "threshold": {
                "line": {"color": COLORS["text"], "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(
        template=get_chart_template(),
        height=300
    )

    return fig


def create_confusion_matrix_plot(
    cm: list,
    class_names: List[str],
    title: str = "Confusion Matrix"
) -> go.Figure:
    """Create a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2D list)
        class_names: List of class names
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14}
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template=get_chart_template()
    )

    return fig


def create_roc_curve(
    fpr: list,
    tpr: list,
    auc_score: float,
    title: str = "ROC Curve"
) -> go.Figure:
    """Create an ROC curve plot.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        name=f"ROC (AUC = {auc_score:.3f})",
        line=dict(color=COLORS["primary"], width=2)
    ))

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name="Random Classifier",
        line=dict(color=COLORS["text_muted"], width=1, dash="dash")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=get_chart_template(),
        hovermode="x unified"
    )

    return fig


def create_feature_importance_plot(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 15
) -> go.Figure:
    """Create a feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Chart title
        top_n: Number of top features to show

    Returns:
        Plotly figure
    """
    df_sorted = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig = px.bar(
        df_sorted,
        x="importance",
        y="feature",
        orientation="h",
        title=title
    )

    fig.update_layout(
        template=get_chart_template(),
        yaxis={"categoryorder": "total ascending"}
    )

    return fig


if __name__ == "__main__":
    # Test chart functions
    pass
