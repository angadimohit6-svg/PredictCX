"""Business metrics and analytics for customer intelligence"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key performance indicators.

    Args:
        df: Customer DataFrame

    Returns:
        Dictionary of KPIs
    """
    kpis = {}

    # Basic counts
    kpis["total_customers"] = len(df)
    kpis["churned_customers"] = (df["Churn"] == "Yes").sum()
    kpis["retained_customers"] = (df["Churn"] == "No").sum()

    # Churn rate
    kpis["churn_rate"] = (kpis["churned_customers"] / kpis["total_customers"]) * 100 if kpis["total_customers"] > 0 else 0

    # Tenure metrics
    kpis["avg_tenure"] = df["tenure"].mean()
    kpis["median_tenure"] = df["tenure"].median()

    # Monthly charges
    kpis["avg_monthly_charges"] = df["MonthlyCharges"].mean()
    kpis["median_monthly_charges"] = df["MonthlyCharges"].median()

    # Total charges
    if "TotalCharges" in df.columns:
        kpis["avg_total_charges"] = df["TotalCharges"].mean()
        kpis["total_revenue"] = df["TotalCharges"].sum()

    logger.info(f"Calculated {len(kpis)} KPIs")
    return kpis


def calculate_churn_by_segment(df: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    """Calculate churn rate by segment.

    Args:
        df: Customer DataFrame
        segment_column: Column to segment by

    Returns:
        DataFrame with churn by segment
    """
    segment_stats = df.groupby(segment_column).agg({
        "customerID": "count",
        "Churn": lambda x: (x == "Yes").sum()
    }).rename(columns={"customerID": "total_customers", "Churn": "churned_customers"}).copy()

    segment_stats.loc[:, "churn_rate"] = (
        segment_stats["churned_customers"] / segment_stats["total_customers"] * 100
    )

    segment_stats = segment_stats.sort_values("churn_rate", ascending=False)

    return segment_stats


def calculate_tenure_band_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate customer distribution by tenure bands.

    Args:
        df: Customer DataFrame

    Returns:
        DataFrame with tenure band distribution
    """
    # Create tenure bands
    df = df.copy()
    df.loc[:, "TenureBand"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72, np.inf],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72", "72+"]
    )

    tenure_stats = df.groupby("TenureBand", observed=False).agg({
        "customerID": "count",
        "Churn": lambda x: (x == "Yes").sum()
    }).rename(columns={"customerID": "total_customers", "Churn": "churned_customers"}).copy()

    tenure_stats.loc[:, "churn_rate"] = (
        tenure_stats["churned_customers"] / tenure_stats["total_customers"] * 100
    )

    return tenure_stats


def calculate_revenue_at_risk(df: pd.DataFrame, predictions: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Calculate revenue at risk based on churn predictions.

    Args:
        df: Customer DataFrame
        predictions: Optional churn probability predictions

    Returns:
        Dictionary with revenue at risk metrics
    """
    revenue_metrics = {}

    if predictions is not None:
        # High-risk customers (probability > 0.7)
        high_risk_mask = predictions > 0.7
        revenue_metrics["high_risk_customers"] = high_risk_mask.sum()
        revenue_metrics["high_risk_revenue"] = df.loc[high_risk_mask, "TotalCharges"].sum()

        # Medium-risk customers (probability 0.4-0.7)
        medium_risk_mask = (predictions >= 0.4) & (predictions <= 0.7)
        revenue_metrics["medium_risk_customers"] = medium_risk_mask.sum()
        revenue_metrics["medium_risk_revenue"] = df.loc[medium_risk_mask, "TotalCharges"].sum()
    else:
        # Use historical churn
        churned = df["Churn"] == "Yes"
        revenue_metrics["churned_revenue"] = df.loc[churned, "TotalCharges"].sum()

    return revenue_metrics


def calculate_customer_lifetime_value_proxy(df: pd.DataFrame) -> pd.Series:
    """Calculate a simple proxy for customer lifetime value.

    Args:
        df: Customer DataFrame

    Returns:
        Series with CLV proxy values
    """
    # Simple CLV proxy: MonthlyCharges * tenure
    clv_proxy = df["MonthlyCharges"] * df["tenure"]
    return clv_proxy


def generate_insights(df: pd.DataFrame) -> list:
    """Generate business insights from the data.

    Args:
        df: Customer DataFrame

    Returns:
        List of insight strings
    """
    insights = []

    # Churn rate by contract
    contract_churn = calculate_churn_by_segment(df, "Contract")
    highest_churn_contract = contract_churn.index[0]
    highest_churn_rate = contract_churn.iloc[0]["churn_rate"]
    insights.append(
        f"Highest churn is in {highest_churn_contract} contracts at {highest_churn_rate:.1f}%"
    )

    # Churn rate by payment method
    payment_churn = calculate_churn_by_segment(df, "PaymentMethod")
    highest_churn_payment = payment_churn.index[0]
    highest_payment_rate = payment_churn.iloc[0]["churn_rate"]
    insights.append(
        f"Highest churn by payment method: {highest_churn_payment} at {highest_payment_rate:.1f}%"
    )

    # Tenure insight
    avg_tenure_churned = df[df["Churn"] == "Yes"]["tenure"].mean()
    avg_tenure_retained = df[df["Churn"] == "No"]["tenure"].mean()
    insights.append(
        f"Churned customers have average tenure of {avg_tenure_churned:.1f} months vs "
        f"{avg_tenure_retained:.1f} months for retained customers"
    )

    # Monthly charges insight
    avg_charges_churned = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
    avg_charges_retained = df[df["Churn"] == "No"]["MonthlyCharges"].mean()
    insights.append(
        f"Churned customers pay ${avg_charges_churned:.2f}/month on average vs "
        f"${avg_charges_retained:.2f}/month for retained customers"
    )

    return insights


def calculate_retention_scenario(
    eligible_customers: int,
    campaign_reach: float,
    incremental_retention_rate: float,
    contact_cost_per_customer: float,
    contribution_margin_per_retained: float
) -> Dict[str, Any]:
    """Calculate retention scenario outcomes.

    Args:
        eligible_customers: Number of eligible customers
        campaign_reach: Proportion of eligible customers reached (0-1)
        incremental_retention_rate: Incremental retention rate from intervention (0-1)
        contact_cost_per_customer: Cost to contact each customer
        contribution_margin_per_retained: Contribution margin per retained customer

    Returns:
        Dictionary with scenario results
    """
    # Calculations
    customers_targeted = eligible_customers
    customers_reached = eligible_customers * campaign_reach
    incremental_retained = customers_reached * incremental_retention_rate
    campaign_cost = customers_reached * contact_cost_per_customer
    estimated_value = incremental_retained * contribution_margin_per_retained
    net_value = estimated_value - campaign_cost

    # Break-even analysis
    if contact_cost_per_customer > 0:
        break_even_rate = contact_cost_per_customer / contribution_margin_per_retained
    else:
        break_even_rate = 0

    scenario = {
        "customers_targeted": int(customers_targeted),
        "customers_reached": int(customers_reached),
        "incremental_retained": int(incremental_retained),
        "campaign_cost": float(campaign_cost),
        "estimated_value": float(estimated_value),
        "net_value": float(net_value),
        "break_even_rate": float(break_even_rate),
        "roi": (net_value / campaign_cost * 100) if campaign_cost > 0 else 0
    }

    return scenario


if __name__ == "__main__":
    # Test analytics functions
    pass
