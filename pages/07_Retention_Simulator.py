"""Retention Scenario Simulator"""
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.metrics import calculate_retention_scenario
from src.navigation import render_sidebar_navigation
from src.ui import configure_dashboard_ui, render_page_header

# Page config
st.set_page_config(
    page_title="Retention Simulator",
    page_icon="💡",
    layout="wide"
)
configure_dashboard_ui()
render_sidebar_navigation(project_root)

render_page_header(
    "Retention Planning",
    "💡",
    "Retention Scenario Simulator",
    "Model campaign reach, retention lift, and financial impact before launching an intervention.",
)

# Load data for context
@st.cache_data
def load_data():
    cleaned_path = project_root / "data" / "processed" / "telco_customer_churn_clean.csv"
    if not cleaned_path.exists():
        st.error("Dataset not found. Please run data ingestion first.")
        return None
    return pd.read_csv(cleaned_path)

df = load_data()

if df is None:
    st.stop()

# Scenario parameters
st.header("Scenario Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Target Segment")

    contract_options = ["All Customers"] + sorted(df["Contract"].unique().tolist())
    target_segment = st.selectbox("Target Segment", contract_options)

    if target_segment == "All Customers":
        eligible_customers = len(df)
    else:
        eligible_customers = len(df[df["Contract"] == target_segment])

    st.metric("Eligible Customers", eligible_customers)

    # Allow manual override
    manual_override = st.checkbox("Manually set eligible customer count")
    if manual_override:
        eligible_customers = st.number_input(
            "Eligible Customers",
            min_value=1,
            max_value=len(df),
            value=eligible_customers
        )

with col2:
    st.subheader("Campaign Parameters")

    campaign_reach = st.slider(
        "Campaign Reach (%)",
        min_value=10,
        max_value=100,
        value=80,
        help="Percentage of eligible customers that will be reached by the campaign"
    ) / 100

    incremental_retention_rate = st.slider(
        "Incremental Retention Rate (%)",
        min_value=1,
        max_value=50,
        value=15,
        help="Estimated percentage improvement in retention due to the intervention"
    ) / 100

# Financial parameters
st.header("Financial Parameters")

col3, col4 = st.columns(2)

with col3:
    contact_cost_per_customer = st.number_input(
        "Contact Cost per Customer ($)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        help="Cost to contact each customer (email, call, offer, etc.)"
    )

with col4:
    contribution_margin_per_retained = st.number_input(
        "Contribution Margin per Retained Customer ($)",
        min_value=0.0,
        max_value=5000.0,
        value=500.0,
        help="Estimated contribution margin from retaining a customer for a year"
    )

# Calculate scenario
st.header("Scenario Results")

scenario = calculate_retention_scenario(
    eligible_customers=eligible_customers,
    campaign_reach=campaign_reach,
    incremental_retention_rate=incremental_retention_rate,
    contact_cost_per_customer=contact_cost_per_customer,
    contribution_margin_per_retained=contribution_margin_per_retained
)

# Display results
col5, col6, col7 = st.columns(3)

with col5:
    st.metric("Customers Targeted", scenario["customers_targeted"])
    st.metric("Customers Reached", scenario["customers_reached"])

with col6:
    st.metric("Incremental Retained", scenario["incremental_retained"])
    st.metric("Campaign Cost", f"${scenario['campaign_cost']:,.2f}")

with col7:
    st.metric("Estimated Value", f"${scenario['estimated_value']:,.2f}")
    metric_value = scenario["net_value"]
    delta_value = scenario["roi"]
    st.metric("Net Scenario Value", f"${metric_value:,.2f}", delta=f"{delta_value:.1f}% ROI")

# Break-even analysis
st.header("Break-Even Analysis")

st.metric(
    "Break-Even Retention Rate",
    f"{scenario['break_even_rate'] * 100:.2f}%",
    help="Minimum incremental retention rate needed to break even"
)

if incremental_retention_rate > scenario['break_even_rate']:
    st.success(f"✅ Scenario is profitable. Current rate ({incremental_retention_rate * 100:.1f}%) exceeds break-even ({scenario['break_even_rate'] * 100:.1f}%)")
else:
    st.warning(f"⚠️ Scenario is not profitable. Current rate ({incremental_retention_rate * 100:.1f}%) is below break-even ({scenario['break_even_rate'] * 100:.1f}%)")

# Sensitivity analysis
st.header("Sensitivity Analysis")

st.subheader("Scenario Sensitivity to Incremental Retention Rate")

retention_rates = [i / 100 for i in range(5, 51, 5)]
net_values = []

for rate in retention_rates:
    sensitivity_scenario = calculate_retention_scenario(
        eligible_customers=eligible_customers,
        campaign_reach=campaign_reach,
        incremental_retention_rate=rate,
        contact_cost_per_customer=contact_cost_per_customer,
        contribution_margin_per_retained=contribution_margin_per_retained
    )
    net_values.append(sensitivity_scenario["net_value"])

fig_sensitivity = go.Figure()
fig_sensitivity.add_trace(go.Scatter(
    x=[r * 100 for r in retention_rates],
    y=net_values,
    mode="lines+markers",
    name="Net Value",
    line=dict(color="#00D4FF")
))

fig_sensitivity.add_hline(
    y=0,
    line_dash="dash",
    line_color="white",
    annotation_text="Break-Even"
)

fig_sensitivity.update_layout(
    title="Net Scenario Value vs Incremental Retention Rate",
    xaxis_title="Incremental Retention Rate (%)",
    yaxis_title="Net Scenario Value ($)",
    template="plotly_dark",
    hovermode="x unified"
)

st.plotly_chart(fig_sensitivity, width='stretch')

# Scenario comparison
st.header("Scenario Comparison")

st.subheader("Compare Multiple Scenarios")

scenario_names = ["Conservative", "Base Case", "Optimistic"]
scenario_params = [
    {"reach": 0.6, "retention": 0.10},
    {"reach": 0.8, "retention": 0.15},
    {"reach": 0.9, "retention": 0.25}
]

comparison_data = []

for name, params in zip(scenario_names, scenario_params):
    comp_scenario = calculate_retention_scenario(
        eligible_customers=eligible_customers,
        campaign_reach=params["reach"],
        incremental_retention_rate=params["retention"],
        contact_cost_per_customer=contact_cost_per_customer,
        contribution_margin_per_retained=contribution_margin_per_retained
    )
    comparison_data.append({
        "Scenario": name,
        "Reach": f"{params['reach'] * 100:.0f}%",
        "Retention Rate": f"{params['retention'] * 100:.0f}%",
        "Customers Reached": comp_scenario["customers_reached"],
        "Incremental Retained": comp_scenario["incremental_retained"],
        "Campaign Cost": f"${comp_scenario['campaign_cost']:,.0f}",
        "Estimated Value": f"${comp_scenario['estimated_value']:,.0f}",
        "Net Value": f"${comp_scenario['net_value']:,.0f}",
        "ROI": f"{comp_scenario['roi']:.1f}%"
    })

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, width='stretch')

# Disclaimer
st.header("Disclaimer")
st.warning("""
**Important Notes:**
- These are hypothetical estimates, not actual savings or proven business outcomes
- The simulator does not automatically equate predicted churn probability with retention offer success
- Actual results may vary significantly based on execution, market conditions, and customer behavior
- The contribution margin estimate should be based on your business's actual economics
- Use these estimates for planning and scenario analysis, not as guaranteed predictions
- Always validate assumptions with historical data and pilot programs when possible

**Formula Used:**
- Customers Reached = Eligible Customers × Campaign Reach
- Incremental Retained = Customers Reached × Incremental Retention Rate
- Campaign Cost = Customers Reached × Contact Cost per Customer
- Estimated Value = Incremental Retained × Contribution Margin per Retained
- Net Value = Estimated Value - Campaign Cost
- ROI = (Net Value / Campaign Cost) × 100
""")
