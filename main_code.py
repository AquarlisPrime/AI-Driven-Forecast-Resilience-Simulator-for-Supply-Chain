import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import shap
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import io
from datetime import timedelta

import warnings

warnings.filterwarnings("ignore")


from pyvis.network import Network
import streamlit.components.v1 as components

def draw_pyvis_graph(graph):
    net = Network(height='400px', width='100%', directed=True)
    net.from_nx(graph)

    for node in net.nodes:
        node['title'] = node['id']
        node['size'] = 25

    for edge in net.edges:
        edge['title'] = (
            f"{edge['from']} ➜ {edge['to']}<br>"
            f"Transit: {graph.edges[edge['from'], edge['to']]['transit_days']} days<br>"
            f"Cost: ${graph.edges[edge['from'], edge['to']]['cost']}"
        )
        edge['color'] = 'gray'

    net.repulsion(node_distance=120, spring_length=200)
    net.show_buttons(filter_=['physics'])

    net.save_graph('graph.html')
    with open('graph.html', 'r', encoding='utf-8') as f:
        html = f.read()
    components.html(html, height=450, scrolling=True)



# --- Helper Functions ---
def train_prophet(df, forecast_days, start_date=None):
    model = Prophet()

    # Optional filtering based on forecast start date
    if start_date is not None:
        df = df[df['ds'] <= start_date]
        if df.empty:
            raise ValueError("No data available before selected forecast start date.")

    model.fit(df)

    # Ensure future starts right after last date in training data
    last_train_date = df['ds'].max()
    if start_date is not None and start_date > last_train_date:
        future_start = start_date
    else:
        future_start = last_train_date + pd.Timedelta(days=1)

    future = pd.date_range(start=future_start, periods=forecast_days)
    future_df = pd.DataFrame({'ds': future})

    forecast = model.predict(future_df)
    return forecast, model


def train_xgboost(df, forecast_days, start_date=None):
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds').resample('D').sum().reset_index()

    df['dayofyear'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    df['rolling_mean_7'] = df['y'].shift(1).rolling(window=7).mean()
    df.dropna(inplace=True)

    features = ['dayofyear', 'year', 'month', 'dayofweek', 'lag_1', 'lag_7', 'rolling_mean_7']
    X, y = df[features], df['y']

    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBRegressor()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

    final_model = xgb.XGBRegressor()
    final_model.fit(X, y)

    if start_date is not None:
        last_known = df[df['ds'] <= start_date].iloc[-1].copy()
    else:
        last_known = df.iloc[-1].copy()

    forecast_start_date = last_known['ds'] + pd.Timedelta(days=1)
    future_data = []
    for _ in range(forecast_days):
        new_day = last_known['ds'] + pd.Timedelta(days=1)
        row = {
            'ds': new_day,
            'dayofyear': new_day.dayofyear,
            'year': new_day.year,
            'month': new_day.month,
            'dayofweek': new_day.dayofweek,
            'lag_1': last_known['y'],
            'lag_7': last_known['lag_7'],
            'rolling_mean_7': last_known['rolling_mean_7']
        }
        row_df = pd.DataFrame([row])
        row_df['yhat'] = final_model.predict(row_df[features])
        last_known = row_df.iloc[0].copy()
        last_known['y'] = row_df['yhat'].values[0]
        future_data.append(row)

    future_df = pd.DataFrame(future_data)
    future_df['yhat'] = final_model.predict(future_df[features])

    explainer = shap.Explainer(final_model, X)
    shap_values = explainer(X)
    return future_df[['ds', 'yhat']], final_model, X, shap_values

def train_lstm(df, forecast_days, start_date=None):
    df_lstm = df.copy()
    df_lstm = df_lstm.set_index('ds').resample('D').sum().fillna(0)
    values = df_lstm['y'].values.reshape(-1, 1)

    window = 14
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(32, activation='tanh', input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3)])

    pred_input = values[-window:].reshape(1, window, 1)
    preds, dates = [], []
    last_date = df_lstm.index[-1] if start_date is None else start_date - pd.Timedelta(days=1)

    for _ in range(forecast_days):
        yhat = model.predict(pred_input, verbose=0)
        preds.append(yhat[0, 0])
        new_input = np.concatenate([pred_input[:, 1:, :], yhat.reshape(1, 1, 1)], axis=1)
        pred_input = new_input
        last_date += pd.Timedelta(days=1)
        dates.append(last_date)

    return pd.DataFrame({'ds': dates, 'yhat': preds}), model

def evaluate_forecast(true_y, pred_y):
    mape = np.mean(np.abs((true_y - pred_y) / true_y)) * 100
    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    mae = mean_absolute_error(true_y, pred_y)
    return mape, rmse, mae


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📈 AI-Driven Forecast & Resilience Simulator")

st.sidebar.header("📁 Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = pd.read_csv(r"C:\\Users\\Dell\\Downloads\\sales_data.csv")

required_cols = {'product_id', 'date', 'sales'}
if not required_cols.issubset(df_raw.columns):
    st.error("❌ CSV must contain columns: product_id, date, sales")
    st.stop()


product_id = st.sidebar.selectbox("🔍 Select Product ID", df_raw['product_id'].unique())
df = df_raw[df_raw['product_id'] == product_id][['date', 'sales']].copy()
df.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

min_date = df['ds'].min().date()
max_date = df['ds'].max().date()

# --- Select Forecast Start Date ---
forecast_start = st.date_input("📅 Forecast Start Date", value=max_date + timedelta(days=1), min_value=min_date)

# --- Select Forecast End Date ---
forecast_end = st.date_input("📅 Forecast End Date", value=forecast_start + timedelta(days=90), min_value=forecast_start)

# Convert to Timestamp for timedelta calculation
forecast_start_ts = pd.to_datetime(forecast_start)
forecast_end_ts = pd.to_datetime(forecast_end)

forecast_days = (forecast_end_ts - forecast_start_ts).days
st.caption(f"📅 Forecasting {forecast_days} days from {forecast_start_ts.date()} to {forecast_end_ts.date()}")

model_choice = st.sidebar.selectbox("🧠 Choose Model", ["Auto", "Prophet", "XGBoost", "LSTM"])
explain_toggle = st.sidebar.checkbox("🧠 Show Explainability (SHAP)", value=True)

if len(df) < 30:
    st.warning("⚠️ Not enough data (min 30 points).")
    st.stop()

with st.spinner("⚙️ Training models..."):
    forecast_prophet, prophet_model = train_prophet(df.copy(), forecast_days, start_date=forecast_start_ts)
    forecast_xgb, xgb_model, xgb_X, xgb_shap = train_xgboost(df.copy(), forecast_days, start_date=forecast_start_ts)
    forecast_lstm, lstm_model = train_lstm(df.copy(), forecast_days, start_date=forecast_start_ts)

true_y = df['y'][-30:].values
mape_p, rmse_p, mae_p = evaluate_forecast(true_y, forecast_prophet['yhat'][-30:].values)
mape_x, rmse_x, mae_x = evaluate_forecast(true_y, forecast_xgb['yhat'][:30].values)
mape_l, rmse_l, mae_l = evaluate_forecast(true_y, forecast_lstm['yhat'][:30].values)

import seaborn as sns  # Add this import at top if not already there

# Create model comparison DataFrame
metrics_df = pd.DataFrame({
    'Model': ['Prophet', 'XGBoost', 'LSTM'],
    'MAPE': [mape_p, mape_x, mape_l],
    'RMSE': [rmse_p, rmse_x, rmse_l],
    'MAE': [mae_p, mae_x, mae_l]
})

st.markdown("### 📊 Model Comparison")

# Format numeric columns and highlight best model
float_cols = metrics_df.select_dtypes(include='number').columns

# Highlight the best (lowest) values per row for MAPE, RMSE, MAE
def highlight_best(s):
    is_min = s == s.min()
    return ['background-color: lightgreen' if v else '' for v in is_min]

# Apply formatting + conditional styling
styled_df = (
    metrics_df.style
    .format({col: "{:.2f}" for col in float_cols})
    .apply(highlight_best, subset=float_cols, axis=0)  # highlight best per column
    .background_gradient(cmap="OrRd", subset=float_cols)  # color-code errors
)

st.markdown("### 📊 Model Comparison")
st.dataframe(styled_df)



if model_choice == "Auto":
    model_choice = metrics_df.sort_values("MAPE").iloc[0]['Model']
    st.success(f"✅ Auto-picked best model: **{model_choice}**")

if model_choice == "Prophet":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat'], name='Forecast', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat_upper'], name='Upper', fill=None, line=dict(color='lightblue')))
    fig.add_trace(go.Scatter(x=forecast_prophet['ds'], y=forecast_prophet['yhat_lower'], name='Lower', fill='tonexty', line=dict(color='lightblue')))
    st.plotly_chart(fig)
    
    st.subheader("📥 Download Prophet Forecast")
    download_df = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    download_df.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
    st.download_button(
        label="📁 Download Forecast CSV",
        data=download_df.to_csv(index=False).encode('utf-8'),
        file_name="prophet_forecast.csv",
        mime="text/csv"
    )


elif model_choice == "XGBoost":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_xgb['ds'], y=forecast_xgb['yhat'], name='XGB Forecast', line=dict(color='green')))
    st.plotly_chart(fig)
    if explain_toggle:
        st.markdown("#### 📌 SHAP Feature Impact")
        fig_buf = io.BytesIO()
        shap.plots.beeswarm(xgb_shap, show=False)
        plt.tight_layout()
        plt.savefig(fig_buf, format='png', bbox_inches='tight')
        st.image(fig_buf)

elif model_choice == "LSTM":
    fig = go.Figure(go.Scatter(x=forecast_lstm['ds'], y=forecast_lstm['yhat'], name='LSTM Forecast', line=dict(color='orange')))
    st.plotly_chart(fig)

# --- Phase 2 Placeholder ---
st.markdown("---")
st.header("🚧 Digital Twin Simulation")

st.subheader("🛠️ Customize Supply Chain Graph")
import json
import networkx as nx
import seaborn as sns

default_structure = {
  "nodes": [
    {"name": "Factory_A", "type": "source"},
    {"name": "Factory_B", "type": "source"},
    {"name": "Port_X", "type": "intermediate"},
    {"name": "Port_Y", "type": "intermediate"},
    {"name": "Warehouse_North", "type": "sink"},
    {"name": "Warehouse_South", "type": "sink"}
  ],
  "edges": [
    {"from": "Factory_A", "to": "Port_X", "transit_days": 4, "cost": 1800},
    {"from": "Factory_B", "to": "Port_Y", "transit_days": 5, "cost": 2200},
    {"from": "Port_X", "to": "Warehouse_North", "transit_days": 3, "cost": 1200},
    {"from": "Port_Y", "to": "Warehouse_South", "transit_days": 4, "cost": 1500},
    {"from": "Port_Y", "to": "Warehouse_North", "transit_days": 6, "cost": 2000},
    {"from": "Factory_A", "to": "Warehouse_South", "transit_days": 10, "cost": 4000}
  ]
}


user_json = st.text_area("📋 Edit JSON (nodes + edges)", json.dumps(default_structure, indent=2), height=250)

try:
    parsed = json.loads(user_json)
    nodes = [(n['name'], {"type": n['type']}) for n in parsed['nodes']]
    edges = [(e['from'], e['to'], {"transit_days": e['transit_days'], "cost": e['cost']}) for e in parsed['edges']]
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    st.subheader("📍 Live Network Graph")
    draw_pyvis_graph(G)

    # 📦 Node Inventory Controls
    st.subheader("📦 Node Inventory Controls")
    inventory_state = {}
    cols = st.columns(len(G.nodes))
    for i, node in enumerate(G.nodes):
        with cols[i]:
            inventory_state[node] = st.slider(
                f"{node} Inventory", 0, 1000, 500, key=f"inv_{node}"
            )

    # 🚚 Transit Delay Adjustments
    st.subheader("🚚 Transit Delay Adjustments")
    delay_state = {}
    for edge in G.edges:
        src, dst = edge
        current = G.edges[edge]['transit_days']
        delay_state[edge] = st.slider(
            f"{src} ➜ {dst} Delay (days)",
            min_value=1,
            max_value=30,
            value=current,
            key=f"delay_{src}_{dst}"
        )
        G.edges[edge]['transit_days'] = delay_state[edge]  # Apply live update

except Exception as e:
    st.error(f"❌ Error parsing JSON: {e}")



def disruption_heatmap(graph):
    matrix = pd.DataFrame(index=graph.nodes, columns=graph.nodes, dtype=float)
    for src in graph.nodes:
        for dst in graph.nodes:
            if src != dst:
                try:
                    path = nx.shortest_path(graph, src, dst, weight='transit_days')
                    total_days = sum(graph[u][v]['transit_days'] for u, v in zip(path[:-1], path[1:]))
                    matrix.loc[src, dst] = total_days
                except nx.NetworkXNoPath:
                    matrix.loc[src, dst] = None
    return matrix

import io

# --- Transit Delay Heatmap ---
st.subheader("🌡️ Transit Delay Heatmap")
heat = disruption_heatmap(G)
fig_heat, ax_heat = plt.subplots(figsize=(9, 7), dpi=100)
sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax_heat, cbar=False,
            annot_kws={"size": 5})
ax_heat.tick_params(axis='both', labelsize=8)
plt.tight_layout(pad=0.1)
buf1 = io.BytesIO()
fig_heat.savefig(buf1, format="png", bbox_inches='tight')
st.image(buf1)

# --- Simulate Cost Impacts ---
st.subheader("💰 Simulate Cost Impacts")
multiplier = st.slider("Disruption Cost Multiplier", 1.0, 5.0, 1.5)
total_costs = {e: G.edges[e]['cost'] * multiplier for e in G.edges}
cost_df = pd.DataFrame.from_dict(total_costs, orient='index', columns=['Simulated Cost'])
cost_df.index = [f"{e[0]} ➜ {e[1]}" for e in cost_df.index]
st.dataframe(cost_df.style.format("{:.2f}"))

# --- Seasonal Demand Chart ---
st.subheader("📊 Past Disruptions / Seasonal Demand")
fig_season, ax_season = plt.subplots(figsize=(9, 7), dpi=100)
df_season = df.copy()
df_season['month'] = df_season['ds'].dt.month
monthly_avg = df_season.groupby('month')['y'].mean()
monthly_avg.plot(kind='bar', ax=ax_season, color='skyblue')
ax_season.set_title("", fontsize=6)
ax_season.set_xlabel("", fontsize=6)
ax_season.set_ylabel("Avg Sales", fontsize=8)
ax_season.tick_params(axis='both', labelsize=8)
ax_season.set_xticklabels(ax_season.get_xticklabels(), rotation=0)
plt.tight_layout(pad=0.1)
buf2 = io.BytesIO()
fig_season.savefig(buf2, format="png", bbox_inches='tight')
st.image(buf2)

# Phase 3: Disruption Simulation & Risk Propagation
import random

st.markdown("---")
st.header("📉 Disruption Simulation & Risk Alerts")

st.subheader("⏱ Delay Propagation to Downstream Nodes")

# Compute delay to all sink nodes from each source
sink_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'sink']
source_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'source']
delay_matrix = pd.DataFrame(index=source_nodes, columns=sink_nodes)

for src in source_nodes:
    for sink in sink_nodes:
        try:
            path = nx.shortest_path(G, source=src, target=sink, weight='transit_days')
            delay = sum(G.edges[u, v]['transit_days'] for u, v in zip(path[:-1], path[1:]))
            delay_matrix.loc[src, sink] = delay
        except nx.NetworkXNoPath:
            delay_matrix.loc[src, sink] = None

st.dataframe(delay_matrix.style.format("{:.0f} days"))

# --- Monte Carlo Simulation ---
st.subheader("🎲 Monte Carlo Simulation")
sim_runs = st.slider("Number of Simulation Runs", 100, 1000, 500, step=100)
disruption_prob = st.slider("Delay Disruption Probability", 0.0, 1.0, 0.3)
disruption_impact = st.slider("Delay Increase on Disruption (days)", 1, 10, 3)

simulated_results = {sink: [] for sink in sink_nodes}

for _ in range(sim_runs):
    G_sim = G.copy()
    for edge in G_sim.edges:
        if random.random() < disruption_prob:
            G_sim.edges[edge]['transit_days'] += disruption_impact

    for src in source_nodes:
        for sink in sink_nodes:
            try:
                path = nx.shortest_path(G_sim, source=src, target=sink, weight='transit_days')
                delay = sum(G_sim.edges[u, v]['transit_days'] for u, v in zip(path[:-1], path[1:]))
                simulated_results[sink].append(delay)
            except:
                simulated_results[sink].append(np.nan)

# --- Plot Risk Distribution ---
for sink, delays in simulated_results.items():
    st.markdown(f"#### 📦 {sink} Delay Distribution")
    fig_sim, ax_sim = plt.subplots(figsize=(3, 2), dpi=120)
    sns.histplot(delays, bins=20, kde=True, ax=ax_sim, color='tomato')
    ax_sim.set_xlabel("Delay (days)", fontsize=7)
    ax_sim.set_ylabel("Frequency", fontsize=7)
    ax_sim.set_title(f"{sink}", fontsize=8)
    ax_sim.tick_params(axis='both', labelsize=6)
    plt.tight_layout(pad=0.1)
    buf_sim = io.BytesIO()
    fig_sim.savefig(buf_sim, format="png", bbox_inches="tight")
    st.image(buf_sim)

# --- Alerts ---
st.subheader("🚨 Risk Alerts")
risk_threshold = st.slider("High Risk Delay Threshold (days)", 10, 40, 20)
risk_alerts = []

for sink, delays in simulated_results.items():
    pct_risk = np.mean(np.array(delays) > risk_threshold) * 100
    if pct_risk > 30:
        risk_alerts.append(f"🔴 High risk at **{sink}** — {pct_risk:.1f}% of scenarios exceed {risk_threshold} days")
    elif pct_risk > 10:
        risk_alerts.append(f"🟠 Medium risk at **{sink}** — {pct_risk:.1f}% scenarios risky")
    else:
        risk_alerts.append(f"🟢 Low risk at **{sink}** — {pct_risk:.1f}% scenarios")

for alert in risk_alerts:
    st.markdown(alert)

# --- Phase 4: Cost, Emissions & Risk Analysis ---
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

st.markdown("---")
st.header("💸 Cost, Emissions & Risk Analysis")

results = {}

# --- Transport Cost Calculator ---
st.subheader("🚛 Transportation Cost Calculator")
distance = st.number_input("Distance (km)", min_value=0.0)
cost_per_km = st.number_input("Cost per km", min_value=0.0)
quantity = st.number_input("Quantity (tons)", min_value=0.0)
transport_cost = distance * cost_per_km * quantity
st.metric("Total Transport Cost", f"${transport_cost:,.2f}")
results['Transport Cost ($)'] = transport_cost

# --- Holding Cost Calculator ---
st.subheader("🏬 Holding Cost Calculator")
inventory_days = st.number_input("Inventory Days", min_value=0)
unit_holding_cost = st.number_input("Per Unit Holding Cost ($)", min_value=0.0)
holding_cost = inventory_days * unit_holding_cost
st.metric("Total Holding Cost", f"${holding_cost:,.2f}")
results['Holding Cost ($)'] = holding_cost

# --- Emissions Calculator ---
st.subheader("🌍 Emissions Calculator")
ton_km = st.number_input("Total Ton-Kilometers", min_value=0.0)
emission_factor = st.number_input("Emission Factor (kg CO₂ / ton-km)", value=0.062)
emissions = ton_km * emission_factor
st.metric("CO₂ Emissions", f"{emissions:,.2f} kg")
emissions_cost = emissions * 0.02  # $ per kg CO2
results['CO₂ Emissions (kg)'] = emissions
results['Emissions Cost ($)'] = emissions_cost

# --- Node Risk Scores ---
st.subheader("⚠️ Node Risk Score")
risk_scores = {}
centralities = nx.betweenness_centrality(G)
for node in G.nodes:
    disruption_prob = st.slider(f"{node} Disruption Probability", 0.0, 1.0, 0.1, key=f"risk_{node}")
    risk_scores[node] = centralities.get(node, 0) * disruption_prob

risk_df = pd.DataFrame.from_dict(risk_scores, orient='index', columns=['Risk Score'])

# --- Pie Chart: Cost Breakdown ---
st.subheader("📊 Cost Breakdown")

cost_data = {
    "Transport": transport_cost,
    "Holding": holding_cost,
    "Emissions": emissions_cost
}

if any(v > 0 for v in cost_data.values()):
    fig_pie = px.pie(
        names=cost_data.keys(),
        values=cost_data.values(),
        title="Cost Distribution",
        hole=0.4
    )
    fig_pie.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("ℹ️ Enter values above to see Cost Breakdown.")

# --- Sankey Diagram ---
st.subheader("🔄 Supply Chain Flow (Sankey)")
sankey_nodes = list(G.nodes)
node_indices = {n: i for i, n in enumerate(sankey_nodes)}
sankey_links = dict(source=[], target=[], value=[], label=[])

for u, v, d in G.edges(data=True):
    sankey_links['source'].append(node_indices[u])
    sankey_links['target'].append(node_indices[v])
    sankey_links['value'].append(d.get('cost', 1))
    sankey_links['label'].append(f"{u} ➜ {v}: ${d.get('cost', 1)}")

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(label=sankey_nodes, pad=15, thickness=20, color="lightblue"),
    link=dict(source=sankey_links['source'], target=sankey_links['target'],
              value=sankey_links['value'], label=sankey_links['label'])
)])
fig_sankey.update_layout(title_text="Supply Chain Cost Flow", font_size=10, height=350)
st.plotly_chart(fig_sankey, use_container_width=True)

# --- Risk Heatmap ---
st.subheader("🔥 Node Risk Heatmap")
fig_risk, ax_risk = plt.subplots(figsize=(4, 2.5))
sns.heatmap(risk_df.T, annot=True, cmap="Reds", fmt=".2f", cbar=False, ax=ax_risk)
ax_risk.set_xlabel("Node", fontsize=8)
ax_risk.set_ylabel("Risk Score", fontsize=8)
ax_risk.tick_params(axis='x', labelrotation=45, labelsize=7)
plt.tight_layout(pad=0.2)
buf_risk = io.BytesIO()
fig_risk.savefig(buf_risk, format="png", bbox_inches='tight')
st.image(buf_risk)

# --- Export Section ---
st.subheader("💾 Export Results")

# Combine all results into single DataFrame
export_df = pd.DataFrame(results, index=["Phase 4 Summary"]).T
export_df = export_df.round(2)

csv = export_df.to_csv().encode('utf-8')
st.download_button("⬇️ Download CSV", csv, "phase4_results.csv", "text/csv")

# Generate PDF (simple table)
if st.button("⬇️ Download PDF"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Phase 4 Summary Report", ln=True, align='C')
    pdf.ln(10)
    for key, val in results.items():
        pdf.cell(200, 10, f"{key}: {val:,.2f}", ln=True)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    st.download_button("📄 Download PDF", data=pdf_output.getvalue(),
                       file_name="phase4_summary.pdf", mime="application/pdf")

# --- Phase 5: Advanced Analytics Dashboard ---
import plotly.express as px
import seaborn as sns

st.markdown("---")
st.header("📊 Advanced Analytics Dashboard")

# --- Sidebar Section Switcher ---
section = st.sidebar.radio("📊 Dashboard Sections", [
    "🧠 Forecast",
    "🛰 Simulation",
    "💥 Disruption Engine",
    "📈 Optimization",
    "🔍 Analytics"
])

# ------------------ 🧠 Forecast ------------------
if section == "🧠 Forecast":
    st.subheader("📈 Forecast Visualization (Zoomable + Interactive)")

    if model_choice == "Prophet":
        fig.update_layout(
            title="Prophet Forecast",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            margin=dict(t=30, b=20, l=0, r=0),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    elif model_choice == "XGBoost":
        fig.update_layout(
            title="XGBoost Forecast",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    elif model_choice == "LSTM":
        fig.update_layout(
            title="LSTM Forecast",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------ 🛰 Simulation ------------------
elif section == "🛰 Simulation":
    st.subheader("🛰 Live Digital Twin Simulation")
    st.markdown("✅ Controlled from Phase 2 tabs (Inventory + Delays)")

# ------------------ 💥 Disruption Engine ------------------
elif section == "💥 Disruption Engine":
    st.subheader("💥 Disruption Engine")
    st.markdown("✅ Monte Carlo Simulation & Risk Alerts handled in Phase 3")

# ------------------ 📈 Optimization ------------------
elif section == "📈 Optimization":
    st.subheader("🔧 Optimization Module (Coming Soon)")
    st.info("Here you'll simulate what-if delay/cost tradeoffs and optimize paths.")

# ------------------ 🔍 Analytics ------------------
elif section == "🔍 Analytics":
    st.subheader("📊 Advanced Analytics")

    # --- Animated Time Series ---
    st.markdown("### 📽 Animated Time Series by Month")
    df_ani = df.copy()
    df_ani["month"] = df_ani["ds"].dt.strftime("%Y-%m")
    monthly_anim = df_ani.groupby(["month"])["y"].mean().reset_index()

    fig_anim = px.bar(monthly_anim, x="month", y="y", animation_frame="month",
                      range_y=[0, df['y'].max() * 1.1], title="Monthly Average Demand Over Time")
    st.plotly_chart(fig_anim, use_container_width=True)

    # --- Clustered Impact Reports ---
    st.markdown("### 🧪 Clustered Impact Heatmap (Product vs Region)")
    # Dummy clusterable data for demo
    cluster_df = df_raw.copy()
    if 'region' not in cluster_df.columns:
        cluster_df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(cluster_df))

    grouped = cluster_df.groupby(['product_id', 'region'])['sales'].mean().unstack()
    fig_cluster, ax_cluster = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(grouped, cmap="coolwarm", annot=True, fmt=".0f", ax=ax_cluster)
    ax_cluster.set_title("Average Sales per Product-Region Cluster", fontsize=9)
    ax_cluster.tick_params(axis='x', labelrotation=30)
    st.pyplot(fig_cluster)


# --- Phase 6: What-If Optimization & Auto Grid Search ---
import itertools
from operator import itemgetter

st.markdown("---")
st.header("🧪 Phase 6: What-If Optimization")

st.subheader("📥 Scenario Inputs (Live-Linked to Phase 2 Sliders)")

# Link demand spike to Forecast tab via session_state
if "demand_spike_pct" not in st.session_state:
    st.session_state.demand_spike_pct = 20

selected_ports = st.multiselect("Select Ports to Delay", options=list(G.nodes), default=list(G.nodes)[:2])
delay_days = st.slider("Delay to Apply (days)", 0, 10, 3)
demand_spike_pct = st.slider("Demand Spike (%)", 0, 100, st.session_state["demand_spike_pct"])
st.session_state.demand_spike_pct = demand_spike_pct

run_mode = st.radio("Run Mode", ["🔁 Single Run", "🧮 Batch Simulation with Optimization"])

def simulate_what_if(port_delays, demand_spike_pct):
    G_sim = G.copy()
    for port in port_delays:
        for neighbor in G_sim.successors(port):
            G_sim.edges[port, neighbor]['transit_days'] += port_delays[port]

    stockout_risk = 0
    avg_delay = 0
    total_cost = 0

    for src in source_nodes:
        for sink in sink_nodes:
            try:
                path = nx.shortest_path(G_sim, source=src, target=sink, weight='transit_days')
                delay = sum(G_sim.edges[u, v]['transit_days'] for u, v in zip(path[:-1], path[1:]))
                cost = sum(G_sim.edges[u, v]['cost'] for u, v in zip(path[:-1], path[1:]))
                avg_delay += delay
                total_cost += cost
                if delay > 10:
                    stockout_risk += 1
            except:
                stockout_risk += 1

    n_paths = len(source_nodes) * len(sink_nodes)
    avg_delay = avg_delay / n_paths if n_paths else 0
    stockout_rate = (stockout_risk / n_paths) * (1 + demand_spike_pct / 100)
    return avg_delay, stockout_rate, total_cost

if run_mode == "🔁 Single Run":
    port_delay_dict = {p: delay_days for p in selected_ports}
    avg_delay, stockouts, cost = simulate_what_if(port_delay_dict, demand_spike_pct)

    st.metric("⏱ Average Delay", f"{avg_delay:.1f} days")
    st.metric("📦 Stockout Risk", f"{stockouts:.1%}")
    st.metric("💸 Cost Impact", f"${cost:,.2f}")

else:
    st.subheader("🧮 Grid Search & Optimization")
    ports_to_test = st.multiselect("Ports to Grid Search", options=list(G.nodes), default=list(G.nodes)[:2])
    delay_range = st.slider("Max Delay Range (per port)", 1, 7, 3)

    grid = list(itertools.product(range(delay_range + 1), repeat=len(ports_to_test)))
    results = []

    for combo in grid:
        port_delay_dict = dict(zip(ports_to_test, combo))
        avg_delay, stockouts, cost = simulate_what_if(port_delay_dict, demand_spike_pct)
        results.append({
            "Port Delays": str(port_delay_dict),
            "Avg Delay": avg_delay,
            "Stockout %": stockouts * 100,
            "Cost Impact": cost,
            **port_delay_dict
        })

    df_opt = pd.DataFrame(results)

    # Auto-optimization
    best_delay = df_opt.loc[df_opt["Avg Delay"].idxmin()]
    best_cost = df_opt.loc[df_opt["Cost Impact"].idxmin()]

    st.success(f"🎯 Lowest Delay Config: {best_delay['Port Delays']} — ⏱ {best_delay['Avg Delay']:.2f} days")
    st.success(f"💰 Lowest Cost Config: {best_cost['Port Delays']} — 💸 ${best_cost['Cost Impact']:,.2f}")

    st.dataframe(df_opt[["Port Delays", "Avg Delay", "Stockout %", "Cost Impact"]])

    # Export
    st.download_button(
        "📁 Export Optimization Results CSV",
        data=df_opt.to_csv(index=False).encode(),
        file_name="optimization_grid_results.csv",
        mime="text/csv"
    )

    # 3D Plot
    st.subheader("📈 Optimization Landscape (3D Plot)")
    if len(ports_to_test) >= 2:
        import plotly.express as px
        port_x = ports_to_test[0]
        port_y = ports_to_test[1]
        fig3d = px.scatter_3d(df_opt,
                              x=port_x, y=port_y, z="Avg Delay",
                              color="Cost Impact",
                              size="Stockout %",
                              hover_data=["Port Delays"],
                              title="Avg Delay vs Port Delays (colored by Cost)")
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.info("Add at least 2 ports to view 3D surface.")

