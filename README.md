# Supply Chain Simulation & Forecasting Platform

> **Your digital twin for resilient, data-driven supply chain decision-making**

This platform is a modular, AI-powered application MVP, designed to simulate, forecast, and optimize supply chain operations — all in real time. It empowers supply chain professionals, analysts, and researchers to model complex networks, predict demand, run disruption scenarios, visualize downstream risks, and perform cost-emissions-risk tradeoffs, all through an intuitive and interactive Streamlit interface.

Whether you're managing global logistics or prototyping smart factories, this tool turns your supply chain into a living, learning system.

Streamlit link: https://withouttensorflowmaincodepy-udwyn4fyw3zs2xkazszisq.streamlit.app/

---

## 1️. Tech Stack

Seamlessly integrates forecasting, simulation, optimization, and explainability using modern open-source tools.

- **Frontend**: [Streamlit](https://streamlit.io), [Plotly](https://plotly.com)
- **Forecasting**: [Prophet](https://facebook.github.io/prophet/), [XGBoost](https://xgboost.readthedocs.io), [LSTM (Keras)](https://keras.io)
- **Explainability**: [SHAP](https://github.com/slundberg/shap)
- **Simulation & Graphs**: [PyVis](https://pyvis.readthedocs.io), [NetworkX](https://networkx.org)
- **Optimization**: Grid search, scenario modeling, live sliders
- **Backend**: Python 3.10+, modular architecture

---

## 2️. Platform Capabilities

###  2.1 Phase 1: AI-Powered Demand Forecasting
1. Built-in Prophet, XGBoost, and LSTM models
2. Forecast for user-defined calendar periods
3. Auto-selection using MAPE, RMSE, MAE
4. SHAP visualizations for explainability

###  2.2 Phase 2: Interactive Supply Chain Digital Twin
1. Drag-and-drop PyVis supply chain network
2. JSON editor for node/edge structure
3. Real-time inventory & delay sliders
4. Live recalculation of network flow

###  2.3 Phase 3: Disruption Simulation Engine
1. Monte Carlo simulation for edge disruptions
2. Delay propagation to downstream nodes
3. Alert system for risk thresholds
4. Distribution plots for scenario delays

###  2.4 Phase 4: Cost, Emissions, and Risk Analysis
1. Transport, holding, emissions calculators:
   -  Transport = Distance × Cost/km × Qty  
   -  Holding = Inventory Days × Unit Cost  
   -  CO₂ = Ton-km × Emission Factor
2. Interactive pie charts, Sankey flows, risk heatmaps
3. Carbon pricing visualization
4. Exportable CSV/PDF reports

###  2.5 Phase 5: Advanced Analytics Dashboard
1. Sidebar tab system (Forecast, Simulation, Disruption, Optimization, Analytics)
2. Brushing + zoomable forecast plots
3. Animated time series comparison
4. Product/region clustered impact analysis

###  2.6 Phase 6: What-If Optimization
1. User inputs: "Delay port X by Y days", "Spike demand 20%"
2. Simulation outputs: ⏱ Avg Delay, 📦 Stockouts, 💸 Cost Impact
3. Run modes: Single + Batch (grid search)
4. Auto-optimization + 3D plot visualization

---

## 3️. Getting Started

```bash
git clone https://github.com/your-username/supply-chain-simulator.git
cd supply-chain-simulator
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 4. Requirements

Python 3.10+
Libraries listed in requirements.txt, including:
streamlit
prophet
xgboost
tensorflow
shap
pyvis, networkx, plotly, matplotlib

### 5. Deployment (Streamlit Cloud)

Easily deploy on Streamlit Cloud:
Connect GitHub repo
Set main file as streamlit_app.py
Use Python version ≥ 3.10

### 6. License
This project is licensed under the MIT License.


