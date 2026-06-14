import streamlit as st
import sys
from pathlib import Path

# Adjust path to find ktv core packages
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient
from ktv.core.tracking import repo_tracking_uri

# Page configuration
st.set_page_config(
    page_title="KTV Experiment Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 KTV Experiment Tracking Dashboard")
st.markdown("Interactive analysis of keyframe selection methods and token pruning experiments.")

# Load data helper
@st.cache_data(ttl=10)
def load_mlflow_data():
    tracking_uri = repo_tracking_uri(REPO_ROOT)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name("ktv")
    if experiment is None:
        return []
    
    runs = client.search_runs(experiment.experiment_id, max_results=1000)
    
    data = []
    for run in runs:
        params = run.data.params
        tags = run.data.tags
        metrics = run.data.metrics
        
        method = tags.get("stage") or tags.get("method_family") or params.get("prune_mode") or "unknown"
        if "temporal_chain" in method:
            method = "Temporal Chain"
        elif "keyframe_ranking" in method or "clustering" in method:
            method = "Clustering (CLIP)"
        elif "query_aware" in method:
            method = "Query-Aware"
        elif "inference" in method or "qa_inference" in method:
            prune_mode = params.get("prune_mode")
            if prune_mode == "cls_new_token_sim":
                method = "KTV (Token Pruning)"
            elif prune_mode == "uniform_token":
                method = "Uniform Token Pruning"
            else:
                method = "Baseline/Inference"
                
        dataset = tags.get("dataset") or params.get("dataset") or "unknown"
        tokens = params.get("tokens_num") or params.get("tokens") or "unknown"
        accuracy = metrics.get("overall.accuracy") or metrics.get("accuracy")
        
        if accuracy is None:
            continue
            
        try:
            tokens_val = int(tokens)
        except ValueError:
            tokens_val = None
            
        data.append({
            "Run ID": run.info.run_id,
            "Run Name": tags.get("mlflow.runName") or run.info.run_name,
            "Dataset": dataset,
            "Method": method,
            "Tokens Budget": tokens_val,
            "Accuracy": float(accuracy),
            "Duration (s)": metrics.get("duration_seconds")
        })
    return data

data = load_mlflow_data()

if not data:
    st.warning("No runs found in the MLflow tracking store at `mlruns/` with accuracy metrics logged.")
    st.info("Ensure you have run experiments and that accuracy logs are present in MLflow.")
else:
    # Sidebar
    st.sidebar.header("Filter Experiments")
    datasets = sorted(list(set(item["Dataset"] for item in data)))
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)
    
    methods = sorted(list(set(item["Method"] for item in data)))
    selected_methods = st.sidebar.multiselect("Select Methods", methods, default=methods)
    
    # Filter data
    filtered_data = [
        item for item in data 
        if item["Dataset"] == selected_dataset 
        and item["Method"] in selected_methods
        and item["Tokens Budget"] is not None
    ]
    
    if not filtered_data:
        st.warning("No runs matched the selected filters.")
    else:
        # Layout columns
        col1, col2 = st.columns([2, 1])
        
        # We need pandas for formatting the data
        try:
            import pandas as pd
            has_pandas = True
        except ImportError:
            has_pandas = False
            st.error("Pandas is not installed. Please run: pip install pandas")
            
        if has_pandas:
            with col1:
                st.subheader("Pareto Frontier: Accuracy vs. Tokens Budget")
                
                try:
                    import plotly.express as px
                    
                    df = pd.DataFrame(filtered_data)
                    pivot_df = df.groupby(["Method", "Tokens Budget"], as_index=False)["Accuracy"].max()
                    
                    fig = px.line(
                        pivot_df,
                        x="Tokens Budget",
                        y="Accuracy",
                        color="Method",
                        markers=True,
                        labels={"Accuracy": "Accuracy", "Tokens Budget": "Token Budget"},
                    )
                    fig.update_layout(
                        hovermode="x unified",
                        yaxis_tickformat=".2%",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except ImportError:
                    st.info("Install `plotly` for advanced interactive charts.")
                    df = pd.DataFrame(filtered_data)
                    chart_df = df.pivot_table(index="Tokens Budget", columns="Method", values="Accuracy", aggfunc="max")
                    st.line_chart(chart_df)
                    
            with col2:
                st.subheader("Dataset Leaders")
                df_leader = pd.DataFrame(filtered_data)
                best_runs = df_leader.sort_values(by="Accuracy", ascending=False).drop_duplicates(subset=["Method"])
                
                for index, row in best_runs.iterrows():
                    st.metric(
                        label=f"Best {row['Method']}", 
                        value=f"{row['Accuracy'] * 100:.2f}%",
                        help=f"Tokens: {row['Tokens Budget']} | Name: {row['Run Name']}"
                    )
                    
            # Comparison Table
            st.subheader("All Runs Comparison")
            df_all = pd.DataFrame(filtered_data)[["Run Name", "Method", "Tokens Budget", "Accuracy", "Duration (s)", "Run ID"]]
            df_all["Accuracy"] = df_all["Accuracy"].map(lambda x: f"{x * 100:.2f}%")
            st.dataframe(df_all.sort_values(by="Tokens Budget"), use_container_width=True)
