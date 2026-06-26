import json
import os

notebook_path = "visualization.ipynb"

if not os.path.exists(notebook_path):
    if os.path.exists("playground/visualization.ipynb"):
        notebook_path = "playground/visualization.ipynb"
    else:
        raise FileNotFoundError("Could not locate visualization.ipynb.")

print(f"Reading notebook from: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "# Video Eventness Score Visualizer" in "".join(cell["source"]):
        source_text = "".join(cell["source"])
        if "representativeness" not in source_text.lower():
            source_text = source_text.replace(
                "visualize eventness scores for videos",
                "visualize eventness and cluster representativeness scores for videos"
            )
            source_text = source_text.replace(
                "3. **Compute Eventness Scores**: Compute L2 distance over specified `delta_t` windows",
                "3. **Compute Eventness & Representativeness Scores**: Compute L2 distance over specified `delta_t` windows and K-Means cluster representativeness"
            )
            source_text = source_text.replace(
                "4. **Interactive Plotting**: Plot raw and smoothed eventness curves using Plotly",
                "4. **Interactive Plotting**: Plot raw/smoothed eventness curves and cluster representativeness using Plotly"
            )
            cell["source"] = [line + "\n" for line in source_text.splitlines()]
            print("Updated Markdown cell.")
        
    elif cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        
        # Cell 1: Configuration
        if "DEFAULT_DELTA_T_VALUES = [" in source_text and "import os" in source_text:
            if "import sys" not in source_text:
                source_text = source_text.replace(
                    "import os",
                    "import os\nimport sys\n\n# Add parent directory to sys.path to allow importing ktv modules\nparent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))\nif parent_dir not in sys.path:\n    sys.path.append(parent_dir)"
                )
                cell["source"] = [line + "\n" for line in source_text.splitlines()]
                print("Updated Configuration cell.")
            
        # Cell 3: Compute Eventness & Representativeness Scores
        elif "def compute_moving_average" in source_text:
            if "def compute_representativeness_scores" not in source_text:
                addition = """

def compute_representativeness_scores(features, num_clusters=5, score_normalizer="minmax"):
    \"\"\"
    Compute cluster representativeness score for each frame.
    Uses KMeans clustering and normalizes the negative distance to cluster centers.
    \"\"\"
    from ktv.methods.clustering import perform_clustering
    from ktv.methods.temporal_chain import normalize_scores
    
    # perform_clustering returns labels, centers, r_cluster
    labels, centers, r_cluster = perform_clustering(features, num_clusters, clustering_method="kmeans")
    
    # normalize the scores using the requested score_normalizer
    if score_normalizer is None or score_normalizer == "raw":
        normalized_scores = r_cluster
    else:
        normalized_scores = normalize_scores(r_cluster, score_normalizer)
        
    return normalized_scores, labels
"""
                source_text = source_text.strip() + addition
                cell["source"] = [line + "\n" for line in source_text.splitlines()]
                print("Updated Compute Scores cell.")
            
        # Cell 4: Plotting & Interactive Charts
        elif "def plot_smoothed_eventness_curves" in source_text:
            if "def plot_representativeness_curve" not in source_text:
                addition = """

def plot_representativeness_curve(timestamps, scores, labels, num_clusters):
    \"\"\"
    Plot interactive cluster representativeness score curve over time.
    Colors the markers by their cluster assignment and shows cluster ID in hover.
    \"\"\"
    fig = go.Figure()
    
    # Custom hover data showing Cluster ID
    customdata = [f"Cluster {l}" for l in labels]
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=scores,
        mode='lines+markers',
        marker=dict(
            color=labels,
            colorscale='Viridis',
            size=8,
            showscale=True,
            colorbar=dict(
                title="Cluster ID",
                tickvals=list(range(num_clusters)),
                ticktext=[f"C{i}" for i in range(num_clusters)]
            )
        ),
        line=dict(color='lightgray', width=1.5),
        customdata=customdata,
        hovertemplate='Time: %{x:.2f}s<br>Representativeness: %{y:.4f}<br>%{customdata}<extra></extra>',
        name='Representativeness'
    ))
    
    fig.update_layout(
        title=f"Cluster Representativeness Over Time ({num_clusters} Clusters)",
        xaxis_title="Video Time (seconds)",
        yaxis_title="Representativeness Score",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
"""
                source_text = source_text.strip() + addition
                cell["source"] = [line + "\n" for line in source_text.splitlines()]
                print("Updated Plotting cell.")
            
        # Cell 6: Usability API
        elif "def visualize_video_eventness" in source_text:
            # Replace signature if not already updated
            old_sig = "def visualize_video_eventness(video_path, feature_path, delta_t_values, moving_average_window=DEFAULT_MOVING_AVERAGE_WINDOW):"
            if old_sig in source_text:
                new_sig = """def visualize_video_eventness(
    video_path,
    feature_path,
    delta_t_values,
    moving_average_window=DEFAULT_MOVING_AVERAGE_WINDOW,
    num_clusters=5,
    score_normalizer="minmax"
):"""
                source_text = source_text.replace(old_sig, new_sig)
            
            # Insert logic and chart rendering inside visualize_video_eventness try block
            if "compute_representativeness_scores" not in source_text:
                target_str = "display_plotly_figure(smoothed_fig)"
                replacement_str = """display_plotly_figure(smoothed_fig)

        # Compute representativeness scores
        print(f"Computing representativeness scores (num_clusters={num_clusters}, normalizer={score_normalizer})...")
        rep_scores, labels = compute_representativeness_scores(
            features, num_clusters=num_clusters, score_normalizer=score_normalizer
        )

        # Display representativeness chart
        print("\\n--- Interactive Cluster Representativeness Chart ---")
        rep_fig = plot_representativeness_curve(timestamps, rep_scores, labels, num_clusters)
        display_plotly_figure(rep_fig)"""
                
                source_text = source_text.replace(target_str, replacement_str)
                cell["source"] = [line + "\n" for line in source_text.splitlines()]
                print("Updated Usability API cell with chart drawing.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
