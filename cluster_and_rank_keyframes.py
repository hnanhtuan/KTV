import hydra
from omegaconf import DictConfig

from ktv.core.tracking import track_run, default_shell_artifact_paths
from ktv.methods.clustering import run_clustering, load_clip_model, resolve_path

@hydra.main(
    config_path="configs/keyframe_ranking", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    output_dir = resolve_path(cfg.save_cluster_path)
    with track_run(
        cfg,
        stage="keyframe_ranking",
        script_path=__file__,
        output_dir=output_dir,
        extra_tags={
            "dataset": cfg.dataset,
            "query_aware_ranking": getattr(cfg, "enable_query_aware_ranking", True),
        },
    ) as tracker:
        tracker.log_params_from_config(cfg)
        tracker.log_resolved_config(cfg)
        if getattr(cfg, "enable_query_aware_ranking", True):
            load_clip_model(cfg.device)
        summary = run_clustering(
            json_path=resolve_path(cfg.json_path),
            video_path=resolve_path(cfg.video_path),
            video_frame_tensor_path=resolve_path(cfg.video_frame_tensor_path),
            save_cluster_path=output_dir,
            dataset=cfg.dataset,
            combined_output_path=resolve_path(cfg.combined_output_path),
            num_keyframes=getattr(cfg, "num_keyframes", 12),
            enable_query_aware_ranking=getattr(cfg, "enable_query_aware_ranking", True),
            max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
            clustering_method=getattr(cfg, "clustering_method", "kmeans"),
        )
        tracker.log_metrics(
            {
                "saved_count": summary["saved_count"],
                "skipped_existing_count": summary["skipped_existing_count"],
                "missing_tensor_count": summary["missing_tensor_count"],
                "skipped_empty_frame_count": summary["skipped_empty_frame_count"],
                "duration_seconds": summary["duration_seconds"],
            }
        )
        tracker.log_artifacts(
            [summary["summary_path"], summary.get("combined_output_path")],
            artifact_path="outputs",
        )
        tracker.log_artifacts(default_shell_artifact_paths(), artifact_path="logs")


if __name__ == "__main__":
    main()
