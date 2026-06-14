import hydra
from omegaconf import DictConfig

from ktv.core.tracking import track_run, default_shell_artifact_paths
from ktv.methods.clustering import resolve_path, load_clip_model
from ktv.methods.temporal_chain import run_temporal_chain, DEFAULT_SCORE_NORMALIZER

def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

@hydra.main(
    config_path="configs/keyframe_ranking", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    enable_query_aware_ranking = _as_bool(
        getattr(cfg, "enable_query_aware_ranking", True)
    )
    output_dir = resolve_path(cfg.save_cluster_path)
    with track_run(
        cfg,
        stage="temporal_chain_keyframe_ranking_first_frame",
        script_path=__file__,
        output_dir=output_dir,
        extra_tags={"dataset": cfg.dataset},
    ) as tracker:
        tracker.log_params_from_config(cfg)
        tracker.log_resolved_config(cfg)
        if enable_query_aware_ranking:
            load_clip_model(cfg.device)

        summary = run_temporal_chain(
            json_path=resolve_path(cfg.json_path),
            video_path=resolve_path(cfg.video_path),
            video_frame_tensor_path=resolve_path(cfg.video_frame_tensor_path),
            save_cluster_path=output_dir,
            dataset=cfg.dataset,
            combined_output_path=resolve_path(cfg.combined_output_path),
            num_keyframes=getattr(cfg, "num_keyframes", 12),
            enable_query_aware_ranking=enable_query_aware_ranking,
            lambda_event=getattr(cfg, "lambda_event", 0.5),
            alpha_gap=getattr(cfg, "alpha_gap", 0.6),
            beta_redundancy=getattr(cfg, "beta_redundancy", 0.8),
            max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
            first_frame_strategy=getattr(cfg, "first_frame_strategy", "cluster_event"),
            seed_pool_size=getattr(cfg, "seed_pool_size", 16),
            seed_bins=getattr(cfg, "seed_bins", 6),
            score_normalizer=getattr(cfg, "score_normalizer", DEFAULT_SCORE_NORMALIZER),
            num_workers=1,
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
