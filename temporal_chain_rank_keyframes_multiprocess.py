import os
import hydra
from omegaconf import DictConfig

from ktv.core.tracking import track_run, default_shell_artifact_paths
from ktv.methods.clustering import resolve_path
from ktv.methods.temporal_chain import run_temporal_chain, DEFAULT_SCORE_NORMALIZER

@hydra.main(
    config_path="configs/keyframe_ranking", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    output_dir = resolve_path(cfg.save_cluster_path)
    with track_run(
        cfg,
        stage="temporal_chain_keyframe_ranking_multiprocess",
        script_path=__file__,
        output_dir=output_dir,
        extra_tags={"dataset": cfg.dataset},
    ) as tracker:
        tracker.log_params_from_config(cfg)
        tracker.log_resolved_config(cfg)
        
        num_workers = getattr(cfg, "num_workers", None)
        if num_workers is None:
            num_workers = max(1, min(os.cpu_count() or 1, 8))
            
        summary = run_temporal_chain(
            json_path=resolve_path(cfg.json_path),
            video_path=resolve_path(cfg.video_path),
            video_frame_tensor_path=resolve_path(cfg.video_frame_tensor_path),
            save_cluster_path=output_dir,
            dataset=cfg.dataset,
            combined_output_path=resolve_path(cfg.combined_output_path),
            num_keyframes=getattr(cfg, "num_keyframes", 12),
            enable_query_aware_ranking=getattr(cfg, "enable_query_aware_ranking", True),
            lambda_event=getattr(cfg, "lambda_event", 0.5),
            alpha_gap=getattr(cfg, "alpha_gap", 0.6),
            beta_redundancy=getattr(cfg, "beta_redundancy", 0.8),
            score_normalizer=getattr(cfg, "score_normalizer", DEFAULT_SCORE_NORMALIZER),
            max_frames_to_extract=getattr(cfg, "max_frames_to_extract", 5400),
            first_frame_strategy=getattr(cfg, "first_frame_strategy", "cluster_event"),
            seed_pool_size=getattr(cfg, "seed_pool_size", 16),
            seed_bins=getattr(cfg, "seed_bins", 6),
            num_workers=num_workers,
            worker_blas_threads=getattr(cfg, "worker_blas_threads", 1),
            start_method=getattr(cfg, "mp_start_method", "fork"),
            chunksize=getattr(cfg, "mp_chunksize", None),
        )
        tracker.log_metrics(
            {
                "saved_count": summary["saved_count"],
                "skipped_existing_count": summary["skipped_existing_count"],
                "missing_tensor_count": summary["missing_tensor_count"],
                "skipped_empty_frame_count": summary["skipped_empty_frame_count"],
                "num_workers": num_workers,
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
