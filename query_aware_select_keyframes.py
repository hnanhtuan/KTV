import hydra
from omegaconf import DictConfig

from ktv.core.tracking import track_run, default_shell_artifact_paths
import ktv.methods.clustering as base
from ktv.methods.query_aware import run_query_aware_selection, derive_output_paths, resolve_path

@hydra.main(
    config_path="configs/query_aware_keyframe_ranking",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    save_cluster_path, combined_output_path = derive_output_paths(cfg)
    video_frame_tensor_path = (
        resolve_path(cfg.video_frame_tensor_path)
        if cfg.video_frame_tensor_path is not None
        else None
    )
    with track_run(
        cfg,
        stage="query_aware_keyframe_ranking",
        script_path=__file__,
        output_dir=save_cluster_path,
        extra_tags={
            "selection_mode": cfg.selection_mode,
            "query_mode": cfg.query_mode,
        },
    ) as tracker:
        tracker.log_params_from_config(cfg)
        tracker.log_resolved_config(cfg)
        base.load_clip_model(cfg.device)
        summary = run_query_aware_selection(
            json_path=resolve_path(cfg.json_path),
            video_path=resolve_path(cfg.video_path),
            video_frame_tensor_path=video_frame_tensor_path,
            save_cluster_path=save_cluster_path,
            combined_output_path=combined_output_path,
            selection_mode=cfg.selection_mode,
            query_mode=cfg.query_mode,
            num_keyframes=cfg.num_keyframes,
            dense_candidate_pool_size=cfg.dense_candidate_pool_size,
            output_top_k=cfg.output_top_k,
            max_frames_to_extract=cfg.max_frames_to_extract,
            skip_existing=cfg.skip_existing,
            sample_limit=cfg.sample_limit,
        )
        tracker.log_metrics(
            {
                "saved_count": summary["saved_count"],
                "skipped_existing_count": summary["skipped_existing_count"],
                "missing_tensor_count": summary["missing_tensor_count"],
                "non_full_output_count": summary["non_full_output_count"],
                "duration_seconds": summary["duration_seconds"],
            }
        )
        tracker.log_artifacts(
            [summary["summary_path"], summary["combined_output_path"]],
            artifact_path="outputs",
        )
        tracker.log_artifacts(default_shell_artifact_paths(), artifact_path="logs")


if __name__ == "__main__":
    main()
