# Temporal-Chain First-Frame Strategies

The current temporal-chain selector bootstraps the chain with:

```text
argmax(normalized_cluster_representativeness + lambda_event * normalized_eventness)
```

That seed is important because later frames are selected relative to the
already selected set through temporal gap and redundancy terms. A strong but
unlucky first frame can therefore bias the whole chain toward one scene.

## Added Strategies

- `cluster_event`: the current baseline behavior.
- `cluster_only`: starts from the most representative frame.
- `event_only`: starts from the strongest temporal transition.
- `start`: starts from the chronological first sampled frame as a sanity check.
- `middle`: starts from the temporal midpoint.
- `early_event`: starts from the strongest event in the first third of the video.
- `balanced`: uses the baseline score plus a small temporal-center prior.
- `lookahead`: builds a diverse seed pool, completes one chain per seed, and keeps
  the chain with the best global quality, coverage, and redundancy score.

## Recommended Improvement

Use `lookahead` as the primary candidate improvement. It directly addresses the
first-frame sensitivity problem by turning the seed choice into a small
multi-start search instead of a one-shot local decision. The seed pool includes:

- top baseline cluster/event frames,
- temporal-bin winners,
- top representative frames,
- top event frames,
- simple anchors at the start, middle, and end.

Each candidate seed is expanded using the same greedy temporal-chain rule. The
completed chains are scored by average representativeness, average eventness,
timeline coverage, and mean visual redundancy. This keeps the comparison close
to the original objective while making the first frame less brittle.

## Suggested Comparison

For an isolated seed comparison, keep query-aware CLIP ranking off and compare
end-to-end QA accuracy with the same number of selected frames:

```bash
bash scripts/run_temporal_chain_first_frame_strategy_sweep.sh \
  --datasets "nextqa videomme intentqa egoschema" \
  --num-keyframes 6 \
  --num-frames 6 \
  --enable-query-aware-ranking 0
```

For the full candidate-pool plus CLIP-ranking pipeline, use:

```bash
bash scripts/run_temporal_chain_first_frame_strategy_sweep.sh \
  --num-keyframes 12 \
  --num-frames 12 \
  --enable-query-aware-ranking 1
```

The sweep writes a CSV summary to:

```text
outputs/temporal_chain_first_frame_strategy_sweep_summary.csv
```
