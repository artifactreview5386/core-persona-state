# Data Format

This artifact expects formal JSONL files for non-smoke training.
All paths are provided by command-line arguments.

## SFT Rows

`train/sft.py` requires every row to include tensorized inputs, labels, and
masks for all five warm-up terms. Rows are stored without a batch dimension.
Tensor widths must match the selected config.

| Field | Shape | Meaning |
|---|---:|---|
| `context_repr` | `[context_dim]` | Turn representation for `u_t` and local context |
| `history_repr` | `[context_dim]` | Dialogue-history representation |
| `slot_repr` | `[slots, slot_dim]` | Slot-schema representations |
| `value_repr` | `[slots, values, value_dim]` | Candidate value representations, excluding null |
| `previous_belief_summary` | `[slots, belief_dim]` | Summary of `b_{t-1}` for evidence scoring |
| `previous_belief` | `[slots, values]` | Prior slot belief distribution |
| `value_mask` | `[slots, values]` | Valid non-null values |
| `gold_belief_value` | `[slots]` | Gold posterior value index for belief loss |
| `gold_evidence` | `[slots]` | Gold evidence index, where `values` denotes null |
| `gold_update_action` | `[slots]` | `0=COMMIT`, `1=DEFER`, `2=IGNORE` |
| `gold_discourse` | `[]` | `0=RESPOND`, `1=CLARIFY` |
| `response_repr` | `[tokens, context_dim]` | Decoder-side response representations |
| `response_labels` | `[tokens]` | Response token labels |
| `belief_label_mask` | `[slots]` | Mask for belief loss |
| `evidence_label_mask` | `[slots]` | Mask for evidence loss |
| `update_label_mask` | `[slots]` | Mask for update-action loss |
| `discourse_label_mask` | `[]` | Mask for discourse loss |
| `response_label_mask` | `[tokens]` | Mask for response loss |

If a supervision item is not applicable, keep the field present and set the
corresponding mask to zero.

## Uncertainty Calibration Rows

`train/calibrate_uncertainty.py` expects tensorized uncertainty examples:

```json
{
  "features": [0.8, 0.2, 0.1, 0.7],
  "targets": [0.8, 0.2, 0.1]
}
```

The three `targets` values are relevance, ambiguity, and conflict. The loader
also accepts the same values as named fields: `relevance`, `ambiguity`, and
`conflict`.

The resulting checkpoint records `frozen_for_ppo: true` and an
`uncertainty_head_fingerprint`.

## PPO Rollout Rows

`train/ppo.py` expects saved rollout rows with complete reward and objective
fields:

```json
{
  "action": 1.0,
  "state": 1.0,
  "answer": 0.8,
  "clarification": 0.0,
  "kl": 0.05,
  "old_logprob": -1.2,
  "new_logprob": -1.1,
  "advantage": 0.7,
  "value": 0.2,
  "return": 0.9,
  "entropy": 0.1,
  "uncertainty_head_fingerprint": "..."
}
```

The fingerprint must match the frozen uncertainty-head checkpoint passed via
`--uncertainty_head`.
