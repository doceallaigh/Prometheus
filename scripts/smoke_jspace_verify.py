"""CPU smoke for jspace_verify: tiny Qwen2 trunk + randomized corrector."""

import json
import tempfile
from pathlib import Path

import torch

from prometheus.retrofit import HiddenDeltaCorrector, jspace_verify, load_trunk

MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

tmp = Path(tempfile.mkdtemp())
traces = tmp / "traces.jsonl"
rows = [
    {"prompt": "Q: What is 12 + 7? Show your work.\n", "completion": "12 + 7 = 19. The tens digit stays 1. #### 19", "answer": "19"},
    {"prompt": "Q: What is 8 * 6? Show your work.\n", "completion": "8 * 6 = 48. #### 48", "answer": "48"},
]
traces.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

device = torch.device("cpu")
model, _ = load_trunk(MODEL, device, torch.float32)
d_model = model.config.hidden_size
num_layers = model.config.num_hidden_layers
tap = max(1, num_layers // 2)

corrector = HiddenDeltaCorrector(d_model=d_model, d_cfc=16)
torch.manual_seed(0)
torch.nn.init.normal_(corrector.delta_head.weight, std=0.05)  # zero-init reads nothing; randomize for the smoke
ckpt_path = tmp / "corrector.pt"
torch.save(
    {"corrector_state": corrector.state_dict(), "config": {"d_model": d_model, "d_cfc": 16, "tap_layer": tap}},
    ckpt_path,
)

results = jspace_verify(
    model_name=MODEL,
    traces_path=traces,
    output_path=tmp / "report.md",
    corrector_path=ckpt_path,
    device_str="cpu",
    num_traces=2,
    positions_per_trace=4,
    directions=3,
    rank=8,
    max_seq_len=64,
)

recon = results["reconstruction"]
assert recon["final_rel_error_max"] < 1e-4, recon
assert recon["tap_rel_error_max"] < 1e-4, recon
assert recon["logit_argmax_agreement"] > 0.999, recon
assert results["alignment"]["corrector_read_fraction_in_jspace"] is not None
print("SMOKE OK", json.dumps(results, indent=2))

# Also cover the final-layer tap edge case (tap == num_layers).
results_final = jspace_verify(
    model_name=MODEL,
    traces_path=traces,
    output_path=None,
    tap_layer=num_layers,
    device_str="cpu",
    num_traces=1,
    positions_per_trace=3,
    directions=2,
    rank=4,
    max_seq_len=64,
)
assert results_final["reconstruction"]["final_rel_error_max"] < 1e-4
print("FINAL-TAP SMOKE OK")
