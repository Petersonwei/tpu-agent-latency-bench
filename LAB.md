# LAB.md — the TPU lab session (loop L1)

Contract for a coding agent on the TPU VM (`follow @LAB.md`), and the script for the video's
"zero to serving" act. **Do not start until `bash scripts/preflight_local.sh` passed on the laptop.**

## Budget
- Target: **one session of ≤ 6 VM-hours** on v6e-1, flex-start with an 8 h auto-delete cap. Check the current v6e price
  on the Cloud TPU pricing page, then multiply. Set a budget alert first.
- Hard stop: if any gate below fails twice, stop, save logs, and **delete the VM**.

## Steps (with expected outputs and gates)
| # | Command | Expected | Gate | ~Time |
|---|---|---|---|---|
| 0 | (laptop) `bash scripts/tpu_create.sh` (flex-start, us-east5-a, 8 h cap), then `bash scripts/vm_sync.sh push` | queued resource ACTIVE, code on VM | — | minutes–hours |
| 1 | `cd agent-latency-bench && bash scripts/setup_vm.sh` | `vllm platform: tpu`, `results/env_versions.txt` | platform line says TPU | 10 min |
| 2 | `source ~/venv-vllm/bin/activate && bash scripts/smoke_tpu.sh` | server ready, `server_ready_s`, smoke table | **success ≥ 0.8 on `tool`; tool_calls > 0; no `truncated_calls`** | 15–25 min (first compile) |
| 3 | `bash scripts/smoke_tpu.sh` again | `server_ready_s` much lower | warm compile cache works | 5 min |
| 4 | `tmux new -s sweep 'bash scripts/sweep_tpu.sh'` | `results/sweep.tsv` grows 54 rows | a failed MTP config is auto-replaced by `ngram3` (see `results/logs/failed_configs.txt`) | 4–4.5 h |
| 5 | `python analysis/analyze.py --results-dir results` | fig1–fig6, `findings.json` (95% CIs) | prediction errors printed; `mech.*` keys present (vLLM /metrics scraped) | 1 min |
| 6 | (laptop) `bash scripts/vm_sync.sh pull` (and `vm_sync.sh watch` during the sweep) | results on laptop | file exists locally | 2 min |
| 7 | (optional, overnight) `follow @program.md` | `results/results.tsv` ≥ 15 rows | — | 4–8 h |
| 8 | (laptop) `bash scripts/tpu_create.sh delete` | VM gone | `gcloud compute tpus tpu-vm list` empty | 1 min |

## If something breaks (fallbacks, in order)
- **Tool calls not parsed** (tool success ≈ 0, tool_calls = 0): check `results/logs/server.log` for
  parser errors → try `TOOL_PARSER=hermes` → try model `Qwen/Qwen3-4B` (CI-validated) with `hermes`.
- **MTP config fails to start on v6e**: automatic — the sweep logs it and runs `ngram3` instead.
  File a GitHub issue with the log line from `results/logs/failed_configs.txt`.
- **Prefix-cache hit rate ≈ 0 on base** (`mech.base.*.prefix_hit_rate`): Qwen3.5 is hybrid and vLLM's
  caching for it is experimental ("align" mode). Check the server log for the mamba cache mode warning;
  report it as a finding. Do not attribute base-vs-noprefix differences to caching.
- **Out of HBM**: lower `GPU_MEMORY_UTILIZATION` to 0.85 or `MAX_MODEL_LEN` to 16384.
- **Thinking never ends / truncated**: raise `--think-budget` or record "thinking on" as truncated.

## Recording for the video
Steps 1–3 (install → first served request → warm restart) are recorded from the laptop with
`scripts/record.sh NAME '<command on VM>'` (asciinema, terminal text only, idle gaps capped at 3 s),
then `bash scripts/render_casts.sh` → `media/gif/*.gif` and `media/mp4/*.mp4`. Rehearse with
`LOCAL=1 bash scripts/record.sh 00_rehearsal 'MOCK=1 bash scripts/smoke_tpu.sh'`.
`server_ready_s` cold vs warm is the first on-screen number.
