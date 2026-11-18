# Released results

Records of the runs reported in the paper. Success counts below are read from the
`results.csv` files in this directory (wins / frozen denominator).

## Main results (Table 1)

| Benchmark | Actor | System I | BLaDe w = 0.5 | BLaDe w = 2 |
|---|---|---:|---:|---:|
| BabyAI (125) | Jev | 13 | 70 | 81 |
| | d3 | 4 | 55 | 67 |
| | Kev | 8 | 29 | 31 |
| MiniHack (40) | Jev | 3 | 7 | 4 |
| | d3 | 2 | 6 | 7 |
| | Kev | 3 | 7 | 5 |
| Baba Is AI (200) | Jev | 24 | 48 | 68 |
| | d3 | 65 | 82 | 90 |
| | Kev | 57 | 67 | 67 |
| ALFWorld (134) | Jev | 6 | 63 | 51 |
| | d3 | 47 | 88 | 94 |
| | Kev | 33 | 107 | 108 |

## History and Joint controls

| Benchmark | Actor | History | Joint |
|---|---|---:|---:|
| BabyAI (125) | Jev | 58 | 66 |
| | d3 | 55 | 57 |
| MiniHack (40) | Jev | 3 | 6 |
| | d3 | 5 | 8 |
| Baba Is AI (200) | Jev | 35 | 79 |
| | d3 | 76 | 87 |

Failed episodes (provider errors) stay in the denominators and count as losses: 11 of the
main episodes above and 2 of the control episodes. See the `failed` column of each
`results.csv`.

## Directories

| Directory | Run | Covers |
|---|---|---|
| `main_short_jev_d3/` | 2026-10-21 | Jev and d3, System I / w = 0.5 / w = 2, on BabyAI, MiniHack, Baba Is AI |
| `main_alfworld_kev/` | 2026-10-23 | all three actors on ALFWorld; Kev on BabyAI, MiniHack, Baba Is AI |
| `ablation_history_joint/` | 2026-10-25 | Jev and d3, History and Joint, on BabyAI, MiniHack, Baba Is AI |

Files:

| File | Content |
|---|---|
| `results.csv` | one row per benchmark × method: completed, failed, wins, score, steps, cost |
| `report.json[.xz]` | full report with per-episode rows and paired comparisons |
| `efficiency.json.xz` | requests, tokens, cost and latency by task, method and provider |
| `timing*.xz`, `timing.csv` | measured per-decision timing (controls run only) |
| `freeze.json`, `manifests/` | frozen configuration, source hashes and case lists |
| `runtime-identities/` | identity reported by the Kev server |
| `execution-events.jsonl` | run start / stage events |
| `episodes.tar.xz` | per-episode `.json` summary and per-step `.jsonl`: action, native feedback, `pi_0` / `pi_h` / `pi_p`, combined distribution, menu |
| `goals.tar.xz` | planner targets as returned |
| `episode-attempts.tar.xz` | per-attempt wall-clock records (controls run only) |

```bash
mkdir -p /tmp/blade-episodes && tar -xJf results/main_alfworld_kev/episodes.tar.xz -C /tmp/blade-episodes
xz -dk results/main_alfworld_kev/report.json.xz
```

## Notes

- `main_short_jev_d3/` holds only `results.csv` and `report.json` for each actor; per-step
  episode logs of that run are not part of this release. That run also covered Crafter and
  NetHack, which the paper does not report; their rows remain in the files, some incomplete.
- In `main_short_jev_d3/`, planner (GPT) dollar figures were rescaled by 1.8 after the run by
  the operator. Jev and d3 figures are unchanged.
- The two later runs sent planner requests through an OpenAI-compatible gateway. Its address
  is replaced by `REDACTED-OPENAI-COMPATIBLE-GATEWAY` in `freeze.json`, so those two files
  are not byte-identical to the originals. Archives were repacked with neutral file
  ownership; their contents are unchanged.
- Raw request ledgers (`requests.sqlite3`) and machine logs are not released.
- The runs used earlier snapshots of the code with package names that differ from this
  repository, so source hashes inside `freeze.json` do not match the current files.
