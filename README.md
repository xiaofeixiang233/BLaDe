# BLaDe: Bayesian Language Decision

Code and experiment records for **"Bayesian Language Decision: Looking Back and Planning Ahead"**.

[Paper (PDF, download)](docs/blade-paper.pdf?raw=true) · [Project page](https://xiaofeixiang233.github.io/BLaDe/)

BLaDe is a training-free way to pair a fast System I actor with a slow System II planner.
One frozen actor is queried under three contexts and the answers are combined with a single weight `w`:

```
pi_w(a)  ∝  pi_h(a) · ( pi_p(a) / pi_0(a) )^w

pi_0 = P(a | s)        observation only
pi_h = P(a | s, h)     observation + the last four real interactions
pi_p = P(a | s, p)     observation + the planner's current target state
```

The first eight actions of an episode use history only. After that the planner (GPT) writes a
target state at up to four fixed points per episode, and every action is the argmax of the
combined distribution over the full native action menu. The program never checks whether a
target was reached, never filters actions, and never looks ahead.

## Repository layout

```
blade/                 the experiment package (python -m blade)
  policy.py            the three queries and the combination rule
  prompts.py           planner prompt
  environment.py       BabyAI / MiniHack / Baba Is AI text adapters (pinned BALROG sources)
  alfworld_adapter.py  ALFWorld valid-unseen adapter
  providers.py         Jev API, GPT planner, local d3 / Kev clients
  d3_*.py, kev_*.py    local model download, verification and serving
  ledger.py            per-request ledger, budgets, exact response reuse
  runner.py            freeze cases, run episodes, resume
  report.py, efficiency.py, timing.py   metrics
  plans.py             the two frozen experiment plans
  configs/             main.json, ablation.json
  tests/               offline tests (no model calls)
docs/PROTOCOL.md       frozen protocol and what the program does and does not decide
docs/blade-paper.pdf   the paper
docs/index.html        the project page (self-contained)
results/               result tables and per-step episode logs of the paper's runs
```

## Experiments

| Plan | Methods | Benchmarks | Episodes |
|---|---|---|---:|
| `main` | `{jev,d3,kev}-flat-h0` (System I), `-w05`, `-w2` (BLaDe) | BabyAI 125, MiniHack 40, Baba Is AI 200, ALFWorld 134 | 4,491 |
| `ablation` | `{jev,d3}-history-h4`, `-joint-h4` | BabyAI, MiniHack, Baba Is AI | 1,460 |

Actors: `jev-1.13.0` (hosted API), `d3` (`vllm-sr/d3`, 27B, local), Kev (`jaredpalmer/kev-27b`, local).
Planner: `gpt-6-sol`, medium reasoning effort, OpenAI Responses API.

## Setup

Linux x86_64, Python 3.12. Nothing in this section calls a model.

```bash
sudo apt-get install build-essential flex bison libncurses-dev zlib1g-dev libbz2-dev tmux

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r blade/requirements-controller.txt

# Pinned native environments, each in its own venv under blade/.runtime/
python -m blade.setup_envs --groups grid nle     # BabyAI, Baba Is AI, MiniHack
python -m blade.setup_alfworld                   # ALFWorld valid-unseen data and runtime

# Offline tests
python -m unittest discover -s blade/tests
```

Create a configuration and freeze the cases:

```bash
python -m blade init --plan main --config config.local.json    # or --plan ablation
python -m blade doctor  --config config.local.json
python -m blade prepare --config config.local.json
```

`prepare` resets every case once, records ids, seeds, menus, observation hashes and source
hashes, and writes `freeze.json`. It needs no credentials and no model server.

### Credentials and budgets

```bash
python -m blade credentials        # hidden prompt; saved to ~/.config/blade/credentials.json (mode 600)
```

Then set in `config.local.json`:

- `credentials_file`: absolute path of that file (it must be outside the repository), or
  export `OPENAI_API_KEY` and `JEV_KEYS` in the shell instead.
- `budgets.jev_usd`, `budgets.gpt_usd`: spending limits for this run. The templates ship with
  `null`, which allows preparation only.

Never put keys in the configuration, source, or command line; the loader rejects them.

### Local actors

d3 (reference: four replicas, two 48 GB GPUs each, CUDA 12.6):

```bash
python3.12 -m venv .venv-d3
.venv-d3/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.venv-d3/bin/pip install -r blade/requirements-d3.txt
.venv-d3/bin/pip install --no-build-isolation causal-conv1d==1.7.0
.venv-d3/bin/python -m blade.d3_model --model-dir blade/.runtime/d3-model
python -m blade.launch_d3 --python .venv-d3/bin/python --model-dir blade/.runtime/d3-model
curl --fail http://127.0.0.1:8902/health
```

With one GPU pair, pass `--gpu-pairs 0,1 --ports 8902`, keep one d3 endpoint in the
configuration and set `workers` to 1.

Kev (one 80 GB GPU). Check out the upstream Kev source at the revision pinned in
`blade/kev_model.py` and install its requirements, then:

```bash
python -m blade.kev_model prepare --model-dir /data/kev-27b --download     # about 51 GB
python -m blade.kev_model serve   --model-dir /data/kev-27b --source-dir /path/to/kev --port 8906
```

Both servers listen on loopback only. Health checks verify checkpoint hashes, compute settings
and library versions before any episode starts.

## Running

```bash
tmux new -s blade
python -m blade run --config config.local.json --execute                      # every method of the plan
python -m blade run --config config.local.json --execute --methods kev-flat-h0 kev-w2 kev-w05
python -m blade report --config config.local.json
```

`run` sends paid requests; it refuses to start without `--execute`, positive budgets, the
frozen case list and healthy servers for the selected actors. Re-running the same command
resumes: stored actions are replayed and checked against the stored observations, and only
unfinished episodes continue. Exit code 3 means a budget pause.

Outputs (under `outputs/<plan>/`):

| File | Content |
|---|---|
| `freeze.json`, `manifests/` | frozen configuration, source hashes, cases |
| `requests.sqlite3` | every request with raw response, usage, cost, latency |
| `episodes/*.jsonl` | every action, native feedback, the three distributions, combined scores |
| `goals/*.jsonl` | planner targets as returned |
| `report.json`, `results.csv`, `efficiency.json`, `timing.json` | success, steps, paired differences, cost, latency |

## Released results

[`results/`](results/README.md) holds the result tables, frozen case manifests, per-step
episode logs and planner targets behind Tables 1, 2 and 4 and the History / Joint comparison.
Raw request ledgers and machine logs are not included.

## License

MIT, see [LICENSE](LICENSE). Benchmarks, model weights and upstream sources fetched by the
setup scripts keep their own licenses.
