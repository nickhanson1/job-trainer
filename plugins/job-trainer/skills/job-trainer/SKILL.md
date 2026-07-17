---
name: av-training-scheduler
description: >-
  Submit, monitor, stop, and resume AV (autonomous-driving) model training jobs
  on the shared GPU job-queue server used in the summer training course. The
  server is NOT a REST API — it speaks raw JSON over a TCP/UNIX socket via the
  JobQueueClient class in av-training/utils/job_queue.py. Use this whenever a
  student wants to launch a training run, check how a run is going, read its
  logs/stdout, stop a run, resume a run from its last checkpoint, list their
  jobs, or debug why a job failed. Also covers configuring a run at three tiers:
  automatic (Claude gathers a little context and decides everything), basic
  (model, dataset, run name, lr, epochs, batch size), and advanced (model
  architecture / layer sizes, image crop & resize, augmentation, history / action
  horizon, sampling, optimizer & scheduler internals). Throughout, Claude leads
  with a recommended choice tailored to the student's dataset. Triggers on "submit
  a training job", "start training", "just set it up for me", "pick everything for
  me", "you choose the settings", "I don't know what to pick", "is my run done",
  "check my job", "tail/show the logs", "stop my job", "resume training", "my job
  crashed/failed", "list my jobs", "change the crop", "make the network bigger",
  "tune the architecture", "set the history length", "configure all the options",
  "advanced config".
---

# AV Training Job Scheduler

## The situation

Students in the AV summer course collect driving data and train models on a
**shared GPU server** that runs a custom job queue (`av-training/utils/job_queue.py`).
A student tells you what they want; you talk to the scheduler on their behalf —
submitting runs, watching them, reading logs, stopping, and resuming.

This is a **high-trust classroom setup with no authentication**. Instead, every
message carries a `user_id` so the scheduler can tell jobs apart and scope a
student to their own runs. Always use the student's real `user_id` and never
touch another student's jobs.

## The protocol (important — not REST)

The scheduler is a socket server, not an HTTP service. The client is
`JobQueueClient` in `av-training/utils/job_queue.py`:

```python
from job_queue import JobQueueClient
client = JobQueueClient("TCP:130.245.191.128:8091")   # or "UNIX:/path/to/sock"
resp = client.send_msg({"user_id": USER_ID, "command": "job:status", "job_id": jid})
```

`send_msg` opens a connection, sends one JSON object, half-closes, reads one
JSON object back, and closes. Every request **must** include `user_id`. Every
response includes a top-level `"status"` of `"ok"` or `"error"` (with `"msg"`).

### Commands

| Command              | Required fields            | Returns                                              |
| -------------------- | -------------------------- | --------------------------------------------------- |
| `job:new`            | `user_id`, `job_config`    | `{status, job_id}`                                   |
| `job:status`         | `user_id`, `job_id`        | `{status, job_status: {...}}`                        |
| `job:stop`           | `user_id`, `job_id`        | `{status}` — stops if running, cancels if queued     |
| `user:get_all_jobs`  | `user_id`                  | `{status, jobs: [job_id, ...]}`                      |
| `admin:get_status`   | `user_id`                  | `{scheduler: {user_map, servers}}` (instructor only) |

A **`job_status`** object contains:

- `status` — one of `queued`, `running`, `stopping`, `cancelled`, `finished`
  (a `finished` job's `returncode` tells you success `0` vs failure `≠0`).
- `config` — the full training config dict (model, epochs, dataset, paths…).
- `queued_time` / `finish_time` — UNIX timestamps.
- `output` — the job's stdout/stderr, **base64-encoded gzip of raw bytes**. You
  must `gzip.decompress(base64.b64decode(output)).decode()` to read it. The
  helper script does this for you.

## How to use this skill

Prefer the bundled helper `scripts/jobctl.py`. It is **standalone** (Python
standard library only — it reimplements the socket client, so it does not need
the heavy `av-training` dependencies installed) and decodes logs for you.

Configure once via env vars (or pass `--socket` / `--user` per call):

```bash
export SCHEDULER_SOCKET="TCP:130.245.191.128:8091"   # see "Known ports" below — CONFIRM with the user first
export SCHEDULER_USER="kmahon"                  # the student's user_id
export SCHEDULER_UPLOAD_URL="http://130.245.191.128:80"  # data-upload server (dataset listing) — CONFIRM first
```

### Known ports (confirm before using)

These are the addresses seen in the existing course code. **Do not assume they
are correct for this student's setup — show them and ask the user to confirm (or
give you the right one) before you submit, stop, or resume anything.** Ports and
hosts drift between terms and machines.

| Address                | Used for                                  | Source                          |
| ---------------------- | ----------------------------------------- | ------------------------------- |
| `TCP:130.245.191.128:8091`   | Student commands (`job:*`, `user:*`)      | Group1 notebook (most recent)   |
| `TCP:130.245.191.128:8081`   | Admin / multi-server view (`admin:*`)     | Group1 notebook (instructor)    |
| `TCP:0.0.0.0:8080`     | Built-in default if a server is started without `--socket` | `job_queue.py` |
| `http://130.245.191.128:80` | **Dataset/model listing** (`GET /datasets`, `/models`) — the data-upload server, a **separate HTTP service** from the scheduler | `utils/data_upload_server.py` |

The **data-upload server is HTTP, not the socket scheduler.** It owns the shared
datasets directory, so listing a student's datasets goes through it (`GET
/datasets?user_id=<id>`), not through the scheduler socket. It has no `__main__`
in the repo, so its port isn't pinned there; the skill defaults to port `80` on
the same host as the scheduler — **confirm this with the user** before relying on
it, exactly like the scheduler socket.

Default to `TCP:130.245.191.128:8091` for normal student work, but first ask
something like: "I'll use the scheduler at `TCP:130.245.191.128:8091` — is that the
right host/port for you?" If they're on a remote server, the host will not be
`127.0.0.1`.

```bash
# List the student's OWN uploaded datasets (HTTP query to the data-upload server)
python scripts/jobctl.py datasets
python scripts/jobctl.py datasets --local   # instead glob the server FS (on-server only)

# List the student's jobs (id + status + model + epoch progress)
python scripts/jobctl.py list

# Show one job's status (decoded), optionally with its config
python scripts/jobctl.py status <job_id>
python scripts/jobctl.py status <job_id> --config

# Read a job's logs / stdout (last N lines)
python scripts/jobctl.py logs <job_id> --tail 200

# Watch a job until it reaches a terminal state (polls status + log tail)
python scripts/jobctl.py monitor <job_id>

# Submit a new training job from a config file (JSON or YAML)
python scripts/jobctl.py submit --config-file my_run.json
python scripts/jobctl.py submit --config-file my_run.yaml --overlay training.epochs=80

# Stop (running) or cancel (queued) a job
python scripts/jobctl.py stop <job_id>

# Resume a finished/stopped job from its last checkpoint (submits a NEW job)
python scripts/jobctl.py resume <job_id>
```

Every command prints JSON. When reporting back to the student, **summarize** —
give them the job id, status, epoch/ETA, and the relevant log lines; don't dump
raw JSON or the entire stdout unless asked.

When you present job status/progress as a table, lay it out **one row per job**
with these columns, in this order: **ID**, **Name**, **Status**, **Epoch**,
**Time Remaining** (this scales cleanly when several jobs are running at once).
The Name is the run name (`config.training.model_spec`), which the default
`status` call does not return — get it from `status <id> --config` or `list`. Use
a labeled header row — do **not** emit a table with an empty/blank top row
(`| | |`). Show Epoch as `current / total` and Time Remaining as a **bare value**
(e.g. `1:57:40`) — do **not** prefix it with a tilde (`~`) or other approximation
marker.

```
| ID                   | Name         | Status  | Epoch  | Time Remaining |
|----------------------|--------------|---------|--------|----------------|
| azA21C+HLO405i1viLhj | claude_test  | running | 2 / 50 | 1:57:40        |
```

## Building a training config (set the values the student wants)

There are **three tiers** of configuration. Pick based on how much the student
wants to be involved:

- **Tier 0 — Automatic** (next section): the student wants *you* to decide
  everything. You gather a little optional context, make **every** choice
  yourself, and present one config to confirm. Use this when the student says
  things like "just set it up for me", "pick everything", "you choose", "do it
  for me", "I don't know what to pick", or gives no opinions.
- **Tier 1 — Basic**: the handful of choices the notebook's `MainWindows` widget
  exposed — dataset, model, run name, learning rate, epochs, batch size. Use this
  when the student wants a say in the main knobs but not the internals.
- **Tier 2 — Advanced / Comprehensive**: the *full* config surface — model
  architecture (layer sizes), cropping, resize, augmentation, history / action
  horizon, sampling, optimizer/scheduler internals, and more. Use it when the
  student says things like "I want to change the crop", "make the network bigger",
  "tune the architecture", "set the history length", or "give me all the options".

When unsure which tier, **default to Tier 0** and offer to go deeper — most
students just want a good run submitted.

Both tiers compose a config from the repo's presets + the student's choices
exactly the way the training system does (see `docs/CONFIG_SYSTEM.md`), via
`scripts/build_config.py`, which writes a ready-to-submit JSON.

### Always lead with a recommendation (both tiers)

Do **not** walk a student through the config as a list of blank questions. For
**every** choice — in both Tier 1 and Tier 2 — present the value **you** think is
best for *their* dataset as a clearly marked **✅ recommended** option, give a
one-line reason, then let them accept it or override. Students in this course
usually don't know what to pick; a sensible default they can veto beats an empty
prompt. Never silently apply your pick, though — the student still confirms.

Show each decision like this:

> **Model** — ✅ **`fastvit-384`** *(recommended: strong, fast default for
> phone-camera driving data; fits the GPU and trains quickly)*.
> Other options: `fastvit-512` (more capacity, slower), `resnet18` (simplest
> baseline), `fastvit-384-attn` (adds attention if you have lots of data).

When you want the student to pick among a few options, use the **AskUserQuestion**
tool and make the recommended option the **first** choice with "(recommended)" in
its label. Base every recommendation on what you know about the student's dataset
(source device, size, task) via the heuristics below. If you genuinely can't tell
(e.g. dataset size unknown), say what you'd need, then still give a safe default.

#### Recommendation heuristics (dataset → good defaults)

These are starting points, not laws — state them *as* recommendations and adapt
to anything the student tells you. Preset defaults (see `reference/config_catalog.md`)
are the baseline; deviate only for a reason you can name.

| If the dataset is… | Recommend | Why |
| ------------------ | --------- | --- |
| **iOS phone data** (the common case) | `--data ios_data` + replace `dataset_paths` with their uploads | correct `type`, crop, resize, and steering factor for the iOS camera FOV |
| **Android phone data** | `--data android_data` (same, replace paths) | different FOV → different crop/steering defaults |
| **Course track / hallway sets** | the matching preset (`track_data`, `large_track_data`, `hallway_data`) | paths + iOS defaults already wired |
| **Model, standard steering** | `fastvit-384` | best accuracy/speed trade-off default; `resnet18` if they want the simplest baseline |
| **Small dataset (≲5k frames)** | keep epochs modest (~50), keep/raise augmentation, watch overfitting | little data overfits fast on a big net |
| **Large dataset (≳20k frames)** | `fastvit-512` or `fastvit-384-attn` are worth it; 60–100 epochs | capacity pays off; don't overtrain |
| **Learning rate** | `1e-4` (Adam default); `5e-4` **with** `--scheduler sched_cosine` for longer/bigger runs | safe default; warmup+cosine tolerates the higher lr |
| **Epochs** | `80` for a normal run | enough to converge without burning the queue |
| **Batch size** | keep effective `1024`; on OOM **lower `mini_batch`** (128→64→32), not `batch_size` | preserves the effective batch while fitting memory |
| **Action-horizon / instructed models** | only if the student's task needs it | most steering runs don't; adds dataset-`type` + arch coupling |

### Tier 0 — Automatic

The student wants a good run submitted without walking a menu. You make **every**
decision using the recommendation heuristics above + the catalog defaults, then
show one summary and get a single confirm. The student's job is just to give you
context and say "go".

**Step 1 — Find their data (don't make them type paths).** Run
`python scripts/jobctl.py datasets` to discover their uploads (an HTTP query to
the data-upload server). If it returns one obvious dataset, use it. If several,
ask which (or whether to combine them). If the upload server isn't reachable, ask
for the dataset name and build the path per "Training on the student's own
collected data".

**Step 2 — Ask for context, not config.** Ask a few **optional** plain-language
questions — make clear they can skip any and you'll pick sensibly. Use these
answers to shape the config; never demand config values here.

| Ask about… | What it changes in your choices |
| ---------- | ------------------------------- |
| **What the model should do** (stay in a lane, follow a track, avoid obstacles, follow commands…) | model family + dataset `type`; e.g. obstacle/command tasks may want a chunk-action or instructed setup, plain lane-keeping wants `fastvit-384` |
| **How much data they recorded** (minutes/laps/number of runs, rough frame count) | epochs, model size, `max_dataset_size`, augmentation — small data → smaller net + fewer epochs + more augment; lots of data → bigger net, more epochs |
| **Phone/device used** (iPhone vs Android) | `ios_data` vs `android_data` preset (crop/resize/steering) |
| **How it's going / prior runs** ("last one wouldn't turn", "overfit", "diverged") | lr, augmentation, epochs, or whether to `resume` instead |
| **Any constraints** (needs to finish tonight, GPU is busy) | epochs, model size, `mini_batch` |

**Step 3 — Decide everything and build.** Apply the heuristics to pick dataset,
model, lr, epochs, batch/`mini_batch`, scheduler, and any data settings the
context implies. **Auto-generate the run name** yourself (you don't need to ask):
`<user_id>-<short-task>-<vN>`, e.g. `kmahon-track-v1`. Build with
`build_config.py` exactly as in Tier 1.

**Step 4 — Present one summary + rationale, then confirm.** Show the resolved key
fields (dataset + frame count, model, lr, epochs, batch, anything notable) with a
**one-line "why" for each non-obvious pick**, and ask a single yes/no: "Submit
this, or want to tweak anything?" This is the one required checkpoint — never
auto-submit without it (see Guardrails). If they want to change something,
you've effectively dropped into Tier 1/2 for that knob.

> Example summary to the student:
> *"Here's what I'd run on your `kmahon^track_v2` data (~8k frames): **FastViT-384**
> (solid default for lane/track following), **80 epochs**, **lr 1e-4**, effective
> batch **1024**, iOS crop/steering defaults. Run name `kmahon-track-v1`. Submit?"*

### Tier 1 — Basic

In the notebook, the first cells choose everything *before* anything runs:
dataset, model, a run name, learning rate, epochs, etc. (the `MainWindows`
widget). The widget only exists in the notebook, so to set these values
yourself, compose them with `build_config.py`.

#### Interview the student first

Go through these choices **with a recommendation attached to each** (see "Always
lead with a recommendation" above) — present your best pick for their dataset,
then let them accept or change it. The only two you must *not* invent are the
**dataset paths** and the **run name**; ask for those outright. For the rest,
propose the recommended default and move on unless the student wants to change it:

| What to ask                | Goes to                          | Notes                                   |
| -------------------------- | -------------------------------- | --------------------------------------- |
| **Dataset** to train on    | `data.dataset_paths` (`--data` / `--data-path`) | A dataset preset, the student's **own uploaded data** (see below), or explicit paths. |
| **Model**                  | `training.model_name` (`--model`) | Preset (with arch args) or a raw name.  |
| **Run name**               | `training.model_spec` (`--name`) | Becomes `run_name = model_name-model_spec-timestamp`. |
| **Learning rate**          | `training.optimizer_args.lr` (`--lr`) | Default `1e-4`.                    |
| **Epochs**                 | `training.epochs` (`--epochs`)   | Default `100`.                          |
| **Batch size**             | `training.batch_size` (`--batch-size`) | Default `1024`; lower if OOM.     |
| anything else              | `--set a.b=val,c=val2`           | e.g. `data.num_workers=8`, `data.max_dataset_size=10000`, `training.mixing=true`. |

#### Option catalog (from the repo — confirm against the live presets)

- **Dataset presets** (`training/preset/*.yaml`): `hallway_data`, `track_data`,
  `large_track_data`, `ios_data`, `android_data`, `carla_data`, `waymo-e2e`,
  `lmdrive`, `all_dataset` (everything). Multiple `--data` presets **combine**
  their paths (append-list). Or give explicit `--data-path /path/...`.
- **The student's own collected data** (the common case for the summer course):
  see "Training on the student's own data" just below.
- **Model presets**: `fastvit-384`, `fastvit-512`, `fastvit-384-attn`,
  `fastvit-512-attn(-l)`, `vit-clip`, `vit-pretrained`. These also set the
  architecture `model_args`.
- **Raw model names** (no preset, no special args needed): `resnet18/34/50/101/152`,
  `efficientnet_b0/b4`, `cnn`, `vit`, `vit_clip`, `resnet18_gru_chunkaction`, … —
  pass any registered name to `--model`.
- **Scheduler**: `constant` (default) or `warmup_cosine` (use `--scheduler sched_cosine`).
- **Optimizer**: any `torch.optim` class via `training.optimizer` (default `Adam`).

#### Build it

```bash
# Compose a config from a model preset + dataset preset + the student's choices
python scripts/build_config.py \
    --model fastvit-384 \
    --data hallway_data \
    --name kmahon-hallway-v1 \
    --lr 1e-4 --epochs 80 \
    --set data.num_workers=8,data.max_dataset_size=10000 \
    -o run.json

# Then submit it (see the scheduler section above)
python scripts/jobctl.py submit --config-file run.json
```

`build_config.py` resolves preset `_include`/`_append` keys, merges dataset paths
with append-list, and warns if `model_name` or `dataset_paths` end up empty. It
also **auto-sets `training.finish_hook`** so the run converts its trained model
into a deployable artifact (see "Model conversion" below) — you don't need to add
this yourself. The
presets are **bundled with this skill** in `scripts/presets/` (a copy of
`av-training/training/preset/`), so it works without an `av-training` checkout —
`--preset-dir` and `--repo-root` both default there. Point them at a live
checkout only if you need fresher presets (`--preset-dir /path/to/av-training/training/preset
--repo-root /path/to/av-training`). It needs **PyYAML** to read presets; if that's
unavailable, hand it pre-merged JSON via `--base`.

> **Keeping `scripts/presets/` in sync:** it's a snapshot, not a live mirror —
> re-copy from `av-training/training/preset/` when presets change upstream. The
> bundled copies rewrite the in-preset `_include` paths from
> `training/preset/<x>.yaml` to bare `<x>.yaml` so they resolve inside the flat
> folder; redo that rewrite (or pass `--repo-root` pointing at a real checkout)
> after re-copying.

After building, **show the student the key fields** (model, dataset count, name,
lr, epochs, batch size) and confirm before submitting — this is the moment to
catch a wrong dataset or a fat-fingered learning rate.

#### Training on the student's own collected data

Most students want to train on the data **they** collected and uploaded, not a
course preset. Uploaded data lands on the server at
`/studentdata/past_summer_camp/summer2026/datasets/`, one folder per dataset
named `<user_id>^<dataset_name>` (the data-upload server's convention; the `^`
keeps one student's data from colliding with another's). Discover a student's
own datasets by asking the **data-upload server** over HTTP (it owns that
directory and lists it via `GET /datasets?user_id=<id>`):

```bash
python scripts/jobctl.py datasets            # uses $SCHEDULER_USER + $SCHEDULER_UPLOAD_URL
python scripts/jobctl.py --user kmahon datasets --json
```

This works **from anywhere** — it's an HTTP call, not a filesystem glob, so you
don't need to be on the GPU server. Confirm the upload-server URL with the user
first (see "Known ports"); override with `--upload-url` or `$SCHEDULER_UPLOAD_URL`.
If the server can't be reached, ask the student for their dataset name(s) and
build the full path yourself as
`/studentdata/past_summer_camp/summer2026/datasets/<user_id>^<name>`.

> If you happen to be running **on the GPU server itself**, `datasets --local`
> globs the filesystem directly (`Path(dir).glob(f"{user_id}^*")`) instead of
> using HTTP — the old behavior, kept as a fallback.

Then build the config from the **`ios_data` preset** (for the iOS dataset `type`
and the right crop/resize/steering defaults) and **replace** `dataset_paths` with
the student's full paths. Use `--set-json` to replace the list (so the preset's
placeholder `"dataset"` path is dropped — don't use `--data-path`, which would
*append* and leave the placeholder in):

```bash
python scripts/build_config.py \
    --model fastvit-384 --data ios_data --name kmahon-mydata-v1 \
    --lr 1e-4 --epochs 80 \
    --set-json 'data.dataset_paths=["/studentdata/past_summer_camp/summer2026/datasets/kmahon^trackrun1","/studentdata/past_summer_camp/summer2026/datasets/kmahon^hallway_loops"]' \
    -o run.json
```

(`jobctl.py datasets` prints a ready-to-paste `--set-json 'data.dataset_paths=[…]'`
line for exactly this. Use `--data android_data` instead for Android-collected
data.)

#### The raw config shape

The result is the dict `job:new` expects (same as `Config.to_dict()`):
`save_dir`, `training.{model_name, model_spec, epochs, batch_size,
optimizer, optimizer_args.lr, scheduler}`, `data.{dataset_paths, type,
num_workers, ...}`. `run_name`/`models_ckpt` are filled in by the server at
launch. You can also point `submit --config-file` at any hand-written JSON/YAML
of this shape, and `submit --overlay key=val,...` still works for last-second tweaks.

#### Model conversion (always keep the finish hook)

A trained run saves a PyTorch `.pth` checkpoint, which is **not** the format the
car/phone app can load. The repo turns it into a deployable **CoreML** artifact
via a **finish hook** — a shell command the trainer runs once after training
completes (`training/__main__.py` → `utils/convert_hook.py`, which converts, names,
and zips the model). The training preset ships with `training.finish_hook: null`,
so **without this set, jobs finish but never produce a usable model.**

Therefore **every submitted config must set `training.finish_hook`.**
`build_config.py` does this for you automatically — it defaults
`training.finish_hook` to `python utils/convert_hook.py` whenever the field is
empty — so you don't need to add anything for a normal run. Just be aware:

- If a student hand-writes a config (or you build one another way), **add
  `training.finish_hook: "python utils/convert_hook.py"` yourself** so the run
  emits a deployable model.
- The default yields to an **explicit** finish hook: if the student sets one via
  `--set`/`--set-json training.finish_hook=...`, that value is kept.
- Pass `--no-convert-hook` only if the student explicitly does **not** want a
  converted artifact (rare). If you do, tell them the run will only leave a
  `.pth` checkpoint.
- The hook runs on the server after training; it needs `utils/convert.py` and its
  conversion deps present there. If a finished job's logs show the convert step
  failing, the training itself still succeeded — the `.pth` is safe and can be
  converted later.

### Tier 2 — Advanced (comprehensive)

When the student wants more than the six basic knobs, switch to the full config
surface. **The authoritative, grouped list of every knob lives in
`reference/config_catalog.md`** — read it and walk the student through it. Each
entry gives the config key, default, valid values/range, and **which models or
datasets it applies to** (many knobs are "if applicable" — e.g. `history_length`
only matters for instructed models, FastViT layer sizes only for `fastvit`).

The catalog is organized into groups; run through them in this order, asking only
about what's relevant to the student's goal. For **each** knob you raise, lead
with a **✅ recommended** value chosen for the student's dataset (per "Always lead
with a recommendation" and the catalog's defaults), give a one-line reason, and
name the trade-off of moving off it — then let the student decide. The advanced
knobs are where a bad guess hurts most, so a grounded recommendation matters more
here, not less. If a knob's default is already right for their case, say so and
move on rather than forcing a choice.

| Group | Covers | Examples of what the student can change |
|-------|--------|------------------------------------------|
| **A. Identity** | run name, seed, W&B | `training.model_spec`, `seed`, `wandb` |
| **B. Model & architecture** | model choice + `model_args` | FastViT `embed_dims`/`layers`/`mlp_ratios`/`token_mixers`, ViT `pretrained`, ResNet `edge_filter`, chunk-action `chunk_size`/`with_command`, ViT-GRU `adapter_dim` |
| **C. Image pipeline** | crop, resize, augmentation | `data.crop_x`/`crop_y`, `data.resize`, `data.augment`, `data.random_shift`, `data.steering_factor`, `data.max_augmentation` |
| **D. Dataset & sampling** | composition, splits, history/horizon | `data.max_dataset_size`, `data.train_test_split`, `data.resample`, `data.repeat`, `data.history_length`, `data.seq_len`, `data.action_length`/`action_stride`, `data.num_workers` |
| **E. Optimization & schedule** | training loop | `training.epochs`, `batch_size`/`mini_batch`, `optimizer`/`optimizer_args`, `scheduler`/`scheduler_args`, `clip_grad`, `mixing` |
| **F. Hooks** | save / eval hooks (rare); **`finish_hook` = model conversion, set by default** | `training.save_hook`, `finish_hook` (defaults to `python utils/convert_hook.py`), `evaluate_hook` |

#### Setting advanced values with build_config.py

Start from a base + model preset + dataset preset (Tier-1 flags), then layer the
advanced knobs on top:

- **Scalars** → `--set a.b=val,c.d=val2` (e.g. `--set training.mini_batch=64,data.max_dataset_size=5000`).
- **Lists / nested / typed** → `--set-json 'KEY=<json>'`, **repeatable, one key
  per flag** (the value is parsed as JSON, so quote bare strings). This is
  required for `crop_x`, `crop_y`, `resize`, `embed_dims`, `layers`,
  `token_mixers`, `augment`, `optimizer_args`, and any `model_args` object.

```bash
# Custom crop + a wider FastViT backbone, cosine schedule, smaller mini-batch
python scripts/build_config.py \
    --model fastvit-384 --data track_data --scheduler sched_cosine \
    --name kmahon-fastvit-wide --epochs 120 --lr 5e-4 \
    --set training.mini_batch=64,data.max_dataset_size=8000 \
    --set-json 'data.crop_x=[120,520]' \
    --set-json 'data.crop_y=[150,340]' \
    --set-json 'training.model_args.embed_dims=[64,128,256,512]' \
    --set-json 'data.augment={"brightness":0.2,"contrast":0.2,"hue":0.05,"saturation":0.1}' \
    -o run.json

# Chunk-action (action horizon) run: model + matching dataset type + horizon
python scripts/build_config.py \
    --model resnet18_gru_chunkaction --data track_data \
    --name kmahon-chunk --epochs 100 \
    --set-json 'data.type="ios_dataset_chunkaction"' \
    --set data.action_length=10 \
    --set-json 'training.model_args={"chunk_size":10,"with_command":false}' \
    -o run.json
```

After building, **show the student the resolved key fields** (model + any arch
changes, dataset/type + count, crop/resize, lr/epochs/batch, anything they
customized) and confirm before submitting — exactly as in Tier 1. The whole point
of Tier 2 is more surface area to fat-finger, so the confirmation step matters more.

## How resume works (there is no `job:resume` command)

Resuming = submitting a **new** job whose config sets:

- `resume: true`
- `load_dir: <the original run's checkpoint dir>` (i.e. its `models_ckpt`,
  which is `save_dir/run_name`).

The server's `initialize()` → `resume()` then loads `ckpt/ckpt_<n>.pth` from that
dir and continues (and resumes the W&B run if configured). `jobctl.py resume
<job_id>` automates this: it fetches the old job's config, sets `resume`/`load_dir`,
and submits a fresh job — which gets a **new** `job_id`. Point any further
monitoring at the new id.

## Debugging a failed job

1. `status <job_id>` — a `finished` status with `returncode != 0` (or
   `reason`/`stopping`) means it failed.
2. `logs <job_id> --tail 200` — read the traceback. Common culprits: OOM (lower
   `training.batch_size`), bad dataset path, missing checkpoint on resume,
   config typo.
3. Explain the root cause and propose a concrete config/code fix.
4. With the student's OK, resubmit a corrected job (don't blindly resubmit the
   exact command that just failed). If progress exists, prefer `resume`.

## Guardrails

- **Only act on the student's own jobs.** `user_id` scopes everything; never
  pass someone else's `user_id` to stop/resume their work. `admin:get_status` is
  instructor-only.
- **Confirm destructive actions.** Stopping a long-running job loses unsaved
  progress — confirm first unless the student clearly already asked.
- **Always confirm before submitting — even in Tier 0.** Automatic mode means you
  *decide* the config, not that you *submit* it unattended. Show the resolved key
  fields and get an explicit "go" before `job:new`; the student presses the
  button. Every submitted run occupies the shared GPU queue.
- **Don't invent a `job_id` or a socket address.** Use the id the server
  returned; if you don't have one, `list` first. If the socket/user is unknown,
  ask rather than guessing the port.
- **Be honest about failures.** If a response is `{"status": "error", ...}` or a
  job's `returncode != 0`, say so and show the message/log — don't claim success.
- **Free GPUs for classmates.** If a run is clearly diverging (loss NaN, wrong
  config), stop it rather than letting it burn the queue.
