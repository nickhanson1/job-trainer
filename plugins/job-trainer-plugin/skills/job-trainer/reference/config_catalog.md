# Advanced Config Catalog (Tier 2)

The authoritative, grouped list of every training-config knob, for use with
`scripts/build_config.py`. Each entry gives the **config key**, **default**,
**valid values/range**, and **which models/datasets it applies to**. Many knobs
are "if applicable" — they only matter for a specific model family or dataset
type; those are marked.

**How to set each type with `build_config.py`:**
- **Scalar / string / bool** → `--set a.b=val,c.d=val2` (type inferred; force with `(str)`/`(int)` etc.).
- **List / nested object / typed** → `--set-json 'KEY=<json>'`, **one key per flag, repeatable** (value parsed as JSON — quote bare strings, e.g. `'data.type="ios_dataset_chunkaction"'`). Required for `crop_x`, `crop_y`, `resize`, `augment`, `embed_dims`, `layers`, `mlp_ratios`, `token_mixers`, `downsamples`, `optimizer_args`, `scheduler_args`, and any `model_args` object.

> Defaults below come from the bundled presets (`scripts/presets/`) and the repo
> source. When in doubt, confirm against the live code — key files:
> `training/models/base.py` (model registry `model_list`, `get_model`),
> `training/utils/config.py`, `training/trainer.py`,
> `training/utils/scheduler.py`, `training/__main__.py`, `training/data/*`.

---

## Group A — Identity

| Key | Default | Valid values | Applies to |
|-----|---------|--------------|------------|
| `training.model_spec` | `"base"` | any string | all — becomes `run_name = model_name-model_spec-timestamp` |
| `seed` | auto-generated if unset | int | all — set for reproducibility |
| `wandb` | `null` | `null` or `{entity: str, project: str}` | all — enables Weights & Biases logging; `sweep_id`/`run_id` auto-filled |
| `save_dir` | `"models_ckpt"` | path str | all — base dir for checkpoints; `models_ckpt = save_dir/run_name` filled at launch |

---

## Group B — Model & architecture (`training.model_name` + `training.model_args`)

`training.model_name` is a **registered** name (registry: `model_list` in
`training/models/base.py`; resolved by `get_model`). `training.model_args` is
passed as kwargs to the model constructor — **only the keys that model accepts
have any effect.**

### FastViT — `model_name: "fastvit"` (or `"fastvit_freeze"`)
Set via `--set-json 'training.model_args.<key>=<json>'`. Presets `fastvit-384/512(-attn)(-l)` preset these.

| `model_args` key | Default (preset `fastvit-384`) | Valid values | Notes |
|------------------|-------------------------------|--------------|-------|
| `layers` | `[2,2,4,2]` | list[int], one per stage (4 stages) | blocks per stage; `-l` variants use `[4,4,12,4]` |
| `embed_dims` | `[48,96,192,384]` | list[int], len 4 | channel width per stage; `512` variants use `[64,128,256,512]` |
| `mlp_ratios` | `[3,3,3,3]` | list[int], len 4 | MLP expansion per stage |
| `downsamples` | `[True,True,True,True]` | list[bool], len 4 | downsample between stages |
| `token_mixers` | `["repmixer",...]` | each `"repmixer"` or `"attention"` | `-attn` variants set the **last** stage to `"attention"` |
| `repmixer_kernel_size` | `3` | int | |
| `drop_rate` | `0` | float | |
| `drop_path_rate` | `0` | float | |
| `inference_mode` | `False` | bool | reparam for inference |

*Making the network "bigger":* raise `embed_dims` and/or `layers`.
*"Add attention":* set the last `token_mixers` entry to `"attention"`.

### ViT — `model_name: "vit"`
| `model_args` key | Default | Valid | Notes |
|---|---|---|---|
| `pretrained` | `False` | bool | `True` → ImageNet ViT_B_16 weights **and requires 224×224 input** (set `data.resize={h:224,w:224}`; the `vit-pretrained` preset does both) |

### ViT-CLIP — `model_name: "vit_clip"` (or `"vit_clip_freeze"`)
No `model_args`. Fixed CLIP-ViT-B/32 backbone (frozen). **Needs 224×224 input** (`vit-clip` preset sets `resize`).

### ResNet — `model_name: "resnet18|34|50|101|152"`
| `model_args` key | Default | Valid | Notes |
|---|---|---|---|
| `edge_filter` | `False` | bool | apply an edge-kernel filter to input |

### EfficientNet / CNN — `efficientnet_b0|b4`, `nvidia_efficientnet_widese_b0|b4`, `cnn`
No `model_args` (EfficientNet loads pretrained NVIDIA hub weights; `cnn` is a plain 5-layer net).

### Chunk-action (action horizon) models
Base class accepts `chunk_size` and `with_command`. Pair with a
`*_chunkaction` **dataset type** and `data.action_length` (see Group D).

| Model | `model_args` | Defaults / valid |
|-------|--------------|------------------|
| `resnet18|34|50|101|152_gru_chunkaction` | `chunk_size`, `with_command` | `chunk_size=10`, `with_command=False` |
| `vit_gru_chunkaction` | `chunk_size`, `with_command`, `pretrained`, `adapter_dim` | `pretrained=False`; `adapter_dim=None` (int → reduce 768→dim) |
| `vit_chunkaction` | `pretrained` | `False` |
| `vit_clip_gru_chunkaction` | `chunk_size`, `with_command`, `adapter_dim`, `freeze` | `adapter_dim=None` (→ reduce 512→dim), `freeze=False` |
| `vit_clip_chunkaction` / `vit_clip_freeze_chunkaction` | none | fixed `chunk_size=10` |

### Other families (advanced/rare)
- **Instructed** (need `history_length`/instruction data + an instructed dataset type): `vilt`, `vilt_freeze`, `vilt_freeze_middle`, `vilt_hist`, `vilt_rnn`, `blip`, `vit_two_token`.
- **Multi-action**: `vit_multi`, `vit_multi_freeze`, `resnet18_multi`.
- **Sequence**: `vit_clip_gru`, `vit_clip_gru_freeze`.
- **Waymo / embedding**: `multi_camera_waypoint_gru`, `embd_attn`, `no_embd`.

---

## Group C — Image pipeline (`data.*`)

Applies to the **mobile** dataset types (`ios_dataset`, `android_dataset`, and
their chunk/instruct/multi variants). Crop/resize/steering defaults differ per
camera FOV — start from the `ios_data` or `android_data` preset rather than
setting from scratch. Use `--set-json` for the list/object-valued ones.

| Key | Default (`ios_data` preset) | Valid | Notes |
|-----|------------------------------|-------|-------|
| `data.crop_x` | `[130, 510]` | `[int, int]` px | horizontal crop window; `android_data` uses `[160,480]` |
| `data.crop_y` | `[140, 330]` | `[int, int]` px | vertical crop window; `android_data` uses `[85,245]` |
| `data.resize` | `{h:160, w:320}` | `{h:int, w:int}` | post-crop size fed to model; **ViT/CLIP need `{h:224,w:224}`** |
| `data.augment` | `{brightness:0.1, contrast:0.1, hue:0.05, saturation:0.05}` | object of floats | color jitter magnitudes |
| `data.random_shift` | `20` | int px | random horizontal shift for augmentation |
| `data.shift_label` | `1` (ios) / — | int | frame shift applied to labels |
| `data.steering_factor` | `142` (ios) / `120` (android) | float | pixel↔steering conversion (tied to camera FOV) |
| `data.max_augmentation` | `130` (ios) / `160` (android) | float | max augmentation magnitude |

---

## Group D — Dataset & sampling (`data.*`)

**`data.type`** selects the loader (dispatched in `training/__main__.py`). Set it
with `--set-json 'data.type="..."'`.

| `data.type` | Use with | Notes |
|-------------|----------|-------|
| `ios_dataset` | single-image BC (fastvit, vit, resnet, cnn, efficientnet) | requires `seq_len==1` |
| `android_dataset` | same, Android-collected data | different crop/steering defaults |
| `ios_dataset_chunkaction` | `*_gru_chunkaction` / chunk-action models | reads `action_length` |
| `ios_dataset_multiaction` | `*_multi` models | |
| `ios_dataset_instructed` | `vilt`/`blip`/`vit_two_token` | extra `processor`, `instruction_augmentation` |
| `carla` / `carla_instruct` | CARLA sim data | uses `training`/`validation`/`obstacles` path lists, not `dataset_paths` |
| `lmdrive` | LMDrive | `dataset_root`, `action_stride`, `action_length` |
| `waymo_e2e` | Waymo end-to-end | `dataset_root`, `use_embd`, `embd_noise` |

| Key | Default | Valid | Applies to |
|-----|---------|-------|------------|
| `data.dataset_paths` | preset placeholder `["dataset"]` | list[str] | mobile types — **replace** via `--set-json` (don't `--data-path`, which appends) |
| `data.max_dataset_size` | `20000` | int | caps training samples loaded |
| `data.max_dataset_size_test` | `null` (→ `max_dataset_size//10`) | int/null | caps test samples |
| `data.train_test_split` | `0.8` | 0–1 float | train fraction |
| `data.random_split` | `false` | bool | random vs. sequential split |
| `data.resample` | `true` | bool | resample dataset |
| `data.repeat` | `5` | int | times dataset is repeated per epoch |
| `data.seq_len` | `1` | int ≥1 | `>1` switches to the sequence loader (forces `preprocessing=false`, `resample=false`) |
| `data.history_length` | `0` | int ≥0 | past actions/frames included — **only meaningful for instructed / history models** |
| `data.action_length` | `1` | int | chunk/lmdrive — number of actions predicted (match model `chunk_size`) |
| `data.action_stride` | — | int | lmdrive — stride between predicted actions |
| `data.num_workers` | `16` | int | DataLoader workers (lower if RAM/CPU bound) |
| `data.num_workers_preprocessing` | `null` | int/null | preprocessing workers |
| `data.preprocessing` | `true` | bool | preload images to RAM |
| `data.persistent_workers` | `true` | bool | keep workers alive across epochs |
| `data.processing_device` | `"cpu"` | `"cpu"`/`"cuda"` | device for image processing |

---

## Group E — Optimization & schedule (`training.*`)

| Key | Default | Valid | Notes |
|-----|---------|-------|-------|
| `training.epochs` | `100` | int | total epochs |
| `training.batch_size` | `1024` | int | **effective** batch size |
| `training.mini_batch` | `128` | int (must divide `batch_size`) | per-step micro-batch; grad-accum steps = `batch_size // mini_batch`. **Lower this (not `batch_size`) to fix OOM while keeping the effective batch.** |
| `training.optimizer` | `"Adam"` | any `torch.optim` class name (`Adam`, `AdamW`, `SGD`, …) | |
| `training.optimizer_args` | `{lr: 0.0001}` | object | kwargs to the optimizer; set via `--set training.optimizer_args.lr=5e-4` or `--set-json` for more keys (e.g. `weight_decay`, `momentum`) |
| `training.scheduler` | `"constant"` | `"constant"` or `"warmup_cosine"` | `build_config.py --scheduler sched_cosine` overlays the cosine preset |
| `training.scheduler_args` | — | object | for `warmup_cosine`: `{warmup_steps: 1000}` |
| `training.clip_grad` | `1.0` | float (or omit to disable) | gradient-norm clip threshold |
| `training.save_freq` | `1` | int | checkpoint every N epochs |
| `training.mixing` | `false` | bool | sample-mixing across datasets |
| `training.device` | `"cuda"` | torch device str | |
| `accelerate` | `false` | bool | HuggingFace Accelerate (distributed) |

---

## Group F — Hooks (rare)

| Key | Default | Valid | Notes |
|-----|---------|-------|-------|
| `training.save_hook` | `null` | shell command str | run after each checkpoint save |
| `training.finish_hook` | `null` | shell command str | run once after training finishes |
| `training.evaluate_hook` | `null` | `"module.function"` path | custom eval callback (gated by `training.eval_hook`) |

---

## Resume keys (see SKILL.md "How resume works")

| Key | Value | Notes |
|-----|-------|-------|
| `resume` | `true` | triggers `resume()` on init |
| `load_dir` | original run's `models_ckpt` (`save_dir/run_name`) | dir holding `ckpt/ckpt_<n>.pth`; `jobctl.py resume <job_id>` sets both and submits a new job |
