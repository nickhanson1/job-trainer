#!/usr/bin/env python3
"""build_config — compose an AV training job_config from presets + user choices.

This reproduces, in standalone form, what the notebook's config widget does:
start from the base training preset, overlay a model preset and one or more
dataset presets, optionally a scheduler preset, then apply the user's choices
(run name, learning rate, epochs, batch size, arbitrary overrides). The result
is written as JSON, ready for:  jobctl.py submit --config-file <out>

Overlay semantics follow av-training/docs/CONFIG_SYSTEM.md:
  - overlay B onto A: recursive dict merge; later values win.
  - dataset presets are merged with APPEND-LIST so their dataset_paths combine
    (same as the training CLI's `--data D1 D2 ...`).
  - the special keys `_include` (overlay) and `_append` (append-list) inside a
    preset are resolved first, recursively.

Requires PyYAML to read the preset .yaml files (the presets are YAML). If PyYAML
is unavailable, pass already-merged JSON configs instead via --base.

Examples:
  python build_config.py --model fastvit-384 --data hallway_data \
      --name my-hallway-run --lr 1e-4 --epochs 80 -o run.json

  # raw dataset paths instead of a preset, plus arbitrary overrides:
  python build_config.py --model resnet18 \
      --data-path /studentdata/ios/dataset/track_cones \
      --name resnet-track --set training.batch_size=512,data.num_workers=8 \
      -o run.json

  # train on the student's own data: start from ios_data, REPLACE dataset_paths
  # (drops the preset's "dataset" placeholder) and set list/nested values:
  python build_config.py --model fastvit-384 --data ios_data --name kmahon-mydata \
      --set-json 'data.dataset_paths=["/studentdata/.../kmahon^track_v2"]' \
      --set-json 'data.crop_x=[130,510]' -o run.json

Overrides: --set takes SCALARS only (int/float/bool/str, comma-separated). For
lists, nested objects, or explicit types, use --set-json 'KEY=<json>' (repeatable,
one key per flag); it JSON-parses the value and replaces the key.
"""

import argparse
import json
import os
import sys

# Presets are bundled alongside this script in scripts/presets/ (a copy of
# av-training/training/preset/), so build_config runs without an av-training
# checkout. Both the named-preset lookup dir and the _include/_append repo root
# point here. Override with --preset-dir / --repo-root to use a live checkout.
PRESET_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "presets")
REPO_ROOT_DEFAULT = PRESET_DIR_DEFAULT


def _die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        _die("PyYAML is required to read presets. `pip install pyyaml`, or "
             "supply pre-merged JSON configs via --base.")


# --------------------------------------------------------------------------- #
# Overlay semantics (per docs/CONFIG_SYSTEM.md)
# --------------------------------------------------------------------------- #
def overlay(a, b, append_list=False):
    """Return A + B (or A (+)append B). Recursive; does not mutate inputs."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = overlay(a[k], v, append_list) if k in a else v
        return out
    if append_list and isinstance(a, list) and isinstance(b, list):
        return a + b
    return b  # B wins for scalars / type mismatch / non-append lists


def load_config(path, repo_root):
    """Load a YAML/JSON config file and resolve its _include / _append keys."""
    if not os.path.exists(path):
        _die(f"Config/preset not found: {path}")
    if path.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
    else:
        yaml = _require_yaml()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    return _resolve_special(raw, repo_root)


def _resolve_special(cfg, repo_root):
    """C = (I1+..+IN + A1(+)..(+)AN) + C\\{special}, applied recursively."""
    if not isinstance(cfg, dict):
        return cfg
    includes = cfg.get("_include", []) or []
    appends = cfg.get("_append", []) or []
    base = {}
    for inc in includes:
        base = overlay(base, load_config(_resolve_path(inc, repo_root), repo_root))
    for app in appends:
        base = overlay(base, load_config(_resolve_path(app, repo_root), repo_root),
                       append_list=True)
    rest = {k: _resolve_special(v, repo_root)
            for k, v in cfg.items() if k not in ("_include", "_append")}
    return overlay(base, rest)


def _resolve_path(p, repo_root):
    """Preset _include paths are repo-relative (e.g. training/preset/x.yaml)."""
    if os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(repo_root, p)


def find_preset(name, preset_dir):
    """Return the preset path for a name/stem/path, or None if not found."""
    if os.path.exists(name):
        return name
    cand = os.path.join(preset_dir, name if name.endswith((".yaml", ".yml"))
                        else name + ".yaml")
    return cand if os.path.exists(cand) else None


def resolve_preset(name, preset_dir):
    """Accept a bare preset name, a file stem, or a path."""
    path = find_preset(name, preset_dir)
    if path is None:
        _die(f"Unknown preset {name!r}; not found in {preset_dir}")
    return path


# --------------------------------------------------------------------------- #
# Overrides
# --------------------------------------------------------------------------- #
def coerce(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    return s


def set_path(cfg, dotted, value):
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            _die(f"Cannot set {dotted}: {p} is not a mapping")
    node[parts[-1]] = value


def apply_set(cfg, expr):
    if not expr:
        return
    for pair in expr.split(","):
        if "=" not in pair:
            _die(f"Bad --set item {pair!r}; expected key=value")
        k, v = pair.split("=", 1)
        set_path(cfg, k.strip(), coerce(v))


def apply_set_json(cfg, exprs):
    """Apply --set-json overrides: key=<json>, one per flag, repeatable.

    The value is parsed as JSON, so it can be a list, nested object, or any
    typed scalar (quote bare strings, e.g. 'data.type="ios_dataset_chunkaction"').
    Unlike --set (scalars only, append semantics never apply), this REPLACES the
    value at the key outright — which is how you drop a preset's placeholder
    dataset_paths: --set-json 'data.dataset_paths=["/real/path"]'.
    """
    for expr in exprs or []:
        if "=" not in expr:
            _die(f"Bad --set-json item {expr!r}; expected key=<json>")
        k, v = expr.split("=", 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError as e:
            _die(f"Bad JSON for --set-json {k.strip()!r}: {v!r} ({e})")
        set_path(cfg, k.strip(), val)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build(args):
    repo_root = args.repo_root
    preset_dir = args.preset_dir

    # 1. base
    cfg = load_config(resolve_preset(args.base, preset_dir), repo_root)

    # 2. model: a preset (sets model_name + model_args) or a raw model_name
    if args.model:
        path = find_preset(args.model, preset_dir)
        if path:
            cfg = overlay(cfg, load_config(path, repo_root))
        else:
            set_path(cfg, "training.model_name", args.model)

    # 3. datasets — presets merge with append-list so dataset_paths combine
    for d in args.data or []:
        cfg = overlay(cfg, load_config(resolve_preset(d, preset_dir), repo_root),
                      append_list=True)
    if args.data_path:
        cfg = overlay(cfg, {"data": {"dataset_paths": list(args.data_path)}},
                      append_list=True)

    # 4. scheduler preset (optional)
    if args.scheduler:
        cfg = overlay(cfg, load_config(resolve_preset(args.scheduler, preset_dir), repo_root))

    # 5. friendly knobs -> canonical keys
    if args.name is not None:
        set_path(cfg, "training.model_spec", args.name)
    if args.lr is not None:
        set_path(cfg, "training.optimizer_args.lr", args.lr)
    if args.epochs is not None:
        set_path(cfg, "training.epochs", args.epochs)
    if args.batch_size is not None:
        set_path(cfg, "training.batch_size", args.batch_size)
    if args.user_id is not None:
        set_path(cfg, "user_id", args.user_id)

    # 6. arbitrary overrides (highest precedence): scalars via --set, then
    #    lists/nested/typed via --set-json (JSON-parsed, replaces the key).
    apply_set(cfg, args.set)
    apply_set_json(cfg, args.set_json)

    return cfg


def warn_missing(cfg):
    notes = []
    if not cfg.get("training", {}).get("model_name"):
        notes.append("training.model_name is unset — pass --model.")
    if not cfg.get("data", {}).get("dataset_paths"):
        notes.append("data.dataset_paths is empty — pass --data or --data-path.")
    for n in notes:
        print(f"warning: {n}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build an AV training job_config.")
    p.add_argument("--base", default="training",
                   help="base preset name or path (default: training)")
    p.add_argument("--model", help="model preset name or raw model_name "
                   "(e.g. fastvit-384, resnet18, vit-clip)")
    p.add_argument("--data", action="append",
                   help="dataset preset name (repeatable; paths are combined)")
    p.add_argument("--data-path", action="append",
                   help="raw dataset path to add (repeatable)")
    p.add_argument("--scheduler", help="scheduler preset (e.g. sched_cosine)")
    p.add_argument("--name", help="run name -> training.model_spec")
    p.add_argument("--lr", type=float, help="learning rate -> optimizer_args.lr")
    p.add_argument("--epochs", type=int, help="training.epochs")
    p.add_argument("--batch-size", type=int, dest="batch_size",
                   help="training.batch_size")
    p.add_argument("--user-id", dest="user_id",
                   help="stamp user_id into the config (optional)")
    p.add_argument("--set", help="scalar overrides, e.g. data.num_workers=8,training.mixing=true")
    p.add_argument("--set-json", action="append", dest="set_json", metavar="KEY=JSON",
                   help="list/nested/typed override from JSON; repeatable, one key "
                        "per flag. e.g. 'data.crop_x=[130,510]' or "
                        "'data.dataset_paths=[\"/path/a\",\"/path/b\"]'. Replaces the key.")
    p.add_argument("--preset-dir", default=PRESET_DIR_DEFAULT, dest="preset_dir")
    p.add_argument("--repo-root", default=REPO_ROOT_DEFAULT, dest="repo_root",
                   help="av-training root (for resolving _include paths)")
    p.add_argument("-o", "--out", help="write config JSON here (else stdout)")
    args = p.parse_args(argv)

    cfg = build(args)
    warn_missing(cfg)
    text = json.dumps(cfg, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
