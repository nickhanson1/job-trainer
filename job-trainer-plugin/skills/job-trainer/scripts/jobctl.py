#!/usr/bin/env python3
"""jobctl — manage AV training jobs on the shared GPU job-queue scheduler.

The scheduler is NOT a REST API. It is a socket server that speaks one JSON
request -> one JSON response per connection (see av-training/utils/job_queue.py,
class JobQueueClient). This helper reimplements that minimal client so it runs
with only the Python standard library — no need to install the av-training deps.

Every request carries a `user_id` (high-trust classroom setup; no auth). Job
stdout comes back base64(gzip(bytes)); this tool decodes it for you.

Configuration (env vars, overridable per-command):
    SCHEDULER_SOCKET   "TCP:<host>:<port>"  or  "UNIX:<path>"   e.g. TCP:127.0.0.1:8091
    SCHEDULER_USER     the student's user_id                    e.g. kmahon

Examples:
    python jobctl.py list
    python jobctl.py datasets [--json]        # your uploaded datasets (FS glob)
    python jobctl.py status <job_id> [--config]
    python jobctl.py logs <job_id> --tail 200
    python jobctl.py monitor <job_id>
    python jobctl.py submit --config-file run.json [--overlay training.epochs=80]
    python jobctl.py stop <job_id>
    python jobctl.py resume <job_id>
"""

import argparse
import base64
import gzip
import json
import os
import re
import socket
import sys
import time

TERMINAL_STATES = {"finished", "cancelled", "canceled"}

# Where the data-upload server drops student uploads, one folder per dataset
# named "<user_id>^<dataset_name>". Override with --datasets-dir or
# $SCHEDULER_DATASETS_DIR if the path is mounted elsewhere.
DATASETS_DIR_DEFAULT = "/studentdata/past_summer_camp/summer2026/datasets"


# --------------------------------------------------------------------------- #
# Socket client (mirrors JobQueueClient.send_msg)
# --------------------------------------------------------------------------- #
def send_msg(sock_addr, msg):
    """Open a connection, send one JSON object, read one JSON object back."""
    kind, addr = sock_addr.split(":", 1)
    if kind == "TCP":
        host, port = addr.rsplit(":", 1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, int(port)))
    elif kind == "UNIX":
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(addr)
    else:
        _die(f"Bad socket spec {sock_addr!r}; use TCP:<host>:<port> or UNIX:<path>")

    try:
        client.sendall(json.dumps(msg).encode())
        client.shutdown(socket.SHUT_WR)
        with client.makefile("r") as f:
            return json.load(f)
    except (ConnectionError, OSError) as e:
        _die(f"Could not talk to scheduler at {sock_addr}: {e}")
    except json.JSONDecodeError as e:
        _die(f"Scheduler returned non-JSON: {e}")
    finally:
        client.close()


def call(args, msg):
    """Resolve config, send, and surface scheduler-level errors."""
    sock = args.socket or os.environ.get("SCHEDULER_SOCKET")
    user = args.user or os.environ.get("SCHEDULER_USER")
    if not sock:
        _die("No socket. Set SCHEDULER_SOCKET or pass --socket TCP:host:port.")
    if not user:
        _die("No user_id. Set SCHEDULER_USER or pass --user.")
    msg = {"user_id": user, **msg}
    resp = send_msg(sock, msg)
    if isinstance(resp, dict) and resp.get("status") == "error":
        _die(f"Scheduler error: {resp.get('msg', resp)}")
    return resp


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _emit(obj):
    print(json.dumps(obj, indent=2, default=str))


def decode_output(job_status):
    """Decode the base64+gzip stdout blob into text ('' if absent)."""
    blob = job_status.get("output")
    if not blob:
        return ""
    try:
        return gzip.decompress(base64.b64decode(blob)).decode("utf-8", "replace")
    except (ValueError, OSError):
        return ""


def parse_progress(stdout):
    """Pull (current_epoch, eta) from tqdm 'Epochs:' lines, like the notebook."""
    epoch, eta = None, None
    for line in re.split(r"[\r\n]", stdout):
        m = re.match(r"Epochs:.*\| (\d+)/\d+ \[.*<(.*?),", line.strip())
        if m:
            epoch, eta = int(m.group(1)), m.group(2)
    return epoch, eta


def get_status(args, job_id):
    resp = call(args, {"command": "job:status", "job_id": job_id})
    js = resp.get("job_status", resp)
    # Some server versions double-wrap; unwrap defensively.
    if isinstance(js, dict) and "job_status" in js and "status" in js.get("job_status", {}):
        js = js["job_status"]
    return js


def load_config_file(path):
    with open(path, "r") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            _die("PyYAML not available; convert the config to JSON or install pyyaml.")
        return yaml.safe_load(text)
    return json.loads(text)


def apply_overlay(config, overlay):
    """Apply 'a.b=val,c=val2' dotted overrides onto a nested config dict."""
    if not overlay:
        return config
    for pair in overlay.split(","):
        if "=" not in pair:
            _die(f"Bad --overlay item {pair!r}; expected key=value")
        key, raw = pair.split("=", 1)
        val = _coerce(raw)
        node = config
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return config


def _coerce(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list(args):
    resp = call(args, {"command": "user:get_all_jobs"})
    job_ids = resp.get("jobs", [])
    rows = []
    for jid in job_ids:
        js = get_status(args, jid)
        epoch, eta = parse_progress(decode_output(js))
        cfg = js.get("config", {}) or {}
        training = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        rows.append({
            "job_id": jid,
            "status": js.get("status"),
            "model": training.get("model_name"),
            "spec": training.get("model_spec"),
            "epoch": epoch,
            "total_epochs": training.get("epochs"),
            "eta": eta,
            "returncode": js.get("returncode"),
        })
    _emit(rows)


def cmd_status(args):
    js = get_status(args, args.job_id)
    out = {
        "job_id": args.job_id,
        "status": js.get("status"),
        "returncode": js.get("returncode"),
        "reason": js.get("reason"),
        "queued_time": js.get("queued_time"),
        "finish_time": js.get("finish_time"),
    }
    epoch, eta = parse_progress(decode_output(js))
    out["epoch"], out["eta"] = epoch, eta
    if args.config:
        out["config"] = js.get("config")
    _emit(out)


def cmd_logs(args):
    js = get_status(args, args.job_id)
    text = decode_output(js)
    lines = text.splitlines()
    if args.tail and len(lines) > args.tail:
        lines = lines[-args.tail:]
    print("\n".join(lines))


def cmd_monitor(args):
    last = None
    while True:
        js = get_status(args, args.job_id)
        status = js.get("status")
        epoch, eta = parse_progress(decode_output(js))
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {args.job_id}: {status}"
        if epoch is not None:
            line += f"  epoch {epoch}  eta {eta}"
        if status != last or epoch is not None:
            print(line)
        last = status
        if status in TERMINAL_STATES:
            rc = js.get("returncode")
            print(f"--- terminal: {status}"
                  + (f" (returncode {rc})" if rc is not None else "") + " ---")
            return
        time.sleep(args.interval)


def cmd_submit(args):
    config = load_config_file(args.config_file)
    if not isinstance(config, dict):
        _die("Config file must contain a JSON/YAML object (dict).")
    apply_overlay(config, args.overlay)
    resp = call(args, {"command": "job:new", "job_config": config})
    _emit(resp)


def cmd_stop(args):
    resp = call(args, {"command": "job:stop", "job_id": args.job_id})
    _emit(resp)


def load_hidden_challenges():
    """Load the instructor-curated hidden fine-tune datasets manifest.

    These datasets are deliberately NOT in the student-facing preset catalog —
    they are surprise/challenge tasks (e.g. "drive clockwise") offered only
    after a base run finishes. The manifest lives next to this script.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hidden_challenges.json")
    if not os.path.exists(path):
        _die(f"No hidden_challenges.json next to jobctl.py ({path}).")
    with open(path) as f:
        data = json.load(f)
    return data.get("challenges", data)


def _challenge_paths(ch, key):
    paths = ch.get("dataset_paths") or []
    if not paths or any("PLACEHOLDER" in str(p) for p in paths):
        _die(f"Challenge {key!r} has no real dataset_paths yet — fill them into "
             "hidden_challenges.json (don't invent dataset paths).")
    return list(paths)


def cmd_challenges(args):
    challenges = load_hidden_challenges()
    rows = []
    for key, ch in challenges.items():
        paths = ch.get("dataset_paths") or []
        ready = bool(paths) and not any("PLACEHOLDER" in str(p) for p in paths)
        rows.append({
            "challenge": key,
            "description": ch.get("description"),
            "ready": ready,
            "num_paths": len(paths),
        })
    _emit(rows)


def cmd_finetune(args):
    challenges = load_hidden_challenges()
    ch = challenges.get(args.challenge)
    if ch is None:
        _die(f"Unknown challenge {args.challenge!r}; run `jobctl.py challenges`.")
    paths = _challenge_paths(ch, args.challenge)

    js = get_status(args, args.job_id)
    config = js.get("config")
    if not isinstance(config, dict):
        _die("Could not read the base job's config; cannot fine-tune.")

    status = js.get("status")
    if status != "finished":
        print(f"warning: base job {args.job_id} status is {status!r}, not "
              "'finished'. Warm-start loads its last.pth — make sure a "
              "checkpoint exists before relying on this.", file=sys.stderr)

    # Capture the base run's checkpoint dir BEFORE the server reassigns it.
    load_dir = config.get("models_ckpt")
    if not load_dir:
        save_dir, run_name = config.get("save_dir"), config.get("run_name")
        if not (save_dir and run_name):
            _die("Base config has no checkpoint dir (models_ckpt / "
                 "save_dir+run_name); nothing to warm-start from.")
        load_dir = f"{save_dir.rstrip('/')}/{run_name}"

    # Warm-start (NOT resume): resume=False => trainer.load() loads last.pth
    # with a fresh epoch counter. See trainer.py load().
    config["resume"] = False
    config["load_dir"] = load_dir

    # Swap in the hidden dataset.
    data = config.setdefault("data", {})
    data["dataset_paths"] = paths
    if ch.get("data_type"):
        data["type"] = ch["data_type"]
    for k, v in (ch.get("data_overrides") or {}).items():
        data[k] = v

    # New run name so initialize() writes a fresh dir and the base run is safe.
    training = config.setdefault("training", {})
    base_spec = training.get("model_spec", "run")
    training["model_spec"] = args.name or f"{base_spec}-ft-{args.challenge}"
    # Let the server stamp a fresh run_name / models_ckpt at launch.
    config.pop("run_name", None)
    config.pop("models_ckpt", None)

    # Hyperparameters: asked each time (per skill design), else inherit base.
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.lr is not None:
        training.setdefault("optimizer_args", {})["lr"] = args.lr
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size

    apply_overlay(config, args.overlay)

    resp = call(args, {"command": "job:new", "job_config": config})
    print(f"fine-tuning base {args.job_id} on challenge '{args.challenge}' "
          f"(warm-start from {load_dir}; fresh run "
          f"'{training['model_spec']}')", file=sys.stderr)
    _emit(resp)
    return resp


def cmd_datasets(args):
    """List the student's OWN uploaded datasets by globbing the server FS.

    This is NOT a scheduler call — it globs `<datasets_dir>/<user_id>^*` exactly
    like the notebook (Path(dir).glob(f"{user_id}^*")). Needs the path to be
    reachable (run on the GPU server, or point --datasets-dir at a mount).
    """
    import glob as _glob
    user = args.user or os.environ.get("SCHEDULER_USER")
    if not user:
        _die("No user_id. Set SCHEDULER_USER or pass --user.")
    root = (args.datasets_dir or os.environ.get("SCHEDULER_DATASETS_DIR")
            or DATASETS_DIR_DEFAULT)
    if not os.path.isdir(root):
        _die(f"Datasets dir not reachable: {root}\n"
             "This globs the SERVER filesystem. Run on the GPU server, or pass "
             "--datasets-dir / set $SCHEDULER_DATASETS_DIR if it's mounted "
             "elsewhere. Otherwise ask the student for their dataset name(s) and "
             f"build paths as {root}/<user_id>^<name>.")
    paths = sorted(p for p in _glob.glob(os.path.join(root, f"{user}^*"))
                   if os.path.isdir(p))
    names = [os.path.basename(p).split("^", 1)[1] for p in paths]
    set_json = "data.dataset_paths=" + json.dumps(paths)

    if args.json:
        _emit({
            "user_id": user,
            "datasets_dir": root,
            "datasets": [{"name": n, "path": p} for n, p in zip(names, paths)],
            "set_json": set_json,
        })
        return

    if not paths:
        print(f"No datasets found for user {user!r} in {root} "
              f"(looked for '{user}^*').")
        return
    print(f"{len(paths)} dataset(s) for {user} in {root}:")
    for n, p in zip(names, paths):
        print(f"  {n}\t{p}")
    print("\nReady-to-paste for build_config.py (replaces dataset_paths):")
    print(f"  --set-json '{set_json}'")


def cmd_resume(args):
    js = get_status(args, args.job_id)
    config = js.get("config")
    if not isinstance(config, dict):
        _die("Could not read the original job's config; cannot resume.")
    load_dir = config.get("models_ckpt")
    if not load_dir:
        save_dir, run_name = config.get("save_dir"), config.get("run_name")
        if not (save_dir and run_name):
            _die("Original config has no checkpoint dir (models_ckpt / "
                 "save_dir+run_name); nothing to resume from.")
        load_dir = f"{save_dir.rstrip('/')}/{run_name}"
    config["resume"] = True
    config["load_dir"] = load_dir
    resp = call(args, {"command": "job:new", "job_config": config})
    print(f"resuming {args.job_id} from {load_dir}", file=sys.stderr)
    _emit(resp)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="Manage AV training jobs on the scheduler.")
    p.add_argument("--socket", help="TCP:host:port or UNIX:path (else $SCHEDULER_SOCKET)")
    p.add_argument("--user", help="user_id (else $SCHEDULER_USER)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list your jobs with status/progress").set_defaults(func=cmd_list)

    s = sub.add_parser("datasets",
                       help="list your uploaded datasets (server filesystem glob, not a job)")
    s.add_argument("--json", action="store_true",
                   help="machine-readable output + ready-to-paste --set-json line")
    s.add_argument("--datasets-dir", dest="datasets_dir",
                   help="datasets root (else $SCHEDULER_DATASETS_DIR or server default)")
    s.set_defaults(func=cmd_datasets)

    s = sub.add_parser("status", help="show one job's status")
    s.add_argument("job_id")
    s.add_argument("--config", action="store_true", help="also print the full config")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("logs", help="print a job's decoded stdout")
    s.add_argument("job_id")
    s.add_argument("--tail", type=int, default=200, help="last N lines (default 200)")
    s.set_defaults(func=cmd_logs)

    s = sub.add_parser("monitor", help="poll a job until it finishes")
    s.add_argument("job_id")
    s.add_argument("--interval", type=float, default=10.0, help="poll seconds")
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser("submit", help="submit a new job from a config file")
    s.add_argument("--config-file", required=True, help="JSON or YAML training config")
    s.add_argument("--overlay", help="dotted overrides, e.g. training.epochs=80,data.num_workers=8")
    s.set_defaults(func=cmd_submit)

    s = sub.add_parser("stop", help="stop a running job / cancel a queued one")
    s.add_argument("job_id")
    s.set_defaults(func=cmd_stop)

    s = sub.add_parser("resume", help="resume a job from its last checkpoint (new job)")
    s.add_argument("job_id")
    s.set_defaults(func=cmd_resume)

    sub.add_parser("challenges",
                   help="list the hidden fine-tune challenge datasets"
                   ).set_defaults(func=cmd_challenges)

    s = sub.add_parser("finetune",
                       help="warm-start a finished model on a hidden challenge dataset (new job)")
    s.add_argument("job_id", help="the FINISHED base job to fine-tune from")
    s.add_argument("--challenge", required=True,
                   help="hidden challenge key (see `jobctl.py challenges`)")
    s.add_argument("--name", help="run name -> training.model_spec "
                   "(default: <base>-ft-<challenge>)")
    s.add_argument("--lr", type=float, help="fine-tune learning rate")
    s.add_argument("--epochs", type=int, help="fine-tune epochs")
    s.add_argument("--batch-size", type=int, dest="batch_size",
                   help="override training.batch_size")
    s.add_argument("--overlay", help="dotted overrides, e.g. data.num_workers=8")
    s.set_defaults(func=cmd_finetune)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
