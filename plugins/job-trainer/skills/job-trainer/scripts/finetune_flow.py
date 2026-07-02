#!/usr/bin/env python3
"""finetune_flow — interactive, end-to-end fine-tuning of a finished AV model.

This is the scripted version of the "fine-tune a finished model on a hidden
challenge dataset" workflow. It walks the whole flow start to finish:

    1. resolve a FINISHED base job  (flag, or pick from your finished jobs)
    2. pick a hidden challenge dataset  (flag, or pick from the ready ones)
    3. choose lr / epochs  (flags, or prompted — blank inherits the base run)
    4. confirm a summary, submit the warm-start job
    5. optionally monitor it to completion

Anything you don't pass on the command line is prompted for interactively, so
`python finetune_flow.py` with no args runs the guided flow. It reuses the
socket client, the hidden-challenge manifest, and the warm-start config
transform from jobctl.py (no logic is duplicated).

Config (same as jobctl): $SCHEDULER_SOCKET, $SCHEDULER_USER, or --socket/--user.

Examples:
    python finetune_flow.py                       # fully guided
    python finetune_flow.py --job <id> --challenge clockwise --lr 5e-5 --epochs 5
    python finetune_flow.py --challenge clockwise --no-monitor
"""

import argparse
import os
import sys
import types

# Reuse everything from the sibling jobctl module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobctl


def _die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _interactive():
    return sys.stdin.isatty()


def _ask(prompt, default=None):
    """Prompt for a line. Returns default (may be None) on blank input."""
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return raw if raw else default


def _ask_choice(prompt, options, labeler):
    """Numbered single-choice picker over a list. Returns the chosen item."""
    if len(options) == 1:
        only = options[0]
        print(f"{prompt}: {labeler(only)}  (only option — selected)")
        return only
    print(f"{prompt}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {labeler(opt)}")
    while True:
        raw = _ask("  choose #")
        if raw is None:
            _die("No selection made.")
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print("  invalid choice; enter a number from the list.")


def _base_args(args):
    """A namespace carrying socket/user for jobctl's call()/get_status()."""
    ns = types.SimpleNamespace(socket=args.socket, user=args.user)
    sock = ns.socket or os.environ.get("SCHEDULER_SOCKET")
    user = ns.user or os.environ.get("SCHEDULER_USER")
    if not sock:
        _die("No socket. Set $SCHEDULER_SOCKET or pass --socket TCP:host:port.")
    if not user:
        _die("No user_id. Set $SCHEDULER_USER or pass --user.")
    return ns


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def resolve_base_job(args, base):
    """Return a finished base job id, prompting from the user's jobs if needed."""
    if args.job:
        js = jobctl.get_status(base, args.job)
        status = js.get("status")
        if status != "finished":
            print(f"warning: base job {args.job} status is {status!r}, not "
                  "'finished' — warm-start needs a checkpoint.", file=sys.stderr)
        elif js.get("returncode") not in (0, None):
            print(f"warning: base job {args.job} finished with returncode "
                  f"{js.get('returncode')} — it may not have a usable checkpoint.",
                  file=sys.stderr)
        return args.job

    if not _interactive():
        _die("No --job given and not running interactively. Pass --job <id>.")

    resp = jobctl.call(base, {"command": "user:get_all_jobs"})
    finished = []
    for jid in resp.get("jobs", []):
        js = jobctl.get_status(base, jid)
        if js.get("status") == "finished" and js.get("returncode") in (0, None):
            cfg = js.get("config", {}) or {}
            spec = (cfg.get("training", {}) or {}).get("model_spec", "?")
            model = (cfg.get("training", {}) or {}).get("model_name", "?")
            finished.append((jid, f"{model}-{spec}", js.get("finish_time")))
    if not finished:
        _die("You have no successfully-finished jobs to fine-tune from.")
    # Most-recently-finished first.
    finished.sort(key=lambda r: r[2] or 0, reverse=True)
    chosen = _ask_choice(
        "Pick a finished base run to fine-tune from",
        finished,
        lambda r: f"{r[1]:30s} {r[0]}")
    return chosen[0]


def resolve_challenge(args, base):
    """Return a ready challenge key, prompting from the manifest if needed."""
    challenges = jobctl.load_hidden_challenges()
    ready = []
    for key, ch in challenges.items():
        paths = ch.get("dataset_paths") or []
        if paths and not any("PLACEHOLDER" in str(p) for p in paths):
            ready.append((key, ch))
    if not ready:
        _die("No challenges with real dataset paths in hidden_challenges.json.")

    if args.challenge:
        match = dict(ready).get(args.challenge)
        if match is None:
            _die(f"Challenge {args.challenge!r} is unknown or not ready "
                 "(check `jobctl.py challenges`).")
        return args.challenge

    if not _interactive():
        _die("No --challenge given and not running interactively.")

    chosen = _ask_choice(
        "Pick a hidden challenge dataset",
        ready,
        lambda kc: f"{kc[0]:14s} — {kc[1].get('description', '')}")
    return chosen[0]


def _ask_number(prompt, cast):
    """Prompt for a number, re-prompting on bad input. Blank => None (inherit)."""
    while True:
        v = _ask(prompt)
        if not v:
            return None
        try:
            return cast(v)
        except ValueError:
            print(f"  not a valid {cast.__name__}; try again (or blank to inherit).")


def resolve_hparams(args):
    """Return (lr, epochs, batch_size); prompt if interactive and unset."""
    lr, epochs, batch = args.lr, args.epochs, args.batch_size
    if _interactive():
        if lr is None:
            lr = _ask_number("Fine-tune learning rate (blank = inherit base)", float)
        if epochs is None:
            epochs = _ask_number("Fine-tune epochs (blank = inherit base)", int)
        if batch is None:
            batch = _ask_number("Batch size (blank = inherit base)", int)
    return lr, epochs, batch


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Interactive end-to-end fine-tune of a finished AV model.")
    p.add_argument("--socket", help="TCP:host:port or UNIX:path (else $SCHEDULER_SOCKET)")
    p.add_argument("--user", help="user_id (else $SCHEDULER_USER)")
    p.add_argument("--job", help="finished base job id (else pick interactively)")
    p.add_argument("--challenge", help="hidden challenge key (else pick interactively)")
    p.add_argument("--name", help="run name -> training.model_spec")
    p.add_argument("--lr", type=float, help="fine-tune learning rate")
    p.add_argument("--epochs", type=int, help="fine-tune epochs")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="batch size")
    p.add_argument("--overlay", help="extra dotted overrides, e.g. training.scheduler=constant")
    p.add_argument("--yes", action="store_true", help="skip the confirm prompt")
    p.add_argument("--monitor", dest="monitor", action="store_true", default=None,
                   help="monitor after submit")
    p.add_argument("--no-monitor", dest="monitor", action="store_false",
                   help="don't monitor after submit")
    args = p.parse_args(argv)

    base = _base_args(args)

    job = resolve_base_job(args, base)
    challenge = resolve_challenge(args, base)
    lr, epochs, batch = resolve_hparams(args)

    # Summary + confirm.
    print("\n--- fine-tune plan ---")
    print(f"  base job   : {job}")
    print(f"  challenge  : {challenge}")
    print(f"  lr         : {lr if lr is not None else '(inherit base)'}")
    print(f"  epochs     : {epochs if epochs is not None else '(inherit base)'}")
    print(f"  batch_size : {batch if batch is not None else '(inherit base)'}")
    print(f"  run name   : {args.name or '(auto: <base>-ft-' + challenge + ')'}")
    print("  mechanism  : warm-start (resume=false + load_dir); base run untouched")
    if not args.yes and _interactive():
        if (_ask("Submit this fine-tune? [y/N]") or "").lower() not in ("y", "yes"):
            _die("Aborted.")

    # Submit via jobctl's transform (single source of truth).
    ft_args = types.SimpleNamespace(
        socket=args.socket, user=args.user, job_id=job, challenge=challenge,
        name=args.name, lr=lr, epochs=epochs, batch_size=batch, overlay=args.overlay)
    resp = jobctl.cmd_finetune(ft_args)
    new_id = (resp or {}).get("job_id")
    if not new_id:
        _die("Submit did not return a job_id.")
    print(f"\nsubmitted fine-tune job: {new_id}")

    # Monitor?
    do_monitor = args.monitor
    if do_monitor is None:
        do_monitor = _interactive() and \
            (_ask("Monitor it to completion now? [Y/n]") or "y").lower() in ("y", "yes")
    if do_monitor:
        mon_args = types.SimpleNamespace(
            socket=args.socket, user=args.user, job_id=new_id, interval=15.0)
        jobctl.cmd_monitor(mon_args)
    else:
        print(f"Track it with:  python jobctl.py monitor {new_id}")


if __name__ == "__main__":
    main()
