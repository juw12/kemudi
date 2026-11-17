"""Command line: kemudi fetch | extract | measure | report | reproduce | say | serve."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from . import __version__
from .fetch import DEFAULT_MODEL, fetch_model, is_complete, model_dir


def resolve_model(model: str) -> tuple[Path, str]:
    """Return (folder, short name). A local folder is used as is; a repo id is downloaded once."""
    if Path(model).is_dir():
        return Path(model), Path(model).name
    folder = model_dir(model)
    if not is_complete(folder):
        print(f"Downloading {model} to {folder} (about 1 GB, one time)...")
        fetch_model(model)
    return folder, model.split("/")[-1]


def open_engine(args):
    from .engine import Engine

    folder, name = resolve_model(args.model)
    return Engine.load(folder, args.threads), name


def open_calibration(name: str, args):
    from . import calibration

    folder = Path(args.calibration) if args.calibration else calibration.calibration_dir(name)
    if not (folder / "calibration.json").is_file():
        sys.exit(f"No calibration for {name} in {folder}. Run: kemudi extract --model {args.model}")
    return folder, *calibration.load(folder)


def command_fetch(args) -> None:
    print(resolve_model(args.model)[0])


def command_extract(args) -> None:
    from . import calibration
    from .extract import extract

    engine, name = open_engine(args)
    summary, units = extract(engine, calibration.load_axes_data(), name)
    folder = Path(args.calibration) if args.calibration else calibration.calibration_dir(name)
    calibration.save(folder, summary, units)
    print(f"Saved {len(units)} vectors to {folder}")


def command_measure(args) -> None:
    from . import calibration
    from .measure import measure

    engine, name = open_engine(args)
    folder, summary, axes = open_calibration(name, args)
    samples_path = folder / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8")) if samples_path.is_file() else {}
    samples.update(measure(engine, summary, axes, calibration.load_axes_data(), only=args.axis))
    calibration.save_summary(folder, summary)
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved the measurements to {folder}")


def command_report(args) -> None:
    from . import calibration
    from .report import write_report

    name = Path(args.model).name if Path(args.model).is_dir() else args.model.split("/")[-1]
    folder = Path(args.calibration) if args.calibration else calibration.calibration_dir(name)
    for path in write_report(folder, Path(args.out)):
        print(path)


def command_reproduce(args) -> None:
    """Everything behind the published numbers, in order: vectors, measurements, charts and tables."""
    for step in (command_extract, command_measure, command_report):
        step(args)


def parse_knobs(pairs: list[str]) -> dict[str, float]:
    knobs = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        knobs[name] = float(value)
    return knobs


def command_say(args) -> None:
    engine, name = open_engine(args)
    _folder, _summary, axes = open_calibration(name, args)
    knobs = parse_knobs(args.knob)
    unknown = set(knobs) - set(axes)
    if unknown:
        sys.exit(f"Unknown knob {sorted(unknown)}. Available: {sorted(axes)}")
    engine.set_knobs(axes, knobs)
    print(args.story, end="", flush=True)
    for piece, _readings in engine.stream(engine.story_prompt(args.story), args.tokens, seed=args.seed):
        print(piece, end="", flush=True)
    print()


def command_serve(args) -> None:
    from .server import App, serve

    engine, name = open_engine(args)
    _folder, summary, axes = open_calibration(name, args)
    server = serve(App(engine, axes, summary), args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Kemudi {__version__} with {name}: {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kemudi", description="Steering knobs for a local language model.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--model", default=DEFAULT_MODEL, help="repo id or local folder")
        command.add_argument("--calibration", help="folder with vectors.npz and calibration.json")
        command.add_argument("--threads", type=int, help="CPU threads for the model")
        command.set_defaults(handler=handler)
        return command

    add("fetch", command_fetch, "download the model")
    add("extract", command_extract, "find the steering vectors and check them on held-out text")
    measure = add("measure", command_measure, "measure the dose-response of every knob")
    measure.add_argument("--axis", action="append", help="measure only this axis (repeatable)")
    report = add("report", command_report, "write the charts and the results tables")
    reproduce = add("reproduce", command_reproduce, "extract, measure and report in one go (one to two hours on a CPU)")
    reproduce.set_defaults(axis=None)
    for command in (report, reproduce):
        command.add_argument("--out", default="docs", help="folder for the charts and tables")
    say = add("say", command_say, "continue a story in the terminal")
    say.add_argument("story")
    say.add_argument("--knob", action="append", default=[], metavar="AXIS=DOSE", help="for example mood=0.16")
    say.add_argument("--tokens", type=int, default=80)
    say.add_argument("--seed", type=int, default=0)
    serve_command = add("serve", command_serve, "open the web app")
    serve_command.add_argument("--port", type=int, default=8765)
    serve_command.add_argument("--open", action="store_true", help="open the browser")

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
