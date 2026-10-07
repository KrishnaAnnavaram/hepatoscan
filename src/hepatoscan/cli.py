"""Command line interface: ``hepatoscan <command> ...``."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import synthetic
from .config import load_dotenv, resolve_device, settings_from_env
from .splits import split_cases
from .volume import load_file


def load_segmenter(path: str | Path, device: str = "cpu"):
    path = Path(path)
    if path.suffix == ".pkl":
        from .baseline import VoxelBaseline

        return VoxelBaseline.load(path)
    if path.suffix == ".pt":
        try:
            from .models.train_unet import load_checkpoint
        except ImportError:
            sys.exit("a .pt model needs the torch extra: pip install -e \".[torch]\"")
        return load_checkpoint(path, device)
    sys.exit(f"unknown model file type: {path.name} (use .pkl or .pt)")


def _split(data_dir: Path, a):
    vols = synthetic.load_dataset(data_dir)
    return vols, split_cases(list(vols), a.val_frac, a.test_frac, a.split_seed)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def _print_eval(report: dict) -> None:
    print(f"model {report['model']}  volumes {report['volumes']}")
    for key, s in report["summary"].items():
        print(f"  {key:28s} {s['mean']:.4f}  [{s['ci_low']:.4f}, {s['ci_high']:.4f}]  n={s['n_finite']}/{s['n']}")


def cmd_synth(a) -> None:
    s = settings_from_env()
    out = Path(a.out or s.data_dir)
    print(f"wrote {synthetic.write_dataset(out, a.cases, a.seed, tuple(a.shape))}")


def cmd_train_baseline(a) -> None:
    from .baseline import VoxelBaseline

    s = settings_from_env()
    vols, sp = _split(Path(a.data or s.data_dir), a)
    model = VoxelBaseline(seed=a.seed, per_class=a.per_class).fit([vols[c] for c in sp.train])
    out = Path(a.out or s.results_dir / "baseline.pkl")
    model.save(out)
    _write_json({"train": sp.train, "val": sp.val, "test": sp.test}, out.with_name("split.json"))
    print(f"saved {out}")


def cmd_train_unet(a) -> None:
    try:
        from .models.train_unet import TrainConfig, save_checkpoint, train
        from .models.unet import UNetConfig
    except ImportError:
        sys.exit("train-unet needs the torch extra: pip install -e \".[torch]\"")
    s = settings_from_env()
    vols, sp = _split(Path(a.data or s.data_dir), a)
    device = resolve_device(a.device or s.device)
    seg, hist = train([vols[c] for c in sp.train], [vols[c] for c in sp.val],
                      UNetConfig(base=a.base, attention_gates=not a.no_attention),
                      TrainConfig(seed=a.seed, epochs=a.epochs, batch_size=a.batch_size), device=device)
    out = Path(a.out or s.results_dir / "unet.pt")
    save_checkpoint(seg, out)
    _write_json({"history": hist, "split": {"train": sp.train, "val": sp.val, "test": sp.test}}, out.with_suffix(".json"))


def cmd_evaluate(a) -> None:
    from .evaluate import evaluate

    s = settings_from_env()
    vols, sp = _split(Path(a.data or s.data_dir), a)
    seg = load_segmenter(a.model, resolve_device(s.device))
    report = evaluate(seg, [vols[c] for c in getattr(sp, a.split)])
    _print_eval(report)
    _write_json(report, Path(s.results_dir) / f"eval_{Path(a.model).stem}_{a.split}.json")


def cmd_segment(a) -> None:
    from .overlay import overlay_rgb
    from .segmenter import segment

    s = settings_from_env()
    seg = load_segmenter(a.model or s.model_path or s.results_dir / "baseline.pkl", resolve_device(s.device))
    vol = load_file(a.file)
    mask, summary = segment(seg, vol)
    out = Path(a.out_dir or s.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{vol.case_id}_mask.npz", mask=mask, spacing=np.asarray(vol.spacing))
    _write_json(summary.as_dict(), out / f"{vol.case_id}_summary.json")
    try:
        from .overlay import to_png

        (out / f"{vol.case_id}_overlay.png").write_bytes(to_png(overlay_rgb(vol.image, mask)))
    except ImportError:
        print("pillow is not installed: no overlay PNG (extra 'image')")
    print(summary.as_text())


def _assistant(s):
    from .assistant.chat import Assistant
    from .assistant.llm import make_llm

    llm = make_llm(s.llm_provider, s.llm_base_url, s.llm_model, s.llm_api_key, s.llm_timeout_s)
    return Assistant(llm, max_exchanges=s.max_history, ttl_s=s.session_ttl_s)


def cmd_ask(a) -> None:
    s = settings_from_env()
    summary_text = None
    if a.summary:
        d = json.loads(Path(a.summary).read_text(encoding="utf-8"))
        summary_text = (f"liver region {d['liver_volume_ml']} mL, lesion candidates {d['lesion_count']}, "
                        f"lesion volume {d['tumor_volume_ml']} mL")
    reply = _assistant(s).ask(a.question, summary_text=summary_text)
    print(reply.answer)


def cmd_serve(a) -> None:
    try:
        import uvicorn

        from .service.api import create_app
    except ImportError:
        sys.exit("serve needs the api extra: pip install -e \".[api]\"")
    from .service.audit import AuditLog
    from .service.handlers import Service

    s = settings_from_env()
    if not s.api_token:
        print("warning: HEPATOSCAN_API_TOKEN is not set, so /segment and /chat answer 503")
    seg = load_segmenter(a.model or s.model_path or s.results_dir / "baseline.pkl", resolve_device(s.device))
    service = Service(seg, _assistant(s), AuditLog(s.audit_log), s.api_token, int(s.max_upload_mb * 1024 * 1024))
    uvicorn.run(create_app(service), host=a.host, port=a.port, log_level="info")


def cmd_demo(a) -> None:
    """Offline demo: phantoms, baseline training, evaluation, one segmentation and two assistant questions."""
    from .baseline import VoxelBaseline
    from .evaluate import evaluate
    from .segmenter import segment

    work = Path(a.out_dir) if a.out_dir else Path(tempfile.mkdtemp(prefix="hepatoscan_demo_"))
    synthetic.write_dataset(work / "data", a.cases, a.seed)
    vols = synthetic.load_dataset(work / "data")
    sp = split_cases(list(vols), 0.15, 0.25, a.seed)
    print(f"synthetic phantoms: train {len(sp.train)}, val {len(sp.val)}, test {len(sp.test)} volumes")
    model = VoxelBaseline(seed=a.seed).fit([vols[c] for c in sp.train])
    report = evaluate(model, [vols[c] for c in sp.test])
    _print_eval(report)
    _, summary = segment(model, vols[sp.test[0]])
    print(summary.as_text())
    from .assistant.chat import Assistant

    bot = Assistant()
    for q in ("What does the lesion count mean?", "Is my tumor cancer?"):
        r = bot.ask(q, summary_text=summary.as_text())
        print(f"\nQ: {q}\nA: {r.answer}")
    print("\nsynthetic phantoms only: these numbers show that the pipeline runs, not clinical performance")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hepatoscan", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def split_args(sp):
        sp.add_argument("--data", help="dataset folder with manifest.csv (default HEPATOSCAN_DATA_DIR)")
        sp.add_argument("--val-frac", type=float, default=0.15)
        sp.add_argument("--test-frac", type=float, default=0.25)
        sp.add_argument("--split-seed", type=int, default=0)

    s = sub.add_parser("synth", help="write synthetic CT phantoms with masks")
    s.add_argument("--cases", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--shape", type=int, nargs=3, default=[24, 64, 64], metavar=("D", "H", "W"))
    s.add_argument("--out")
    s.set_defaults(func=cmd_synth)

    tb = sub.add_parser("train-baseline", help="fit the voxel gradient-boosting baseline")
    split_args(tb)
    tb.add_argument("--seed", type=int, default=0)
    tb.add_argument("--per-class", type=int, default=3000)
    tb.add_argument("--out")
    tb.set_defaults(func=cmd_train_baseline)

    tu = sub.add_parser("train-unet", help="train the 2.5-D attention U-Net (torch extra)")
    split_args(tu)
    tu.add_argument("--seed", type=int, default=0)
    tu.add_argument("--epochs", type=int, default=10)
    tu.add_argument("--batch-size", type=int, default=8)
    tu.add_argument("--base", type=int, default=16)
    tu.add_argument("--no-attention", action="store_true")
    tu.add_argument("--device")
    tu.add_argument("--out")
    tu.set_defaults(func=cmd_train_unet)

    ev = sub.add_parser("evaluate", help="per-volume Dice, HD95 and lesion detection with bootstrap intervals")
    split_args(ev)
    ev.add_argument("--model", required=True)
    ev.add_argument("--split", choices=["val", "test"], default="test")
    ev.set_defaults(func=cmd_evaluate)

    sg = sub.add_parser("segment", help="segment one .npz, .nii or .nii.gz volume")
    sg.add_argument("file")
    sg.add_argument("--model")
    sg.add_argument("--out-dir")
    sg.set_defaults(func=cmd_segment)

    ak = sub.add_parser("ask", help="ask the assistant one question")
    ak.add_argument("question")
    ak.add_argument("--summary", help="a *_summary.json file from `segment`")
    ak.set_defaults(func=cmd_ask)

    sv = sub.add_parser("serve", help="run the HTTP API (api extra)")
    sv.add_argument("--model")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    d = sub.add_parser("demo", help="offline demo on synthetic phantoms")
    d.add_argument("--cases", type=int, default=30)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out-dir")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
