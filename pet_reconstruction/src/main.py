"""Top-level dispatcher for the PET reconstruction MVP.

Examples (run from the pet_reconstruction/ directory):
    # 1) Preprocess every volume (one-time, ~hours)
    python -m src.main preprocess

    # 1b) Smoke variant: 50 patients, 128x128 resolution
    python -m src.main preprocess --smoke --limit 50

    # 2) Train Pipeline A
    python -m src.main train --pipeline supervised

    # 2b) Train Pipeline B
    python -m src.main train --pipeline unconditional

    # 3) Reconstruct test volumes with a trained checkpoint
    python -m src.main reconstruct --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029

    # 4) Evaluate a checkpoint on the held-out test split (metrics + figures)
    python -m src.main evaluate --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
        --output-dir evaluations/supervised/

    # 5) Preview ONE test slice — low / recon / full side by side
    python -m src.main preview --pipeline supervised --list
    python -m src.main preview --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
        --patient-id <test_pid> --slice-idx 120

The flags --smoke for preprocess and train shrink the run to a ~1-hour
end-to-end validation of the pipeline.
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="PET reconstruction MVP dispatcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pre = subparsers.add_parser("preprocess", help="Build the cached slice dataset.")
    p_pre.add_argument("--limit", type=int, default=None)
    p_pre.add_argument("--smoke", action="store_true")

    p_train = subparsers.add_parser("train", help="Train one of the pipelines.")
    p_train.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )
    p_train.add_argument("--smoke", action="store_true")
    p_train.add_argument(
        "--resume-from",
        default=None,
        help='Resume training. Pass "latest" to pick the newest checkpoint-epoch-* under '
        "the pipeline's output dir, or the name of a specific checkpoint subdir.",
    )

    p_recon = subparsers.add_parser(
        "reconstruct", help="Reconstruct test volumes from a checkpoint."
    )
    p_recon.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )

    p_eval = subparsers.add_parser(
        "evaluate", help="Run inference + metrics on a held-out split."
    )
    p_eval.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )

    p_preview = subparsers.add_parser(
        "preview",
        help="Render a single-slice low / recon / full figure from a test patient.",
    )
    p_preview.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )

    p_slice = subparsers.add_parser(
        "slice-recon",
        help="Export clean full / low / recon PNGs for one slice (memoria figures).",
    )
    p_slice.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )

    p_mip = subparsers.add_parser(
        "mip-recon",
        help="Export full / low / recon MIP PNGs over one plane (memoria figures).",
    )
    p_mip.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )

    # Subcommands that wrap a per-pipeline script pass their remaining flags straight
    # through. parse_known_args() collects those into `extra` (argparse.REMAINDER does
    # NOT capture leading --options, so it cannot be used here).
    args, extra = parser.parse_known_args()
    if args.command in ("preprocess", "train") and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.command == "preprocess":
        from .config import DataConfig
        from .preprocess import preprocess_all

        cfg = DataConfig()
        if args.smoke:
            cfg.image_size = 128
        preprocess_all(cfg, limit=args.limit)
        return

    if args.command == "train":
        if args.pipeline == "supervised":
            from .config import SupervisedConfig, apply_smoke_overrides
            from .train_supervised import train as train_fn

            cfg = SupervisedConfig()
        elif args.pipeline == "regression":
            from .config import RegressionUNetConfig, apply_smoke_overrides
            from .train_regression import train as train_fn

            cfg = RegressionUNetConfig()
        elif args.pipeline == "cnn":
            from .config import CNNConfig, apply_smoke_overrides
            from .train_cnn import train as train_fn

            cfg = CNNConfig()
        else:
            from .config import UnconditionalConfig, apply_smoke_overrides
            from .train_unconditional import train as train_fn

            cfg = UnconditionalConfig()
        if args.smoke:
            apply_smoke_overrides(cfg)
        train_fn(cfg, resume_from=args.resume_from)
        return

    if args.command == "reconstruct":
        # Re-route to the per-pipeline reconstruct main(). We rewrite sys.argv so the
        # nested argparser sees just its own flags.
        if args.pipeline == "supervised":
            from .reconstruct_supervised import main as recon_main
        elif args.pipeline == "regression":
            from .reconstruct_regression import main as recon_main
        elif args.pipeline == "cnn":
            from .reconstruct_cnn import main as recon_main
        else:
            from .reconstruct_unconditional import main as recon_main
        sys.argv = [sys.argv[0], *extra]
        recon_main()
        return

    if args.command == "evaluate":
        from .evaluate import main as eval_main

        # The pipeline flag is consumed here AND re-injected for the inner parser,
        # so the user can write the same flag set as if calling evaluate.py directly.
        passthrough = ["--pipeline", args.pipeline, *extra]
        sys.argv = [sys.argv[0], *passthrough]
        eval_main()
        return

    if args.command == "preview":
        from .preview_reconstruction import main as preview_main

        passthrough = ["--pipeline", args.pipeline, *extra]
        sys.argv = [sys.argv[0], *passthrough]
        preview_main()
        return

    if args.command == "slice-recon":
        from .slice_recon import main as slice_main

        passthrough = ["--pipeline", args.pipeline, *extra]
        sys.argv = [sys.argv[0], *passthrough]
        slice_main()
        return

    if args.command == "mip-recon":
        from .mip_recon import main as mip_main

        passthrough = ["--pipeline", args.pipeline, *extra]
        sys.argv = [sys.argv[0], *passthrough]
        mip_main()
        return


if __name__ == "__main__":
    main()
