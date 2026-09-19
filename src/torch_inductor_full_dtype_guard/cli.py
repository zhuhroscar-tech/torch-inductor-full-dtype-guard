"""Command-line interface: run the from-scratch diagnosis of the
torch.compile(backend="inductor") torch.full symbolic-fill dtype-cast
divergence against the currently installed torch build, using the
shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-inductor-full-dtype-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.compile(backend='inductor') path silently drops the "
            "dtype cast for torch.full(size, fill_value, dtype=X) when "
            "fill_value is a symbolic/traced scalar (pytorch/pytorch#194062) "
            "-- causing a wrong bool-mask sum and a silently-skipped int8 "
            "overflow check -- and verify the safe_full() guard function "
            "matches eager under compilation. Never trusts a cached or "
            "previously-reported result, always re-runs the repro on "
            "THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-inductor-full-dtype-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_bool_fill_divergence"]:
        print(status_headline(style, "warn", "torch.compile(inductor) bool-fill dtype-cast divergence reproduced on this host"))
    else:
        print(status_headline(style, "info", "no bool-fill divergence reproduced on this host's installed torch build"))

    if report["any_int8_silent_overflow"]:
        print(status_headline(style, "fail", "int8 overflow check silently skipped under torch.compile (eager raises, compiled does not)"))
    else:
        print(status_headline(style, "info", "no silent int8-overflow case reproduced on this host"))

    if report.get("any_overflow_dtype_silent_overflow"):
        print(status_headline(style, "fail", "at least one overflow case (int8, or int16/uint8 under this process's numpy state) silently skipped its check under torch.compile"))
    else:
        print(status_headline(style, "info", "no silent overflow case reproduced across the checked integer dtypes on this host/process"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_full() matches eager on every case, including under torch.compile"))
    else:
        print(status_headline(style, "fail", "guard did NOT match eager on at least one case"))

    section("bool-fill cases (fill value -> eager vs compiled(native) vs compiled(guarded))")
    for c in report["bool_fill_cases"]:
        flag = "DIVERGES" if c["native_diverges"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"fill={c['fill_value']}",
                    f"eager={c['eager_result']}  compiled_native={c['compiled_native_result']}  "
                    f"compiled_guarded={c['compiled_guarded_result']}  {flag:9s}  {guard_flag}",
                )
            ]
        )

    section("int8-overflow cases (fill value -> did each path raise the overflow error?)")
    for c in report["int8_overflow_cases"]:
        flag = "SILENTLY-WRONG" if c["native_silently_wrong"] else "ok"
        guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"fill={c['fill_value']}",
                    f"eager_raised={c['eager_raised']!s:5s}  compiled_native_raised={c['compiled_native_raised']!s:5s}  "
                    f"compiled_guarded_raised={c['compiled_guarded_raised']!s:5s}  {flag:15s}  {guard_flag}",
                )
            ]
        )

    extra_cases = report.get("extra_overflow_dtype_cases") or []
    if extra_cases:
        section("additional narrow-integer-dtype overflow cases (int16/uint8; bug presence here is numpy-process-state dependent, see README)")
        for c in extra_cases:
            flag = "SILENTLY-WRONG" if c["native_silently_wrong"] else "ok"
            guard_flag = "guard-ok" if c["guard_matches_eager"] else "GUARD-FAILED"
            print_fields(
                [
                    (
                        f"dtype={c.get('overflow_dtype', '?')} fill={c['fill_value']}",
                        f"eager_raised={c['eager_raised']!s:5s}  compiled_native_raised={c['compiled_native_raised']!s:5s}  "
                        f"compiled_guarded_raised={c['compiled_guarded_raised']!s:5s}  {flag:15s}  {guard_flag}",
                    )
                ]
            )


    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
