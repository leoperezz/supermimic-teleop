"""Print comparable per-clip evaluation summaries as CSV (no model updates)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("reports", nargs="+", type=Path)
  args = parser.parse_args()
  reports = [json.loads(p.read_text()) for p in args.reports]
  trials = [{(r["clip"], r["trial"]) for r in report["per_clip"]} for report in reports]
  if any(t != trials[0] for t in trials):
    parser.error("Reports must evaluate the same clips and trial counts")
  for key in [
    "seed",
    "steps",
    "delay_ms",
    "jitter_ms",
    "position_noise",
    "dropout",
    "max_dropout_frames",
    "max_age_ms",
    "outage_at",
    "outage_seconds",
  ]:
    if len({r["settings"][key] for r in reports}) != 1:
      parser.error(
        f"Reports differ in {key}; compare policies within the same sensor scenario"
      )
  writer = csv.writer(sys.stdout)
  writer.writerow(
    [
      "report",
      "branch",
      "trials",
      "completed",
      "failure",
      "sensor_timeout",
      "censored",
      "completion_rate",
      "mean_seconds",
      "body_error_m",
      "wrist_error_m",
    ]
  )
  for path, report in zip(args.reports, reports, strict=True):
    outcomes = report["outcomes"]

    def metric(name, report=report):
      values = [r["metrics"][name] for r in report["per_clip"] if name in r["metrics"]]
      return round(float(np.mean(values)), 5) if values else ""

    writer.writerow(
      [
        str(path),
        report["branch"],
        report["trials"],
        outcomes.get("completed", 0),
        outcomes.get("failure", 0),
        outcomes.get("sensor_timeout", 0),
        outcomes.get("step_limit", 0) + outcomes.get("time_limit", 0),
        round(report["completion_rate"], 4),
        round(report["mean_seconds"], 3),
        metric("body_position_error_m"),
        metric("wrist_position_error_m"),
      ]
    )


if __name__ == "__main__":
  main()
