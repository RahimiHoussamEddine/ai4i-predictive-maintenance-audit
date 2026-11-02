"""Build aggregate reports from the pinned UCI source."""

from __future__ import annotations

import json

from maintenance_audit.constants import RAW_DIR
from maintenance_audit.data import (
    data_quality_report,
    parse_archive,
    write_source_manifest,
)
from maintenance_audit.evaluation import build_report


def main() -> None:
    archive = RAW_DIR / "ai4i2020.zip"
    frame = parse_archive(archive)
    quality = data_quality_report(frame, archive)
    manifest = write_source_manifest(archive, frame)
    summary = build_report(frame, quality)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "archive_sha256": manifest["archive_sha256"],
                "metric_rows": len(summary["metrics"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
