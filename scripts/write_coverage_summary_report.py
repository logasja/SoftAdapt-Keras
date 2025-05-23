import json
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

PRECISION = Decimal(".01")


def main():
    project_root = Path(__file__).resolve().parent.parent
    coverage_summary = project_root / "coverage-summary.json"

    coverage_data = json.loads(coverage_summary.read_text(encoding="utf-8"))
    total_data = coverage_data.pop("total")

    lines = [
        "\n",
        "Package | Statements\n",
        "--- | ---\n",
    ]

    for package, data in sorted(coverage_data.items()):
        statements_covered = data["statements_covered"]
        statements = data["statements"]

        rate = Decimal(statements_covered) / Decimal(statements) * 100
        rate = rate.quantize(PRECISION, rounding=ROUND_DOWN)
        lines.append(
            f"{package} | {100 if rate == 100 else rate}% ({statements_covered} / {statements})\n"  # noqa: PLR2004
        )

    total_statements_covered = total_data["statements_covered"]
    total_statements = total_data["statements"]
    total_rate = Decimal(total_statements_covered) / Decimal(total_statements) * 100
    total_rate = total_rate.quantize(PRECISION, rounding=ROUND_DOWN)
    color = "ok" if float(total_rate) >= 95 else "critical"  # noqa: PLR2004
    lines.insert(
        0,
        f"![Code Coverage](https://img.shields.io/badge/coverage-{total_rate}%25-{color}?style=flat)\n",
    )

    lines.append(
        f"**Summary** | {100 if total_rate == 100 else total_rate}% "  # noqa: PLR2004
        f"({total_statements_covered} / {total_statements})\n"
    )

    coverage_report = project_root / "coverage-report.md"
    with coverage_report.open("w", encoding="utf-8") as f:
        f.write("".join(lines))


if __name__ == "__main__":
    main()
