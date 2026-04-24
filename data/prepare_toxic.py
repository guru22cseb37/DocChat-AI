"""Prepare Jigsaw toxic comments as instruction-format JSONL data."""

import csv
import json
from pathlib import Path

from langdetect import LangDetectException, detect
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()
TOXICITY_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEVERITY_ORDER = ["severe_toxic", "threat", "identity_hate", "insult", "obscene", "toxic"]


def is_english(text: str) -> bool:
    """Return True when language detection classifies text as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def parse_binary(value: str) -> int:
    """Convert potentially noisy binary values to 0 or 1 safely."""
    try:
        return 1 if int(float(str(value).strip())) > 0 else 0
    except (TypeError, ValueError):
        return 0


def determine_label(row: dict[str, str]) -> tuple[str, str]:
    """Determine output label; safe if all zero, otherwise most severe active label."""
    active_labels = [col for col in TOXICITY_COLUMNS if parse_binary(row.get(col, "0")) == 1]
    if not active_labels:
        return "safe", "No toxic content detected."

    for label in SEVERITY_ORDER:
        if label in active_labels:
            return label, f"The comment contains {label.replace('_', ' ')} language."
    return active_labels[0], f"The comment contains {active_labels[0].replace('_', ' ')} language."


def format_instruction(comment: str, label: str, explanation: str) -> dict[str, str]:
    """Format one row into required instruction text payload."""
    return {"text": f"### Comment: {comment}\n### Output: {label} — {explanation}"}


def process_row(row: dict[str, str]) -> dict[str, str] | None:
    """Validate and convert a CSV row, returning None for invalid rows."""
    comment = str(row.get("comment_text", "")).strip()
    if not comment or comment.lower() == "nan":
        return None
    if not is_english(comment):
        return None
    label, explanation = determine_label(row)
    return format_instruction(comment, label, explanation)


def read_rows(input_file: Path) -> list[dict[str, str]]:
    """Read all CSV rows with explicit FileNotFoundError handling."""
    try:
        with input_file.open("r", encoding="utf-8", errors="ignore", newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{input_file} not found. Download Jigsaw train.csv and place it in data/."
        ) from exc


def ensure_minimum_rows(processed: list[dict[str, str]]) -> list[dict[str, str]]:
    """Enforce minimum sample counts according to requirements."""
    if len(processed) < 50:
        raise ValueError(
            f"Not enough samples after filtering: {len(processed)}. At least 50 are required."
        )
    if len(processed) < 100:
        repeats = (100 // len(processed)) + 1
        expanded = (processed * repeats)[:100]
        console.print(
            f"[yellow]Only {len(processed)} valid rows found; upsampled to {len(expanded)} rows.[/yellow]"
        )
        return expanded
    return processed


def save_jsonl(output_file: Path, rows: list[dict[str, str]]) -> None:
    """Save processed rows to JSONL with robust I/O handling."""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as out_file:
            for row in rows:
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    except FileNotFoundError as exc:
        console.print(f"[red]Output path not found: {exc}[/red]")
        raise
    except OSError as exc:
        console.print(f"[red]Failed writing JSONL file: {exc}[/red]")
        raise


def main() -> None:
    """Load, filter, convert, and save toxic instruction training data."""
    console.print("[bold cyan]DocChat AI — Toxic Data Preparation[/bold cyan]")
    input_file = Path("data/train.csv")
    output_file = Path("data/toxic_instructions.jsonl")

    try:
        rows = read_rows(input_file)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return

    processed: list[dict[str, str]] = []
    skipped_corrupt = 0
    skipped_empty = 0
    skipped_non_english = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[magenta]Processing comments...[/magenta]", total=len(rows))
        for row in rows:
            try:
                result = process_row(row)
                if result is None:
                    comment = str(row.get("comment_text", "")).strip()
                    if not comment or comment.lower() == "nan":
                        skipped_empty += 1
                    elif not is_english(comment):
                        skipped_non_english += 1
                else:
                    processed.append(result)
            except Exception:
                skipped_corrupt += 1
            finally:
                progress.advance(task)

    try:
        final_rows = ensure_minimum_rows(processed)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise

    save_jsonl(output_file, final_rows)
    console.print(
        "[green]Saved toxic instructions successfully.[/green] "
        f"kept={len(final_rows)} skipped_empty={skipped_empty} "
        f"skipped_non_english={skipped_non_english} skipped_corrupt={skipped_corrupt}"
    )


if __name__ == "__main__":
    main()
