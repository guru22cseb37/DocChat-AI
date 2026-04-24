"""Generate local TinyLlama Q&A pairs from plain-text documents."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from langdetect import LangDetectException, detect
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for input source and output location."""
    parser = argparse.ArgumentParser(description="Generate Q&A pairs with TinyLlama.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/documents",
        help="Input .txt file or directory of .txt files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qa_pairs.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def is_english(text: str) -> bool:
    """Return True if text is detected as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_documents(input_path: Path) -> list[str]:
    """Load document strings from file or directory while handling missing files."""
    documents: list[str] = []
    try:
        if input_path.is_file() and input_path.suffix.lower() == ".txt":
            content = input_path.read_text(encoding="utf-8", errors="ignore")
            documents.append(content)
        elif input_path.is_dir():
            for file_path in sorted(input_path.glob("*.txt")):
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                documents.append(content)
        else:
            console.print(f"[red]Input path not found or invalid: {input_path}[/red]")
    except FileNotFoundError:
        console.print(f"[red]Input path not found: {input_path}[/red]")
    except OSError as exc:
        console.print(f"[red]Failed reading documents: {exc}[/red]")
    return documents


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokenizer tokens for a text sample."""
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return 0


def parse_generated_qa(output_text: str) -> list[dict[str, str]]:
    """Parse generated text into normalized question/answer pairs."""
    pairs: list[dict[str, str]] = []
    question = ""
    answer = ""

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith(("q:", "question:")):
            if question and answer:
                pairs.append({"q": question.strip(), "a": answer.strip()})
            question = line.split(":", 1)[1].strip()
            answer = ""
        elif line.lower().startswith(("a:", "answer:")):
            answer = line.split(":", 1)[1].strip()
        elif answer:
            answer = f"{answer} {line}".strip()

    if question and answer:
        pairs.append({"q": question.strip(), "a": answer.strip()})
    return [item for item in pairs if item["q"] and item["a"]]


def generate_pairs_for_document(doc: str, generator: Any, tokenizer: Any) -> list[dict[str, str]]:
    """Generate 5-10 Q&A pairs for one valid document."""
    if not doc.strip():
        return []
    if not is_english(doc):
        return []
    if count_tokens(doc, tokenizer) < 50:
        return []

    prompt = (
        "Generate between 5 and 10 factual question-answer pairs for the document.\n"
        "Output strictly as lines with `Q:` and `A:` prefixes.\n\n"
        f"Document:\n{doc}\n\nQ:"
    )

    try:
        result = generator(
            prompt,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )
        generated = result[0]["generated_text"]
        pairs = parse_generated_qa(generated)
        return pairs[:10] if len(pairs) >= 5 else []
    except Exception as exc:
        console.print(f"[red]Generation failed for one document: {exc}[/red]")
        return []


def save_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    """Save records to JSONL, handling path and I/O errors safely."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_obj:
            for row in records:
                file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
        console.print(f"[green]Saved {len(records)} rows to {output_path}[/green]")
    except FileNotFoundError:
        console.print(f"[red]Output path missing: {output_path}[/red]")
    except OSError as exc:
        console.print(f"[red]Failed to write output JSONL: {exc}[/red]")


def build_generator() -> tuple[Any, Any]:
    """Load TinyLlama and return text-generation pipeline with tokenizer."""
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Loading {model_id} on {device}...[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    if device == "cuda":
        model = model.to("cuda")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    else:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return pipe, tokenizer


def main() -> None:
    """Run end-to-end Q&A generation and save to `data/qa_pairs.jsonl`."""
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    console.print("[bold cyan]DocChat AI — Q&A Generator[/bold cyan]")

    documents = load_documents(input_path)
    if not documents:
        console.print("[yellow]No input documents found. Nothing to process.[/yellow]")
        save_jsonl([], output_path)
        return

    try:
        generator, tokenizer = build_generator()
    except Exception as exc:
        console.print(f"[red]Model loading failed: {exc}[/red]")
        return

    records: list[dict[str, str]] = []
    skipped_empty = 0
    skipped_lang = 0
    skipped_short = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[blue]Generating Q&A pairs...[/blue]", total=len(documents))
        for document in documents:
            if not document.strip():
                skipped_empty += 1
                progress.advance(task)
                continue
            if not is_english(document):
                skipped_lang += 1
                progress.advance(task)
                continue
            if count_tokens(document, tokenizer) < 50:
                skipped_short += 1
                progress.advance(task)
                continue

            for pair in generate_pairs_for_document(document, generator, tokenizer):
                records.append({"doc": document, "q": pair["q"], "a": pair["a"]})
            progress.advance(task)

    console.print(
        f"[bold]Done.[/bold] kept={len(records)} skipped_empty={skipped_empty} "
        f"skipped_non_english={skipped_lang} skipped_short={skipped_short}"
    )
    save_jsonl(records, output_path)


if __name__ == "__main__":
    main()
