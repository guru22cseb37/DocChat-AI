"""Evaluate DocChat AI on held-out toxicity and Q&A datasets via running API."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import track
from sklearn.metrics import f1_score, confusion_matrix
from rouge_score import rouge_scorer
import evaluate

console = Console()

SERVER_URL = "http://localhost:8000"
API_KEY = "docchat-secret-key"

def print_banner() -> None:
    """Print the evaluation engine banner."""
    banner = """
[bold cyan]
██████╗  ██████╗  ██████╗ ██████╗██╗  ██╗ █████╗ ████████╗    █████╗ ██╗
██╔══██╗██╔═══██╗██╔════╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝   ██╔══██╗██║
██║  ██║██║   ██║██║     ██║     ███████║███████║   ██║█████╗███████║██║
██║  ██║██║   ██║██║     ██║     ██╔══██║██╔══██║   ██║╚════╝██╔══██║██║
██████╔╝╚██████╔╝╚██████╗╚██████╗██║  ██║██║  ██║   ██║      ██║  ██║██║
╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚═╝  ╚═╝╚═╝
[/bold cyan]
[bold white]DocChat AI — Evaluation Engine[/bold white]
    """
    console.print(banner)

def check_server() -> bool:
    """Checks if the API server is online."""
    try:
        response = httpx.get(f"{SERVER_URL}/health", timeout=30.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False

def evaluate_toxicity() -> float:
    """Evaluates toxicity prediction on the test set."""
    test_file = Path("data/test_toxic.jsonl")
    if not test_file.exists():
        console.print("[yellow]No toxicity test set found. Skipping toxicity evaluation.[/yellow]")
        return 0.0

    samples = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    except FileNotFoundError:
        console.print("[yellow]Toxicity test file missing at read time.[/yellow]")
        return 0.0

    if not samples:
        return 0.0

    y_true = []
    y_pred = []

    console.print("[cyan]Evaluating Toxicity Model...[/cyan]")
    
    headers = {"X-API-Key": API_KEY}
    with httpx.Client(timeout=30.0) as client:
        for sample in track(samples, description="Predicting Toxicity..."):
            text = sample.get("text", "")
            # Assuming sample text format: ### Comment: [text]\n### Output: [label]
            if "### Comment:" not in text:
                continue
                
            parts = text.split("### Output:")
            if len(parts) != 2:
                continue
                
            comment_part = parts[0].replace("### Comment:", "").strip()
            true_label_part = parts[1].split("—")[0].strip()
            
            try:
                resp = client.post(
                    f"{SERVER_URL}/predict",
                    json={"comment": comment_part},
                    headers=headers
                )
                if resp.status_code == 200:
                    pred_label = resp.json().get("label", "safe")
                    y_true.append(true_label_part)
                    y_pred.append(pred_label)
            except Exception:
                pass

    if not y_true:
        return 0.0

    f1 = f1_score(y_true, y_pred, average="weighted")
    
    # Print Confusion Matrix
    labels = list(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    table = Table(title="Toxicity Confusion Matrix")
    table.add_column("True \\ Pred", style="cyan")
    for label in labels:
        table.add_column(label)
        
    for i, true_label in enumerate(labels):
        row = [true_label] + [str(x) for x in cm[i]]
        table.add_row(*row)
        
    console.print(table)
    return f1

def evaluate_qa() -> Tuple[float, float]:
    """Evaluates Q&A performance using ROUGE-L and BERTScore."""
    test_file = Path("data/test_qa.jsonl")
    if not test_file.exists():
        console.print("[yellow]No Q&A test set found. Skipping Q&A evaluation.[/yellow]")
        return 0.0, 0.0

    samples = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    except FileNotFoundError:
        console.print("[yellow]Q&A test file missing at read time.[/yellow]")
        return 0.0, 0.0

    if not samples:
        return 0.0, 0.0

    predictions = []
    references = []

    console.print("[cyan]Evaluating Q&A Model...[/cyan]")
    
    headers = {"X-API-Key": API_KEY}
    with httpx.Client(timeout=30.0) as client:
        for sample in track(samples, description="Generating Answers..."):
            try:
                resp = client.post(
                    f"{SERVER_URL}/ask",
                    json={"doc": sample["doc"], "question": sample["q"]},
                    headers=headers
                )
                if resp.status_code == 200:
                    predictions.append(resp.json().get("answer", ""))
                    references.append(sample["a"])
            except Exception:
                pass

    if not predictions:
        return 0.0, 0.0

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
    avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores)

    # BERTScore
    try:
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(predictions=predictions, references=references, lang="en")
        avg_bert = sum(results["f1"]) / len(results["f1"])
    except Exception as e:
        console.print(f"[yellow]Failed to compute BERTScore: {e}[/yellow]")
        avg_bert = 0.0

    return avg_rouge, avg_bert

def main() -> None:
    """Run full evaluation workflow and print final metrics table."""
    print_banner()
    
    if not check_server():
        console.print("[bold red]Error: The FastAPI server is not running or unreachable.[/bold red]")
        console.print("Please start the server first: [white]uv run uvicorn api:app --port 8000[/white]")
        return
        
    f1_tox = evaluate_toxicity()
    rouge_l, bert_f1 = evaluate_qa()
    
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Metric", style="cyan", justify="left")
    results_table.add_column("Score", style="magenta", justify="right")
    
    results_table.add_row("Toxicity F1 (weighted)", f"{f1_tox:.4f}")
    results_table.add_row("Q&A ROUGE-L", f"{rouge_l:.4f}")
    results_table.add_row("Q&A BERTScore (F1)", f"{bert_f1:.4f}")
    
    console.print(results_table)

if __name__ == "__main__":
    main()
