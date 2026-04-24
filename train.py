"""
DocChat AI — Fine-Tuning Engine

This module performs QLoRA fine-tuning on the TinyLlama model. It combines the 
Q&A and Toxicity datasets, applies 4-bit quantization, and uses the PEFT library 
to train an adapter. It handles GPU/CPU detection, OutOfMemory errors, and 
evaluates the model after training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from datasets import Dataset
from rich.console import Console
from rich.live import Live
from rich.table import Table
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

console = Console()


class RichLossCallback(TrainerCallback):
    """Render training loss updates in a live Rich table."""

    def __init__(self) -> None:
        """Initialize callback state."""
        self.current_step = 0
        self.current_loss = 0.0
        self._live: Live | None = None

    def _table(self) -> Table:
        """Build a one-row loss table for live updates."""
        table = Table(title="Live Training Metrics")
        table.add_column("Step", style="cyan")
        table.add_column("Loss", style="magenta")
        table.add_row(str(self.current_step), f"{self.current_loss:.6f}")
        return table

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        """Start live display at beginning of training."""
        self._live = Live(self._table(), refresh_per_second=4, console=console)
        self._live.start()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, Any] | None = None, **kwargs: Any):
        """Update live display when trainer emits logs."""
        if logs and "loss" in logs:
            self.current_step = int(state.global_step)
            self.current_loss = float(logs["loss"])
            if self._live is not None:
                self._live.update(self._table())

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):
        """Stop live display at end of training."""
        if self._live is not None:
            self._live.stop()

def print_banner() -> None:
    """
    Prints the ASCII art banner for the DocChat AI Fine-Tuning Engine.
    """
    banner = """
[bold cyan]
██████╗  ██████╗  ██████╗ ██████╗██╗  ██╗ █████╗ ████████╗    █████╗ ██╗
██╔══██╗██╔═══██╗██╔════╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝   ██╔══██╗██║
██║  ██║██║   ██║██║     ██║     ███████║███████║   ██║█████╗███████║██║
██║  ██║██║   ██║██║     ██║     ██╔══██║██╔══██║   ██║╚════╝██╔══██║██║
██████╔╝╚██████╔╝╚██████╗╚██████╗██║  ██║██║  ██║   ██║      ██║  ██║██║
╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚═╝  ╚═╝╚═╝
[/bold cyan]
[bold white]DocChat AI — Fine-Tuning Engine[/bold white]
    """
    console.print(banner)

def detect_device() -> str:
    """
    Detects the best available device (GPU or CPU) for training.

    Returns:
        str: The device string ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        console.print(f"[bold green]GPU Detected:[/bold green] {device_name}")
        return "cuda"
    else:
        console.print("[bold yellow]WARNING: No GPU detected. Falling back to CPU. Training will be extremely slow.[/bold yellow]")
        return "cpu"

def load_datasets() -> Dataset:
    """
    Loads and merges the Q&A and Toxicity datasets, then shuffles them.

    Returns:
        Dataset: A HuggingFace Dataset object containing the merged data.
    """
    qa_path = Path("data/qa_pairs.jsonl")
    toxic_path = Path("data/toxic_instructions.jsonl")
    
    combined_data: List[Dict[str, str]] = []
    
    qa_count = 0
    if qa_path.exists():
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    text = f"### Document: {item.get('doc', '')}\n### Question: {item.get('q', '')}\n### Answer: {item.get('a', '')}"
                    combined_data.append({"text": text})
                    qa_count += 1
        except FileNotFoundError:
            console.print(f"[yellow]Missing file: {qa_path}[/yellow]")
                
    toxic_count = 0
    if toxic_path.exists():
        try:
            with open(toxic_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    combined_data.append({"text": item.get('text', '')})
                    toxic_count += 1
        except FileNotFoundError:
            console.print(f"[yellow]Missing file: {toxic_path}[/yellow]")
                
    random.shuffle(combined_data)
    
    # Print stats table
    table = Table(title="Dataset Statistics")
    table.add_column("Dataset Type", justify="left", style="cyan")
    table.add_column("Sample Count", justify="right", style="magenta")
    table.add_row("Q&A Pairs", str(qa_count))
    table.add_row("Toxic Instructions", str(toxic_count))
    table.add_row("Total Samples", str(len(combined_data)), style="bold")
    console.print(table)
    
    return Dataset.from_list(combined_data)

def load_model_and_tokenizer(device: str) -> Tuple[Any, Any, str]:
    """
    Loads the TinyLlama model and tokenizer, applying 4-bit quantization if on GPU.

    Args:
        device (str): The device type to load the model on ('cuda' or 'cpu').

    Returns:
        Tuple[Any, Any]: The loaded model and tokenizer.
    """
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if device == "cpu":
        # CPU fallback for stability on low-resource Windows environments.
        model_id = "sshleifer/tiny-gpt2"
        console.print(
            "[yellow]CPU fallback model selected for stability: sshleifer/tiny-gpt2[/yellow]"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        # Fallback for CPU without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
    return model, tokenizer, model_id

def train_model(model: Any, tokenizer: Any, dataset: Dataset, device: str, model_id: str) -> None:
    """
    Applies LoRA to the model and trains it using SFTTrainer with OOM handling.

    Args:
        model (Any): The base language model.
        tokenizer (Any): The tokenizer.
        dataset (Dataset): The training dataset.
        device (str): The device type ('cuda' or 'cpu').
    """
    target_modules = ["q_proj", "v_proj"]
    if "tiny-gpt2" in model_id.lower():
        target_modules = ["c_attn"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # We apply the peft config
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    output_dir = "./checkpoints"
    Path(output_dir).mkdir(exist_ok=True)
    
    has_checkpoints = len(list(Path(output_dir).glob("checkpoint-*"))) > 0
    resume_from_checkpoint = True if has_checkpoints else False
    
    if resume_from_checkpoint:
        console.print("[cyan]Checkpoints found. Resuming from checkpoint.[/cyan]")
        
    batch_size = 2
    
    def get_training_args(bs: int) -> TrainingArguments:
        return TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            learning_rate=2e-4,
            fp16=(device == "cuda"),
            logging_steps=10,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=50,
            report_to="none",
            optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch"
        )
        
    def do_train(bs: int) -> None:
        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=512,
                args=get_training_args(bs),
                callbacks=[RichLossCallback()],
            )
        except TypeError:
            tokenized_dataset = dataset.map(
                lambda sample: tokenizer(
                    sample["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                )
            )
            trainer = SFTTrainer(
                model=model,
                train_dataset=tokenized_dataset,
                args=get_training_args(bs),
                callbacks=[RichLossCallback()],
            )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
    try:
        console.print(f"[magenta]Starting training with batch_size={batch_size}...[/magenta]")
        do_train(batch_size)
    except torch.cuda.OutOfMemoryError:
        console.print("[bold red]CUDA OutOfMemoryError detected![/bold red] Halving batch size and retrying...")
        torch.cuda.empty_cache()
        batch_size = max(1, batch_size // 2)
        try:
            do_train(batch_size)
        except torch.cuda.OutOfMemoryError:
            console.print("[bold red]OOM error again. Training failed.[/bold red]")
            raise
            
    # Save final adapter
    adapter_dir = "./adapter_weights"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    console.print(f"[bold green]Adapter saved to {adapter_dir}[/bold green]")

def verify_model(device: str, model_id: str) -> None:
    """
    Reloads the model with the trained adapter and runs a sample inference.

    Args:
        device (str): The device type to load the model on.
    """
    console.print("[cyan]Verifying saved adapter...[/cyan]")
    adapter_dir = "./adapter_weights"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        
        test_prompt = "### Comment: I love this tool, it is amazing!\n### Output:"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            temperature=0.1, 
            do_sample=False
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        console.print(f"[bold green]PASS:[/bold green] Inference successful.")
        console.print(f"[dim]Output snippet:[/dim] {result[len(test_prompt):].strip()}")
        
    except Exception as e:
        console.print(f"[bold red]FAIL:[/bold red] Verification failed with error: {e}")

def main() -> None:
    """
    Main function to execute the fine-tuning pipeline.
    """
    print_banner()
    
    device = detect_device()
    dataset = load_datasets()
    
    if len(dataset) == 0:
        console.print("[bold red]Error: Dataset is empty. Please run data generation scripts first.[/bold red]")
        return
        
    model, tokenizer, model_id = load_model_and_tokenizer(device)
    train_model(model, tokenizer, dataset, device, model_id)
    verify_model(device, model_id)

if __name__ == "__main__":
    main()
