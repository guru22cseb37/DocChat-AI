"""Serve DocChat AI inference APIs with auth, validation, and robust failover."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from langdetect import LangDetectException, detect
from peft import PeftModel
from pydantic import BaseModel, field_validator
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

console = Console()

# -----------------------------------------------------------------------------
# App State & Settings
# -----------------------------------------------------------------------------

class AppState:
    """In-memory model state shared by request handlers."""

    model: Any = None
    tokenizer: Any = None
    adapter_loaded: bool = False
    device: str = "cpu"

state = AppState()
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_expected_api_key() -> str:
    """Return expected API key from environment with a safe default."""
    return os.environ.get("API_KEY", "docchat-secret-key")

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Validate request API key and raise 401 on mismatch."""
    if api_key_header == get_expected_api_key():
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )

# -----------------------------------------------------------------------------
# Startup / Shutdown
# -----------------------------------------------------------------------------

def print_banner() -> None:
    """Print server startup banner."""
    banner = """
[bold cyan]
========================================
         DOCCHAT AI FASTAPI SERVER
========================================
[/bold cyan]
[bold white]DocChat AI - FastAPI Server[/bold white]
"""
    console.print(banner)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and free resources on shutdown."""
    print_banner()
    console.print("[cyan]Initializing Model Server...[/cyan]")
    
    state.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_dir = "./adapter_weights"
    
    try:
        # Load base model & tokenizer
        state.tokenizer = AutoTokenizer.from_pretrained(model_id)
        state.tokenizer.pad_token = state.tokenizer.eos_token
        
        if state.device == "cuda":
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
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )

        # Attempt to load LoRA adapter — fall back to base model on mismatch
        if os.path.exists(adapter_dir):
            try:
                state.model = PeftModel.from_pretrained(base_model, adapter_dir)
                state.adapter_loaded = True
                console.print(f"[bold green]Model + adapter loaded on {state.device}[/bold green]")
            except Exception as adapter_err:
                console.print(f"[bold yellow]Adapter incompatible ({adapter_err}). Running base model only.[/bold yellow]")
                state.model = base_model
                state.adapter_loaded = False
        else:
            state.model = base_model
            state.adapter_loaded = False
            console.print("[bold yellow]No adapter found. Running with base model only.[/bold yellow]")

    except Exception as e:
        console.print(f"[bold red]Critical Error: Failed to load model. {e}[/bold red]")
        state.model = None
        
    console.print("[bold green]Server Ready![/bold green]")
    console.print("[white]Endpoints:[/white] /health, /predict, /ask, /batch_predict")
    
    yield
    
    # Cleanup
    if state.model is not None:
        del state.model
    if state.tokenizer is not None:
        del state.tokenizer
    if state.device == "cuda":
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan, title="DocChat AI API")

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input payload for toxicity prediction."""

    comment: str

    @field_validator("comment")
    @classmethod
    def comment_must_exist(cls, v: str) -> str:
        """Require string field presence and normalize to raw string."""
        if v is None:
            raise ValueError("comment is required")
        return v


class AskRequest(BaseModel):
    """Input payload for document question answering."""

    doc: str
    question: str

    @field_validator("doc")
    @classmethod
    def doc_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only document text."""
        if not v.strip():
            raise ValueError("Document cannot be empty")
        return v

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only question."""
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v


class BatchPredictRequest(BaseModel):
    """Input payload for batch toxicity prediction."""

    comments: list[str]

    @field_validator("comments")
    @classmethod
    def comments_must_be_non_empty(cls, v: list[str]) -> list[str]:
        """Validate list presence and enforce non-empty comment items."""
        if not isinstance(v, list) or not v:
            raise ValueError("comments must be a non-empty list")
        for item in v:
            if not isinstance(item, str):
                raise ValueError("all comments must be strings")
        return v

# -----------------------------------------------------------------------------
# Inference Helpers
# -----------------------------------------------------------------------------

def is_english(text: str) -> bool:
    """Detect whether a text is English."""
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def _run_inference_sync(prompt: str, max_new_tokens: int, max_input_tokens: int = 2000) -> str:
    """Run synchronous model generation with token truncation."""
    if state.model is None or state.tokenizer is None:
        raise RuntimeError("Model is not loaded.")
        
    inputs = state.tokenizer(prompt, return_tensors="pt")
    
    # Truncate input if necessary
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]
        # recreate attention mask
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        console.print(f"[yellow]Input truncated to {max_input_tokens} tokens.[/yellow]")
        
    inputs = {k: v.to(state.model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Evaluation mode implicit during generate
        outputs = state.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=state.tokenizer.eos_token_id
        )
        
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = state.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return result.strip()

async def generate_response(
    prompt: str,
    max_new_tokens: int = 256,
    max_input_tokens: int = 2000,
    timeout_seconds: float = 120.0,
) -> str:
    """Generate text asynchronously with configurable timeout guard."""
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_inference_sync, prompt, max_new_tokens, max_input_tokens),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        console.print(f"[red]Inference timed out after {timeout_seconds:.0f} seconds.[/red]")
        raise HTTPException(status_code=504, detail="Model inference timed out")
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
def truncate_text_by_tokens(text: str, token_limit: int) -> tuple[str, bool]:
    """Truncate plain text to a tokenizer-based token limit."""
    if state.tokenizer is None:
        return text, False
    ids = state.tokenizer.encode(text)
    if len(ids) <= token_limit:
        return text, False
    truncated_ids = ids[:token_limit]
    return state.tokenizer.decode(truncated_ids, skip_special_tokens=True), True


def parse_label(response_text: str) -> tuple[str, float]:
    """Extract toxicity label and confidence from model output with fallback."""
    valid_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "safe"]
    lower_text = response_text.lower()
    for label in valid_labels:
        if label in lower_text:
            return label, 0.85
    return "safe", 0.60


def fallback_answer_from_document(document: str, question: str) -> str:
    """Return a fast extractive fallback answer when model generation times out."""
    try:
        doc_text = " ".join(document.split())
        if not doc_text:
            return "Not mentioned in the document."
        sentences = [s.strip() for s in doc_text.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return "Not mentioned in the document."

        question_terms = {w.lower() for w in question.split() if len(w) > 2}
        best_sentence = ""
        best_score = -1
        for sentence in sentences:
            words = {w.lower().strip(",;:!?()[]{}") for w in sentence.split()}
            score = len(question_terms.intersection(words))
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_score <= 0:
            return f"Based on the document: {sentences[0][:350]}"
        return f"Based on the document: {best_sentence[:350]}"
    except Exception:
        return "Not mentioned in the document."

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint. No auth required."""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "adapter_loaded": state.adapter_loaded
    }

@app.post("/predict")
async def predict(request: PredictRequest, api_key: str = Depends(get_api_key)):
    """Predict toxicity label, confidence, and explanation for one comment."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
        
    text = request.comment.strip() if request.comment else ""
    
    if not text:
        return {
            "label": "safe",
            "confidence": 1.0,
            "explanation": "No text to analyse"
        }
        
    if not is_english(text):
        return {
            "label": "unknown",
            "confidence": 0.0,
            "explanation": "Non-English input detected"
        }
        
    text, _ = truncate_text_by_tokens(text, 512)
    prompt = f"### Comment: {text}\n### Output:"
    try:
        response_text = await generate_response(prompt, max_new_tokens=64, max_input_tokens=512)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
        
    found_label, confidence = parse_label(response_text)
    explanation = response_text
    if "—" in response_text:
        parts = response_text.split("—", 1)
        if len(parts) > 1:
            explanation = parts[1].strip()
            
    if found_label == "safe" and confidence == 0.60:
        explanation = "Model output was ambiguous, defaulting to safe."

    return {
        "label": found_label,
        "confidence": confidence,
        "explanation": explanation
    }

@app.post("/ask")
async def ask(request: AskRequest, api_key: str = Depends(get_api_key)):
    """Answer a question from a document context with truncation metadata."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
        
    truncated_doc, was_truncated = truncate_text_by_tokens(request.doc, 2000)
    if was_truncated:
        console.print("[yellow]Document truncated to 2000 tokens for /ask.[/yellow]")

    prompt = f"### Document: {truncated_doc}\n### Question: {request.question}\n### Answer:"
    
    try:
        response_text = await generate_response(
            prompt,
            max_new_tokens=250,
            max_input_tokens=2000,
            timeout_seconds=120.0,
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            if e.status_code == 504:
                response_text = fallback_answer_from_document(truncated_doc, request.question)
            else:
                raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))
        
    if not response_text or len(response_text) < 5:
        response_text = "Not mentioned in the document."
        
    # We can estimate tokens used since we don't return them directly from inference helper
    estimated_tokens = len(state.tokenizer.encode(response_text)) if state.tokenizer else len(response_text.split())
        
    return {
        "answer": response_text,
        "tokens_used": estimated_tokens,
        "truncated": was_truncated
    }

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictRequest, api_key: str = Depends(get_api_key)):
    """Predict toxicity for up to 20 comments concurrently."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    if len(request.comments) > 20:
        raise HTTPException(status_code=400, detail="Maximum batch size is 20")

    async def run_one(index: int, comment: str) -> dict[str, Any]:
        """Execute predict path for one comment with index."""
        result = await predict(PredictRequest(comment=comment), api_key)
        result["index"] = index
        return result

    results = await asyncio.gather(*(run_one(idx, text) for idx, text in enumerate(request.comments)))
    return results
