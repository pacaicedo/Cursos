"""Unified LLM interface for all course labs.

Tries Ollama first (local daemon, better quality).
Falls back automatically to a small HuggingFace model if Ollama is not running,
so the same notebook works on Google Colab without any local server.

Usage
-----
    from utils.llm import SimpleLLM
    llm = SimpleLLM()
    print(llm.generate("Explain gradient descent in one sentence."))
"""

from __future__ import annotations

_OLLAMA_URL = "http://localhost:11434"
_HF_FALLBACK = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_OLLAMA_DEFAULT = "llama3.2:1b"


def _detect_ollama() -> bool:
    try:
        import requests
        r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


class SimpleLLM:
    """Unified chat interface that tries Ollama, then falls back to HuggingFace.

    Parameters
    ----------
    ollama_model:
        Name of the Ollama model to use when Ollama is running.
        Default: ``"llama3.2:1b"``.  Run ``ollama pull <name>`` first.
    hf_model:
        HuggingFace model ID used as fallback when Ollama is not available.
        Default: ``"HuggingFaceTB/SmolLM2-1.7B-Instruct"`` (~3.4 GB, runs on
        a free Colab T4 GPU or a modern laptop CPU).
    """

    def __init__(
        self,
        ollama_model: str = _OLLAMA_DEFAULT,
        hf_model: str = _HF_FALLBACK,
    ):
        self.ollama_model = ollama_model
        self.hf_model = hf_model
        self._pipe = None

        if _detect_ollama():
            self._backend = "ollama"
            print(f"[SimpleLLM] Ollama detected → using model '{self.ollama_model}'")
            print(f"            Tip: run `ollama pull {self.ollama_model}` if not yet downloaded.")
        else:
            self._backend = "huggingface"
            print(f"[SimpleLLM] Ollama not found → falling back to '{self.hf_model}' via HuggingFace.")
            print("            The model will be downloaded on first use (~3 GB).")

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs) -> str:
        """Send a single user prompt and return the model's reply."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Send a list of messages and return the assistant's reply.

        Parameters
        ----------
        messages:
            List of ``{"role": str, "content": str}`` dicts.
            Roles: ``"system"``, ``"user"``, ``"assistant"``.
        max_new_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature. ``0`` = greedy (deterministic), ``1`` = creative.
        """
        if self._backend == "ollama":
            return self._ollama_chat(messages, max_new_tokens, temperature)
        return self._hf_chat(messages, max_new_tokens, temperature)

    # ── backends ──────────────────────────────────────────────────────────────

    def _ollama_chat(self, messages, max_new_tokens, temperature):
        import requests
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_new_tokens, "temperature": temperature},
        }
        r = requests.post(f"{_OLLAMA_URL}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _hf_chat(self, messages, max_new_tokens, temperature):
        if self._pipe is None:
            self._pipe = self._load_hf_pipeline()
        out = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_full_text=False,
        )
        generated = out[0]["generated_text"]
        # pipeline with messages returns a list of message dicts
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()

    def _load_hf_pipeline(self):
        print(f"Loading '{self.hf_model}' … (this may take a few minutes the first time)", flush=True)
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=self.hf_model,
            device_map="auto",
            torch_dtype="auto",
        )
        print("Model ready.", flush=True)
        return pipe

    # ── helpers ───────────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Approximate token count using the HuggingFace tokenizer."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            self.hf_model if self._backend == "huggingface" else _HF_FALLBACK,
            trust_remote_code=True,
        )
        return len(tok.encode(text))

    def __repr__(self) -> str:
        model = self.ollama_model if self._backend == "ollama" else self.hf_model
        return f"SimpleLLM(backend={self._backend!r}, model={model!r})"
