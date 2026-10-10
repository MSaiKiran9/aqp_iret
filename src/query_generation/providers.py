"""Local Hugging Face inference provider for structured query generation."""
from dataclasses import dataclass, field
import json
import time
from typing import Protocol


class LLMProvider(Protocol):
    strict: bool
    events: list[dict]
    def complete_json(self, prompt: str) -> dict: ...
    def release(self) -> None: ...


def _json_object(text: str) -> dict:
    """Extract a valid JSON object, including one wrapped in a Markdown fence."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Model output was empty.")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        parts = cleaned.split("\n", 1)
        if len(parts) == 2:
            cleaned = parts[1].rsplit("```", 1)[0].strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ValueError("Model output did not contain a valid JSON object.")


def _output_diagnostic(text: str, new_token_count: int, limit: int = 1000) -> dict:
    excerpt = text[:limit]
    return {"raw_output_repr": repr(excerpt), "raw_output_length": len(text), "raw_output_truncated": len(text) > limit, "new_token_count": new_token_count}


@dataclass
class HuggingFaceLocalProvider:
    model_id: str
    device: str = "auto"
    precision: str = "auto"
    max_new_tokens: int = 192
    temperature: float = 0.2
    do_sample: bool = False
    trust_remote_code: bool = False
    strict: bool = True
    model: object | None = field(default=None, init=False, repr=False)
    tokenizer: object | None = field(default=None, init=False, repr=False)
    events: list[dict] = field(default_factory=list, init=False)
    metadata: dict = field(default_factory=dict, init=False)

    def _load(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            selected_device = "cuda" if self.device == "auto" and torch.cuda.is_available() else ("cpu" if self.device == "auto" else self.device)
            dtype = torch.float32
            if selected_device.startswith("cuda"):
                dtype = torch.bfloat16 if self.precision in {"auto", "bf16"} and torch.cuda.is_bf16_supported() else torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
            kwargs = {"torch_dtype": dtype, "trust_remote_code": self.trust_remote_code}
            if selected_device.startswith("cuda"):
                kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            if not selected_device.startswith("cuda"):
                self.model.to(selected_device)
            memory = {"allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0, "reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0}
            self.metadata = {"model_id": self.model_id, "device": selected_device, "precision": str(dtype).replace("torch.", ""), "trust_remote_code": self.trust_remote_code, "load_seconds": round(time.perf_counter() - started, 4), **memory}
        except Exception as error:
            self.events.append({"kind": "load", "model_id": self.model_id, "success": False, "error": repr(error), "seconds": round(time.perf_counter() - started, 4)})
            raise RuntimeError(f"Could not load {self.model_id}: {error}") from error

    def complete_json(self, prompt: str) -> dict:
        self._load()
        started = time.perf_counter()
        raw_outputs = []
        try:
            messages = [{"role": "system", "content": "Return exactly one valid JSON object matching the requested schema. No Markdown fences, prose, or extra keys."}, {"role": "user", "content": prompt}]
            for attempt in range(2):
                text, new_token_count = self._generate_text(messages, self.max_new_tokens if attempt == 0 else min(64, self.max_new_tokens))
                diagnostic = _output_diagnostic(text, new_token_count)
                raw_outputs.append(diagnostic)
                try:
                    value = _json_object(text)
                    self.events.append({"kind": "generation", "model_id": self.model_id, "success": True, "seconds": round(time.perf_counter() - started, 4), "attempt": attempt + 1, "settings": self.generation_settings()})
                    return value
                except ValueError as error:
                    self.events.append({"kind": "generation_attempt", "model_id": self.model_id, "success": False, "attempt": attempt + 1, "error": str(error), "output": diagnostic})
                    if attempt == 0:
                        messages.extend([{"role": "assistant", "content": text}, {"role": "user", "content": "Your previous output was invalid. Return only one valid JSON object matching the requested schema."}])
                        continue
                    raise
        except Exception as error:
            self.events.append({"kind": "generation", "model_id": self.model_id, "success": False, "error": repr(error), "raw_outputs": raw_outputs, "seconds": round(time.perf_counter() - started, 4), "settings": self.generation_settings()})
            raise RuntimeError(f"{self.model_id} produced no valid structured output: {error}") from error

    def _generate_text(self, messages: list[dict], max_new_tokens: int) -> tuple[str, int]:
        import torch
        if getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        else:
            inputs = self.tokenizer("\n".join(item["content"] for item in messages), return_tensors="pt")
        inputs = inputs.to(next(self.model.parameters()).device)
        generation = {"max_new_tokens": max_new_tokens, "do_sample": self.do_sample, "pad_token_id": self.tokenizer.eos_token_id}
        if self.do_sample:
            generation["temperature"] = self.temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True), len(generated)

    def generation_settings(self) -> dict:
        return {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature, "do_sample": self.do_sample}

    def release(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def build_provider(config: dict, model_spec: dict | None = None) -> LLMProvider | None:
    if not config.get("enabled", False):
        return None
    if config.get("provider") != "huggingface_local":
        raise ValueError("This project experiment path requires provider=huggingface_local.")
    spec = model_spec or {"id": config.get("model"), "trust_remote_code": config.get("trust_remote_code", False)}
    if not spec.get("id"):
        raise ValueError("A Hugging Face model id is required.")
    return HuggingFaceLocalProvider(model_id=spec["id"], device=config.get("device", "auto"), precision=config.get("precision", "auto"), max_new_tokens=int(config.get("max_new_tokens", 192)), temperature=float(config.get("temperature", .2)), do_sample=bool(config.get("do_sample", False)), trust_remote_code=bool(spec.get("trust_remote_code", False)), strict=bool(config.get("strict", True)))
