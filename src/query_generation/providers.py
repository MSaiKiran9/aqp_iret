"""Minimal provider abstraction for structured LLM query planning."""
from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib.request import Request, urlopen


class LLMProvider(Protocol):
    def complete_json(self, prompt: str) -> dict: ...


@dataclass
class OpenAICompatibleProvider:
    model: str
    api_key_env: str
    base_url: str
    timeout_seconds: int = 30

    def complete_json(self, prompt: str) -> dict:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {self.api_key_env}.")
        payload = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": "Return valid JSON only."}, {"role": "user", "content": prompt}]}
        request = Request(self.base_url, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode())
        return json.loads(body["choices"][0]["message"]["content"])


def build_provider(config: dict) -> LLMProvider | None:
    if not config.get("enabled", False):
        return None
    if config.get("provider") != "openai_compatible":
        raise ValueError("Only the documented openai_compatible provider is configured.")
    return OpenAICompatibleProvider(config["model"], config["api_key_env"], config["base_url"], int(config.get("timeout_seconds", 30)))
