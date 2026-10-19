"""OpenRouter text/image requests; image encoding is confined to an in-memory buffer."""

import base64
from io import BytesIO
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .backends import Generation
from .prompts import messages_for

API_URL = "https://openrouter.ai/api/v1"


def request_json(path, *, payload=None, api_key=None, timeout=120):
    headers = {"Accept": "application/json", "User-Agent": "rendering-eval/0.1"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with urlopen(Request(API_URL + path, data=body, headers=headers), timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        # Never include request headers, encoded images, or upstream echoed bodies in saved errors.
        hint = " Check OPENROUTER_API_KEY." if error.code in (401, 403) else ""
        raise RuntimeError(f"OpenRouter HTTP {error.code}.{hint} Rerun the same command to resume.") from None
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"OpenRouter network failure ({type(error).__name__}); rerun to resume") from None
    if "error" in result:
        raise RuntimeError("OpenRouter returned an API error; no prediction was recorded")
    return result


def remote_messages(example, mode, image=None):
    messages = messages_for(example, mode, image)
    if mode == "image":
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        messages[0]["content"][0] = {
            "type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded},
        }
    return messages


class OpenRouterBackend:
    def __init__(self, config, *, transport=request_json):
        self.config, self.transport = config, transport
        self._api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY in the environment (including for :free models)")
        catalog = transport("/models", timeout=config.request_timeout)
        self.catalog = next((m for m in catalog["data"] if m["id"] == config.model_id), None)
        if self.catalog is None:
            raise ValueError(f"Model is not in the current OpenRouter catalog: {config.model_id}")
        architecture = self.catalog.get("architecture", {})
        if "image" not in architecture.get("input_modalities", []):
            raise ValueError("The selected OpenRouter model does not accept images")
        if config.openrouter_free_only:
            pricing = self.catalog.get("pricing", {})
            if not config.model_id.endswith(":free") or any(float(pricing.get(k, 1)) != 0 for k in ("prompt", "completion")):
                raise ValueError("Free-only mode requires zero-priced prompt and completion rates")

    def describe(self):
        return {
            "backend": "openrouter", "model_id": self.config.model_id,
            "weights_revision": "not exposed by provider", "image_preprocessing": "provider-managed",
            "quantization": "provider-managed", "provider": self.config.openrouter_provider,
            "free_only": self.config.openrouter_free_only,
            "catalog": {k: self.catalog.get(k) for k in (
                "id", "name", "created", "context_length", "architecture", "pricing", "supported_parameters",
            )},
        }

    def generate(self, example, mode, image=None):
        parameters = self.catalog.get("supported_parameters", [])
        provider = {"allow_fallbacks": False, "require_parameters": True}
        if self.config.openrouter_provider:
            provider["order"] = [self.config.openrouter_provider]
        if self.config.openrouter_free_only:
            provider["max_price"] = {"prompt": 0, "completion": 0}
        payload = {
            "model": self.config.model_id, "messages": remote_messages(example, mode, image),
            "max_tokens": self.config.max_new_tokens, "temperature": 0, "provider": provider,
            "stream": False,
        }
        if "seed" in parameters:
            payload["seed"] = self.config.seed
        if "reasoning" in parameters:
            payload["reasoning"] = {"enabled": False, "exclude": True}
        started = time.perf_counter()
        result = self.transport(
            "/chat/completions", payload=payload, api_key=self._api_key, timeout=self.config.request_timeout,
        )
        choices = result.get("choices", [])
        if not choices or "error" in choices[0]:
            raise RuntimeError("OpenRouter returned no usable completion")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise RuntimeError("OpenRouter returned an unexpected response format")
        usage = result.get("usage", {})
        return Generation(content, choice.get("finish_reason") == "length", {
            "latency_seconds": time.perf_counter() - started,
            "request_id": result.get("id"), "served_model": result.get("model"),
            "served_provider": result.get("provider"), "finish_reason": choice.get("finish_reason"),
            "usage": usage, "input_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"), "cost": usage.get("cost"),
            "image_preprocessing": "provider-managed" if mode == "image" else None,
        })
