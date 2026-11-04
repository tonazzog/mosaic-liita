"""
LLM Provider Module
===================

Unified interface for multiple LLM providers:
- Mistral
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- Ollama (local models)
- Claude Code (headless CLI, uses the Claude Code login instead of an API key)

Default models (the same in PRISMA, MoSAIC and NL2SPARQL; pinned ids, no "-latest"
aliases, so that results are reproducible):
    Mistral    : mistral-large-4-0   (used in the thesis evaluation)
    Anthropic  : claude-sonnet-4-6
    OpenAI     : gpt-6.1-terra       (reasoning model: no temperature)
    Gemini     : gemini-3.6-flash    (thinking model)
    Ollama     : llama3.1            (local, no cost)
    Claude Code: claude-sonnet-4-6

This is a shared module used by both prisma (stable) and prisma_v2 (experimental).

Usage:
    from shared import create_llm_client

    # Create a client for your preferred provider
    client = create_llm_client(
        provider="anthropic",
        api_key="your-api-key",
        model="claude-sonnet-4-6"
    )

    # Use the client
    response = client.complete(
        prompt="Your prompt here",
        system="Optional system prompt",
        temperature=0.0
    )
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import os
import shutil
import subprocess
import tempfile
import time
import re
from datetime import datetime, timedelta


# Thinking/reasoning models spend part of the output budget on reasoning: keep a floor.
REASONING_MIN_TOKENS = 16000


def is_openai_reasoning_model(model: str) -> bool:
    """gpt-5*, gpt-6* and o-series models (no temperature, max_completion_tokens)."""
    return (model or "").lower().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def is_gemini_thinking_model(model: str) -> bool:
    """Gemini 2.5 and later think by default; the thinking counts in max_output_tokens."""
    m = (model or "").lower()
    return m.startswith("gemini-2.5") or m.startswith("gemini-3")


class LLMProvider(Enum):
    """Supported LLM providers"""
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    CLAUDE_CODE = "claudecode"


class BaseLLM(ABC):
    """
    Abstract base class for LLM clients.

    All provider implementations must implement the complete() method.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "",
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        self.api_key = api_key
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt/message
            system: Optional system prompt for context
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            **kwargs: Provider-specific parameters

        Returns:
            The generated text response
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass


# ============================================================================
# Mistral Implementation
# ============================================================================

def _mistral_text(content) -> str:
    """Answer text of a Mistral message: a plain string, or the text chunks of a chunk list
    (reasoning models return [ThinkChunk, TextChunk, ...]; the thinking is discarded)."""
    if isinstance(content, str):
        return content.strip()
    parts = [getattr(ch, "text", "") for ch in (content or []) if getattr(ch, "type", None) == "text"]
    return "".join(parts).strip()


def _log_mistral_call(model: str, response, seconds: float) -> None:
    """Append one JSON line per call to $MISTRAL_LOG (if set), with token usage incl. reasoning."""
    path = os.environ.get("MISTRAL_LOG")
    if not path:
        return
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else getattr(details, "reasoning_tokens", None)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "requested_model": model,
        "model_used": response.model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning,
        "finish_reason": response.choices[0].finish_reason,
        "duration_s": round(seconds, 2),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class MistralLLM(BaseLLM):
    """Mistral AI client"""

    DEFAULT_MODEL = "mistral-large-4-0"

    MIN_MAX_TOKENS = 16000  # floor for reasoning models (thinking counts towards max_tokens)

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, default_temperature, default_max_tokens)

        try:
            from mistralai import Mistral
            from mistralai.utils import RetryConfig, BackoffStrategy
            # Retry rate limits (429) and 5xx with exponential backoff (ms), up to 10 min per call
            retry = RetryConfig("backoff", BackoffStrategy(2000, 60000, 2.0, 600000), True)
            self.client = Mistral(api_key=api_key, retry_config=retry)
        except ImportError:
            raise ImportError("Please install mistralai: pip install mistralai")

    @property
    def provider_name(self) -> str:
        return "Mistral"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        messages = []

        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        requested = max_tokens if max_tokens is not None else self.default_max_tokens
        started = time.time()
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            top_p=1.0,  # required by recent Mistral models with greedy sampling (temperature 0)
            # Reasoning models (e.g. mistral-large-4) spend part of max_tokens on thinking,
            # so keep a floor; models without reasoning simply stop earlier.
            max_tokens=max(requested, self.MIN_MAX_TOKENS),
        )
        _log_mistral_call(self.model, response, time.time() - started)

        choice = response.choices[0]
        text = _mistral_text(choice.message.content)
        if not text and choice.finish_reason == "length":
            raise RuntimeError("Mistral response truncated before any answer text (max_tokens reached while reasoning)")
        return text


# ============================================================================
# Anthropic (Claude) Implementation
# ============================================================================

class AnthropicLLM(BaseLLM):
    """Anthropic Claude client"""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, default_temperature, default_max_tokens)

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        messages = [{"role": "user", "content": prompt}]

        kwargs_call = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }

        # Anthropic uses separate system parameter
        if system:
            kwargs_call["system"] = system

        # Temperature (Anthropic uses 0-1 range)
        temp = temperature if temperature is not None else self.default_temperature
        if temp > 0:
            kwargs_call["temperature"] = temp

        response = self.client.messages.create(**kwargs_call)

        return response.content[0].text.strip()


# ============================================================================
# OpenAI (GPT) Implementation
# ============================================================================

class OpenAILLM(BaseLLM):
    """OpenAI GPT client"""

    DEFAULT_MODEL = "gpt-6.1-terra"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, default_temperature, default_max_tokens)

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        messages = []

        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        requested = max_tokens if max_tokens is not None else self.default_max_tokens
        params = {"model": self.model, "messages": messages}
        if is_openai_reasoning_model(self.model):
            # Reasoning models (gpt-5*, gpt-6*, o-series) accept only the default
            # temperature and count their thinking in max_completion_tokens.
            params["max_completion_tokens"] = max(requested, REASONING_MIN_TOKENS)
        else:
            params["temperature"] = temperature if temperature is not None else self.default_temperature
            params["max_tokens"] = requested

        response = self.client.chat.completions.create(**params)

        return response.choices[0].message.content.strip()


# ============================================================================
# Google Gemini Implementation
# ============================================================================

class GeminiLLM(BaseLLM):
    """Google Gemini client (uses google-genai SDK)"""

    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, default_temperature, default_max_tokens)

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.types = genai.types
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    def _output_budget(self, max_tokens: Optional[int]) -> int:
        requested = max_tokens if max_tokens is not None else self.default_max_tokens
        return max(requested, REASONING_MIN_TOKENS) if is_gemini_thinking_model(self.model) else requested

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        config = self.types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.default_temperature,
            max_output_tokens=self._output_budget(max_tokens),
            system_instruction=system,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        return response.text.strip()


# ============================================================================
# Ollama (Local) Implementation
# ============================================================================

class OllamaLLM(BaseLLM):
    """Ollama local LLM client"""

    DEFAULT_MODEL = "llama3.1"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        api_key: Optional[str] = None,  # Not needed for Ollama
        model: str = DEFAULT_MODEL,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
        host: str = DEFAULT_HOST,
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, default_temperature, default_max_tokens)
        self.host = host

        try:
            import ollama
            self.client = ollama.Client(host=host)
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")

    @property
    def provider_name(self) -> str:
        return "Ollama"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        messages = []

        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        options = {
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )

        return response["message"]["content"].strip()


# ============================================================================
# Claude Code (headless CLI) Implementation
# ============================================================================

class ClaudeCodeLLM(BaseLLM):
    """
    Claude via the Claude Code CLI in headless mode (`claude -p`).

    Uses the local Claude Code login, so no API key is needed. The CLI does not
    expose temperature or max_tokens, so those arguments are ignored. Tools are
    disabled and the system prompt is replaced, so the model only sees our prompt.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        api_key: Optional[str] = None,  # Not needed for Claude Code
        model: str = DEFAULT_MODEL,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, default_temperature, default_max_tokens)
        self.timeout = timeout

        self.executable = shutil.which("claude")
        if not self.executable:
            raise ImportError("Claude Code CLI not found on PATH: https://docs.claude.com/en/docs/claude-code")

    @property
    def provider_name(self) -> str:
        return "ClaudeCode"

    MAX_ATTEMPTS = 30  # usage-limit waits can span several 5-hour windows

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """One Claude Code call, waiting out usage limits and CLI self-updates."""
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                return self._complete_once(prompt, system)
            except FileNotFoundError:
                # The CLI can briefly disappear while it updates itself: wait and look again
                time.sleep(60)
                self.executable = shutil.which("claude") or self.executable
            except RuntimeError as e:
                wait = _usage_limit_wait_seconds(str(e))
                if wait is None or attempt == self.MAX_ATTEMPTS:
                    raise
                print(f"[claudecode] usage limit reached, waiting {wait / 60:.0f} min: {e}", flush=True)
                time.sleep(wait)
        raise RuntimeError("Claude Code CLI: too many retries")

    def _complete_once(self, prompt: str, system: Optional[str] = None) -> str:
        # Run in an empty temp dir so the project's CLAUDE.md is not picked up
        with tempfile.TemporaryDirectory() as workdir:
            cmd = [
                self.executable, "-p",
                "--output-format", "json",
                "--model", self.model,
                "--tools", "",
                "--max-turns", "1",
                "--strict-mcp-config",
                "--no-session-persistence",
            ]

            # System prompt goes through a file and the prompt through stdin,
            # to stay under the Windows command-line length limit
            if system:
                system_file = os.path.join(workdir, "system_prompt.txt")
                with open(system_file, "w", encoding="utf-8") as f:
                    f.write(system)
                cmd += ["--system-prompt-file", system_file]

            started = time.time()
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=workdir,
                timeout=self.timeout,
            )

        try:
            output = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Claude Code CLI failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )

        _log_claude_code_call(self.model, output, time.time() - started, len(prompt) + len(system or ""))

        if output.get("is_error"):
            raise RuntimeError(f"Claude Code CLI error: {output.get('result') or output.get('subtype')}")

        return output["result"].strip()


def _usage_limit_wait_seconds(message: str) -> Optional[float]:
    """Seconds to wait if message is a Claude usage-limit error, else None.

    Parses e.g. "You've hit your 5-hour limit - resets 1:30pm (Europe/Rome)" (local time);
    falls back to 15 minutes when the reset time cannot be read.
    """
    if "limit" not in message.lower() or "hit your" not in message.lower():
        return None
    m = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", message, re.IGNORECASE)
    if not m:
        return 15 * 60
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    now = datetime.now()
    reset = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if reset <= now:
        if now - reset < timedelta(hours=1):
            return 120  # the reset has just happened: retry shortly
        reset += timedelta(days=1)
    return (reset - now).total_seconds() + 120  # small margin after the reset


def _log_claude_code_call(model: str, output: dict, seconds: float, prompt_chars: int) -> None:
    """Append one JSON line per call to $CLAUDECODE_LOG (if set), with the exact model that answered."""
    path = os.environ.get("CLAUDECODE_LOG")
    if not path:
        return
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "requested_model": model,
        "models_used": list(output.get("modelUsage", {})),
        "is_error": output.get("is_error"),
        "duration_s": round(seconds, 2),
        "prompt_chars": prompt_chars,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================================
# Factory Function
# ============================================================================

def create_llm_client(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs
) -> BaseLLM:
    """
    Factory function to create an LLM client for the specified provider.

    Args:
        provider: One of "mistral", "anthropic", "openai", "gemini", "ollama", "claudecode"
        api_key: API key for the provider (falls back to environment variable if not provided)
        model: Model name (uses provider default if not specified)
        temperature: Default sampling temperature
        max_tokens: Default max tokens for responses
        **kwargs: Provider-specific arguments (e.g., host for Ollama)

    Returns:
        An LLM client instance

    Environment Variables:
        MISTRAL_API_KEY: API key for Mistral
        ANTHROPIC_API_KEY: API key for Anthropic
        OPENAI_API_KEY: API key for OpenAI
        GOOGLE_API_KEY: API key for Google Gemini

    Example:
        client = create_llm_client(
            provider="anthropic",
            api_key="sk-...",
            model="claude-sonnet-4-6"
        )
        response = client.complete("Hello!", system="You are a helpful assistant.")
    """
    import os

    provider_lower = provider.lower().strip()

    # Environment variable mapping for API keys
    env_var_map = {
        "mistral": "MISTRAL_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    # Get API key from parameter or environment variable
    def get_api_key(provider_name: str) -> str:
        if api_key:
            return api_key
        env_var = env_var_map.get(provider_name)
        if env_var:
            env_key = os.environ.get(env_var)
            if env_key:
                return env_key
        raise ValueError(
            f"API key required for {provider_name}. "
            f"Provide via api_key parameter or set {env_var_map.get(provider_name, 'environment variable')}"
        )

    if provider_lower == "mistral":
        return MistralLLM(
            api_key=get_api_key("mistral"),
            model=model or MistralLLM.DEFAULT_MODEL,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    elif provider_lower == "anthropic":
        return AnthropicLLM(
            api_key=get_api_key("anthropic"),
            model=model or AnthropicLLM.DEFAULT_MODEL,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    elif provider_lower == "openai":
        return OpenAILLM(
            api_key=get_api_key("openai"),
            model=model or OpenAILLM.DEFAULT_MODEL,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    elif provider_lower == "gemini" or provider_lower == "google":
        return GeminiLLM(
            api_key=get_api_key("gemini"),
            model=model or GeminiLLM.DEFAULT_MODEL,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    elif provider_lower == "ollama":
        host = kwargs.get("host", OllamaLLM.DEFAULT_HOST)
        return OllamaLLM(
            model=model or OllamaLLM.DEFAULT_MODEL,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
            host=host,
        )

    elif provider_lower == "claudecode":
        return ClaudeCodeLLM(
            model=model or ClaudeCodeLLM.DEFAULT_MODEL,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    else:
        valid_providers = ["mistral", "anthropic", "openai", "gemini", "ollama", "claudecode"]
        raise ValueError(f"Unknown provider: {provider}. Valid providers: {valid_providers}")


__all__ = [
    "BaseLLM",
    "MistralLLM",
    "AnthropicLLM",
    "OpenAILLM",
    "GeminiLLM",
    "OllamaLLM",
    "ClaudeCodeLLM",
    "create_llm_client",
    "LLMProvider",
]
