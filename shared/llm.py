"""
LLM Provider Module
===================

Unified interface for multiple LLM providers:
- Mistral
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- Ollama (local models)

This is a shared module used by both prisma (stable) and prisma_v2 (experimental).

Usage:
    from shared import create_llm_client

    # Create a client for your preferred provider
    client = create_llm_client(
        provider="anthropic",
        api_key="your-api-key",
        model="claude-sonnet-4-20250514"
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


class LLMProvider(Enum):
    """Supported LLM providers"""
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


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

class MistralLLM(BaseLLM):
    """Mistral AI client"""

    DEFAULT_MODEL = "mistral-large-latest"

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
            self.client = Mistral(api_key=api_key)
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

        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
        )

        return response.choices[0].message.content.strip()


# ============================================================================
# Anthropic (Claude) Implementation
# ============================================================================

class AnthropicLLM(BaseLLM):
    """Anthropic Claude client"""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

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

    DEFAULT_MODEL = "gpt-4o"

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
        )

        return response.choices[0].message.content.strip()


# ============================================================================
# Google Gemini Implementation
# ============================================================================

class GeminiLLM(BaseLLM):
    """Google Gemini client"""

    DEFAULT_MODEL = "gemini-1.5-pro"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ):
        super().__init__(api_key, model or self.DEFAULT_MODEL, default_temperature, default_max_tokens)

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
            self._model = None  # Lazy initialization
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    def _get_model(self, system: Optional[str] = None):
        """Get or create the generative model with optional system instruction"""
        config = {
            "temperature": self.default_temperature,
            "max_output_tokens": self.default_max_tokens,
        }

        if system:
            return self.genai.GenerativeModel(
                self.model,
                generation_config=config,
                system_instruction=system
            )
        else:
            return self.genai.GenerativeModel(
                self.model,
                generation_config=config
            )

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        model = self._get_model(system)

        # Override generation config if parameters provided
        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens

        if gen_config:
            response = model.generate_content(prompt, generation_config=gen_config)
        else:
            response = model.generate_content(prompt)

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
        provider: One of "mistral", "anthropic", "openai", "gemini", "ollama"
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
            model="claude-sonnet-4-20250514"
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

    else:
        valid_providers = ["mistral", "anthropic", "openai", "gemini", "ollama"]
        raise ValueError(f"Unknown provider: {provider}. Valid providers: {valid_providers}")


__all__ = [
    "BaseLLM",
    "MistralLLM",
    "AnthropicLLM",
    "OpenAILLM",
    "GeminiLLM",
    "OllamaLLM",
    "create_llm_client",
    "LLMProvider",
]
