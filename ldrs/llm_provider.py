"""
LLM Provider — Centralized multi-provider LLM client using LiteLLM.

This module provides a single, configurable interface for all LLM calls
across the LDRS v2 system. It supports multiple providers:

    - **local**:  Self-hosted models via OpenAI-compatible API
                  (e.g. qwen3-vl on gpu-router)
    - **openai**: OpenAI API (gpt-4o, gpt-4o-mini, etc.)
    - **gemini**: Google Gemini API (gemini/gemini-2.0-flash, etc.)

LiteLLM acts as a unified wrapper — all providers use the same
``completion()`` / ``acompletion()`` interface regardless of backend.

Configuration is read from environment variables (via ``.env``):

    - ``LOCAL_API_KEY``, ``LOCAL_BASE_URL``, ``LOCAL_MODEL``
    - ``OPENAI_API_KEY``, ``OPENAI_MODEL``
    - ``GEMINI_API_KEY``, ``GEMINI_MODEL``
    - ``LLM_PROVIDER``  (default provider: "local")

Usage::

    from ldrs.llm_provider import LLMProvider, get_provider

    # Get the default provider (from LLM_PROVIDER env var)
    provider = get_provider()

    # Or specify explicitly
    provider = get_provider("openai")

    # Async completion
    response = await provider.acompletion(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.4,
    )
    print(response.choices[0].message.content)

    # Sync completion
    response = provider.completion(
        messages=[{"role": "user", "content": "Hello"}],
    )

Architecture notes:
    - All LLM-calling modules receive an ``LLMProvider`` instance
      instead of creating their own ``openai.AsyncOpenAI`` clients.
    - Provider selection can be changed at runtime without restarting.
    - UTF-8 NFC normalization is NOT handled here — that remains at
      the caller's text boundaries per project convention.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# LiteLLM import — unified LLM interface
import litellm

# Suppress LiteLLM's verbose logging by default
litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Supported provider names
SUPPORTED_PROVIDERS = ("local", "openai", "gemini")


@dataclass
class ProviderConfig:
    """
    Configuration for a single LLM provider.

    Attributes:
        name:       Provider name ("local", "openai", "gemini").
        api_key:    API key for authentication.
        base_url:   Base URL for the API (only used by "local" provider).
        model:      Default model name for this provider.
    """

    name: str
    api_key: str = ""
    base_url: Optional[str] = None
    model: str = ""

    def is_configured(self) -> bool:
        """Return True if this provider has the minimum required config."""
        return bool(self.api_key and self.model)


def _load_provider_config(provider_name: str) -> ProviderConfig:
    """
    Load provider configuration from environment variables.

    Each provider has its own set of env vars:
        - local:  LOCAL_API_KEY, LOCAL_BASE_URL, LOCAL_MODEL
        - openai: OPENAI_API_KEY, OPENAI_MODEL
        - gemini: GEMINI_API_KEY, GEMINI_MODEL

    For backward compatibility, the "local" provider also falls back
    to API_KEY / BASE_URL if LOCAL_* vars are not set.
    """
    if provider_name == "local":
        return ProviderConfig(
            name="local",
            api_key=os.getenv("LOCAL_API_KEY", os.getenv("API_KEY", "")),
            base_url=os.getenv("LOCAL_BASE_URL", os.getenv("BASE_URL", "")),
            model=os.getenv("LOCAL_MODEL", os.getenv("LDRS_MODEL", "qwen3-vl")),
        )
    elif provider_name == "openai":
        return ProviderConfig(
            name="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=None,  # OpenAI uses default URL
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    elif provider_name == "gemini":
        return ProviderConfig(
            name="gemini",
            api_key=os.getenv("GEMINI_API_KEY", ""),
            base_url=None,  # Gemini uses LiteLLM's built-in routing
            model=os.getenv("GEMINI_MODEL", "gemini/gemini-3.0-flash"),
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. Supported: {SUPPORTED_PROVIDERS}"
        )


# ---------------------------------------------------------------------------
# LLMProvider class
# ---------------------------------------------------------------------------


class LLMProvider:
    """
    Centralized LLM provider using LiteLLM.

    Wraps LiteLLM's ``completion()`` and ``acompletion()`` functions
    with provider-specific configuration (API key, base URL, model).

    All LLM-calling modules in LDRS v2 should use an ``LLMProvider``
    instance rather than creating their own OpenAI clients directly.

    Args:
        provider_name: Provider to use ("local", "openai", "gemini").
        model_override: Override the default model for this provider.
        config:         Pre-built ProviderConfig (if None, loaded from env).

    Example::

        provider = LLMProvider("openai", model_override="gpt-4o")
        response = await provider.acompletion(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0,
        )
    """

    def __init__(
        self,
        provider_name: str = "local",
        model_override: Optional[str] = None,
        config: Optional[ProviderConfig] = None,
    ):
        self.config = config or _load_provider_config(provider_name)
        if model_override:
            self.config.model = model_override

        logger.debug(
            "LLMProvider init  provider=%s  model=%s  base_url=%s",
            self.config.name,
            self.config.model,
            self.config.base_url or "(default)",
        )

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        return self.config.name

    @property
    def model(self) -> str:
        """Current model name."""
        return self.config.model

    @model.setter
    def model(self, value: str) -> None:
        """Allow changing the model at runtime."""
        self.config.model = value

    def _build_kwargs(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **extra: Any,
    ) -> Dict[str, Any]:
        """
        Build the kwargs dict for litellm.completion / acompletion.

        Handles provider-specific routing:
        - local: uses ``openai/`` prefix + custom base_url + api_key
        - openai: uses model name directly + api_key
        - gemini: uses ``gemini/`` prefix + api_key
        """
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        kwargs.update(extra)

        if self.config.name == "local":
            # For local OpenAI-compatible servers, LiteLLM needs the
            # openai/ prefix and custom base_url
            model_name = self.config.model
            if not model_name.startswith("openai/"):
                model_name = f"openai/{model_name}"
            kwargs["model"] = model_name
            kwargs["api_key"] = self.config.api_key
            if self.config.base_url:
                kwargs["api_base"] = self.config.base_url

        elif self.config.name == "openai":
            kwargs["model"] = self.config.model
            kwargs["api_key"] = self.config.api_key

        elif self.config.name == "gemini":
            # LiteLLM expects gemini/ prefix for Google models
            model_name = self.config.model
            if not model_name.startswith("gemini/"):
                model_name = f"gemini/{model_name}"
            kwargs["model"] = model_name
            kwargs["api_key"] = self.config.api_key

        return kwargs

    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """
        Async LLM completion call via LiteLLM.

        Args:
            messages:    Chat messages list.
            temperature: Sampling temperature.
            **kwargs:    Extra params passed through to LiteLLM.

        Returns:
            LiteLLM response object (OpenAI-compatible format).
        """
        call_kwargs = self._build_kwargs(messages, temperature, **kwargs)
        logger.debug(
            "LLMProvider.acompletion  model=%s  msgs=%d  temp=%.1f",
            call_kwargs.get("model"),
            len(messages),
            temperature,
        )
        return await litellm.acompletion(**call_kwargs)

    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """
        Sync LLM completion call via LiteLLM.

        Args:
            messages:    Chat messages list.
            temperature: Sampling temperature.
            **kwargs:    Extra params passed through to LiteLLM.

        Returns:
            LiteLLM response object (OpenAI-compatible format).
        """
        call_kwargs = self._build_kwargs(messages, temperature, **kwargs)
        logger.debug(
            "LLMProvider.completion  model=%s  msgs=%d  temp=%.1f",
            call_kwargs.get("model"),
            len(messages),
            temperature,
        )
        return litellm.completion(**call_kwargs)

    def get_available_providers(self) -> List[str]:
        """
        Return list of configured (usable) provider names.

        Checks environment variables for each supported provider
        and returns only those that have the minimum required config.
        """
        available = []
        for name in SUPPORTED_PROVIDERS:
            try:
                cfg = _load_provider_config(name)
                if cfg.is_configured():
                    available.append(name)
            except Exception:
                pass
        return available

    def __repr__(self) -> str:
        return (
            f"LLMProvider(provider={self.config.name!r}, model={self.config.model!r})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

# Cache of provider instances keyed by (provider_name, model)
_provider_cache: Dict[str, LLMProvider] = {}


def get_provider(
    provider_name: Optional[str] = None,
    model_override: Optional[str] = None,
) -> LLMProvider:
    """
    Get or create an LLMProvider instance.

    Uses a simple cache so repeated calls with the same provider + model
    reuse the same instance.

    Args:
        provider_name: Provider to use. If None, reads from ``LLM_PROVIDER``
                       env var (default: "local").
        model_override: Override the default model for this provider.

    Returns:
        Configured LLMProvider instance.
    """
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "local")

    cache_key = f"{provider_name}:{model_override or 'default'}"
    if cache_key not in _provider_cache:
        _provider_cache[cache_key] = LLMProvider(
            provider_name=provider_name,
            model_override=model_override,
        )
    return _provider_cache[cache_key]


def clear_provider_cache() -> None:
    """Clear the provider cache (useful for testing or config changes)."""
    _provider_cache.clear()


def list_available_providers() -> List[str]:
    """
    List all providers that have valid configuration in the environment.

    Returns:
        List of provider name strings (e.g. ["local", "openai","gemini"]).
    """
    available = []
    for name in SUPPORTED_PROVIDERS:
        try:
            cfg = _load_provider_config(name)
            if cfg.is_configured():
                available.append(name)
        except Exception:
            pass
    return available


def get_provider_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed info about all providers and their configuration status.

    Returns a dict keyed by provider name with fields:
        - configured: bool
        - model: str (default model)
        - has_api_key: bool
        - has_base_url: bool
    """
    info = {}
    for name in SUPPORTED_PROVIDERS:
        try:
            cfg = _load_provider_config(name)
            info[name] = {
                "configured": cfg.is_configured(),
                "model": cfg.model,
                "has_api_key": bool(cfg.api_key),
                "has_base_url": bool(cfg.base_url),
            }
        except Exception as e:
            info[name] = {
                "configured": False,
                "model": "",
                "has_api_key": False,
                "has_base_url": False,
                "error": str(e),
            }
    return info
