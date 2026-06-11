from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from .config import ModelConfig, resolve_api_key


def _resolve_model_api_key(model_config: ModelConfig) -> str | None:
    api_key = resolve_api_key(model_config)
    if api_key:
        return api_key
    if model_config.provider.lower() in {"deepseek", "openai"}:
        fallback = os.getenv("OPENAI_API_KEY")
        if fallback:
            return fallback
    return None


def create_chat_model(model_config: ModelConfig) -> ChatOpenAI:
    """创建聊天模型实例。

    根据配置创建ChatOpenAI实例，支持自定义API端点。

    Args:
        model_config: 模型配置对象

    Returns:
        ChatOpenAI: 配置好的聊天模型实例

    Raises:
        RuntimeError: 当需要API密钥但未提供时抛出
    """
    api_key = _resolve_model_api_key(model_config)
    if model_config.api_key_env and not api_key:
        if model_config.provider.lower() in {"deepseek", "openai"}:
            raise RuntimeError(
                f"Missing API key: set {model_config.api_key_env} (or OPENAI_API_KEY for fallback)"
            )
        raise RuntimeError(f"Missing API key: set {model_config.api_key_env}")

    kwargs: dict[str, object] = {
        "model": model_config.model,
        "temperature": model_config.temperature,
        "streaming": True,
    }
    if model_config.api_base:
        kwargs["base_url"] = model_config.api_base
    if api_key:
        kwargs["api_key"] = api_key

    return ChatOpenAI(**kwargs)
