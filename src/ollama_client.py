from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def get_ollama_model(default: str = "qwen2.5vl:latest") -> str:
    """Возвращает имя локальной модели Ollama."""
    return os.getenv("OLLAMA_MODEL", default).strip() or default


def ollama_chat_json(
    prompt: str,
    image_path: str | Path | None = None,
    model: str | None = None,
) -> str:
    """Отправляет запрос в Ollama и возвращает текст ответа.

    Для VLM-запроса передаётся image_path. Для обычного LLM-запроса image_path не нужен.
    Ollama должен быть установлен, запущен, а модель скачана через `ollama pull ...`.
    """
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен Python-пакет ollama. Выполните команду: pip install -r requirements.txt"
        ) from exc

    model_name = model or get_ollama_model()
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip() or "http://localhost:11434"
    client = ollama.Client(host=host)

    message: dict[str, Any] = {"role": "user", "content": prompt}
    if image_path is not None:
        message["images"] = [str(Path(image_path))]

    try:
        response = client.chat(
            model=model_name,
            messages=[message],
            format="json",
            options={
                "temperature": 0.2,
                "num_ctx": 8192,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            "Не удалось подключиться к Ollama или запустить модель.\n\n"
            "Проверьте по шагам:\n"
            "1) Установлен ли Ollama.\n"
            "2) Запущен ли Ollama в фоне.\n"
            "3) Скачана ли модель: ollama pull qwen2.5vl\n"
            "4) В .env указано: OLLAMA_MODEL=qwen2.5vl:latest\n\n"
            f"Техническая ошибка: {exc}"
        ) from exc

    # Новые версии ollama-python возвращают объект, старые — словарь.
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return str(response.message.content)
    if isinstance(response, dict):
        return str(response.get("message", {}).get("content", ""))
    return str(response)
