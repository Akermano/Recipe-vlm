from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import image_to_data_url


DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_GROQ_RECIPE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Не найден GROQ_API_KEY.\n\n"
            "1) Создайте API-ключ в console.groq.com\n"
            "2) Откройте файл .env\n"
            "3) Запишите строку: GROQ_API_KEY=ваш_ключ\n"
        )
    return api_key


def get_groq_model(default: str = DEFAULT_GROQ_MODEL) -> str:
    return os.getenv("GROQ_MODEL", default).strip() or default


def get_groq_recipe_model(default: str = DEFAULT_GROQ_RECIPE_MODEL) -> str:
    return os.getenv("GROQ_RECIPE_MODEL", "").strip() or get_groq_model(default)


def _make_client():
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "Не установлен Python-пакет groq. Выполните команду: pip install -r requirements.txt"
        ) from exc

    return Groq(api_key=get_groq_api_key())


def groq_chat_json(
    prompt: str,
    image_path: str | Path | None = None,
    model: str | None = None,
    max_completion_tokens: int = 1400,
) -> str:
    """Отправляет запрос в Groq API и возвращает текст ответа."""
    client = _make_client()
    model_name = model or get_groq_model()

    if image_path is not None:
        data_url = image_to_data_url(image_path)
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=max_completion_tokens,
            top_p=1,
            response_format={"type": "json_object"},
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Не удалось выполнить запрос к Groq API.\n\n"
            "Проверьте по шагам:\n"
            "1) Есть ли интернет.\n"
            "2) Правильно ли указан GROQ_API_KEY в .env.\n"
            f"3) Доступна ли модель {model_name}.\n\n"
            f"Техническая ошибка: {exc}"
        ) from exc

    return str(completion.choices[0].message.content or "")
