from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}



def validate_image_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл изображения не найден: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат {path.suffix}. Используйте JPG, PNG, WEBP или GIF."
        )
    return path



def image_to_data_url(image_path: str | Path) -> str:
    """Преобразует локальное изображение в data URL для передачи в VLM."""
    path = validate_image_path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"



def extract_json_object(text: str) -> dict[str, Any]:
    """Достаёт JSON-объект из ответа модели.

    Модель иногда оборачивает JSON в ```json ... ```, поэтому функция сначала
    пытается распарсить ответ целиком, а потом ищет первый блок {...}.
    """
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Не удалось найти JSON в ответе модели:\n{text}")
        return json.loads(match.group(0))



def split_products(value: str) -> set[str]:
    """Разбивает строку 'молоко, яйца, сыр' на множество нормализованных продуктов."""
    if not value:
        return set()
    parts = re.split(r"[,;\n]+", value.lower())
    return {p.strip().replace("ё", "е") for p in parts if p.strip()}



def products_to_names(products: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in products:
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def top_product_names(products: list[dict[str, Any]], limit: int = 3) -> list[str]:
    """Возвращает названия продуктов с наибольшей уверенностью модели."""
    ranked_products = sorted(
        products,
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
        reverse=True,
    )
    names: list[str] = []
    seen: set[str] = set()
    for name in products_to_names(ranked_products):
        normalized_names = split_products(name)
        for normalized_name in normalized_names:
            if normalized_name not in seen:
                names.append(normalized_name)
                seen.add(normalized_name)
            if len(names) >= limit:
                return names
    return names



def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
