from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

from .groq_client import get_groq_model, groq_chat_json
from .prompts import PRODUCT_RECOGNITION_PROMPT
from .utils import extract_json_object, validate_image_path

load_dotenv()


ALLOWED_CATEGORIES = {"овощи", "фрукты", "молочные", "мясо", "рыба", "крупы", "напитки", "соусы", "другое"}



def _normalize_recognition(data: dict[str, Any]) -> dict[str, Any]:
    """Приводит ответ модели к единому формату для таблицы Streamlit и Eval."""
    normalized_products: list[dict[str, Any]] = []
    products = data.get("products", [])

    if isinstance(products, str):
        products = [p.strip() for p in products.replace(";", ",").split(",") if p.strip()]

    if isinstance(products, list):
        for item in products:
            if isinstance(item, str):
                name = item.strip()
                category = "другое"
                confidence = 0.6
                notes = "модель вернула продукт строкой"
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                category = str(item.get("category", "другое")).strip().lower()
                confidence = item.get("confidence", 0.6)
                notes = str(item.get("notes", "")).strip()
            else:
                continue

            if not name:
                continue
            if category not in ALLOWED_CATEGORIES:
                category = "другое"
            try:
                confidence_float = float(confidence)
            except Exception:
                confidence_float = 0.6
            confidence_float = max(0.0, min(1.0, confidence_float))

            normalized_products.append(
                {
                    "name": name.lower().replace("ё", "е"),
                    "category": category,
                    "confidence": round(confidence_float, 2),
                    "notes": notes,
                }
            )

    possible_missed = data.get("possible_missed", [])
    if isinstance(possible_missed, str):
        possible_missed = [p.strip() for p in possible_missed.split(",") if p.strip()]
    if not isinstance(possible_missed, list):
        possible_missed = []

    return {
        "products": normalized_products,
        "summary": str(data.get("summary", "")).strip(),
        "possible_missed": [str(x).strip() for x in possible_missed if str(x).strip()],
    }



def recognize_products(image_path: str | Path, model: str | None = None) -> dict[str, Any]:
    """Распознаёт продукты на фото через Groq Vision API."""
    image_path = validate_image_path(image_path)
    model = model or get_groq_model()
    prompt = PRODUCT_RECOGNITION_PROMPT + "\n\nВерни только JSON-объект, без текста до и после JSON."
    raw_text = groq_chat_json(prompt=prompt, image_path=image_path, model=model)
    data = extract_json_object(raw_text)
    return _normalize_recognition(data)
