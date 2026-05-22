from __future__ import annotations

from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

from .groq_client import get_groq_recipe_model, groq_chat_json
from .prompts import RECIPE_GENERATION_PROMPT
from .utils import extract_json_object, products_to_names

load_dotenv()

ALLOWED_DIFFICULTIES = {"легко", "средне", "сложно"}
DIFFICULTY_REQUIREMENTS = {
    "легко": "простые техники, минимум подготовки и короткие понятные шаги",
    "средне": "несколько этапов приготовления и умеренная подготовка без сложной техники",
    "сложно": "многоэтапное приготовление с более точной техникой и аккуратной сборкой блюда",
}


def _normalize_difficulty(value: Any, default: str = "легко") -> str:
    difficulty = str(value or default).strip().lower()
    return difficulty if difficulty in ALLOWED_DIFFICULTIES else default


def _normalize_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, min(maximum, normalized))


def _normalize_recipes(
    data: dict[str, Any],
    recipe_count: int | None = None,
    requested_difficulty: str | None = None,
    requested_servings: int | None = None,
) -> dict[str, Any]:
    recipes_raw = data.get("recipes", [])
    recipes: list[dict[str, Any]] = []

    if isinstance(recipes_raw, list):
        for item in recipes_raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "Рецепт")).strip() or "Рецепт"
            try:
                time_minutes = int(item.get("time_minutes", 30))
            except Exception:
                time_minutes = 30
            difficulty = _normalize_difficulty(item.get("difficulty"))
            if requested_difficulty is not None:
                difficulty = _normalize_difficulty(requested_difficulty)
            servings = _normalize_int(item.get("servings"), default=2, minimum=1, maximum=10)
            if requested_servings is not None:
                servings = _normalize_int(requested_servings, default=2, minimum=1, maximum=10)

            def to_list(value: Any) -> list[str]:
                if isinstance(value, list):
                    return [str(x).strip() for x in value if str(x).strip()]
                if isinstance(value, str):
                    return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
                return []

            steps = item.get("steps", [])
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split("\n") if s.strip()]
            if not isinstance(steps, list):
                steps = []

            ingredients_used = to_list(item.get("ingredients_used", []))
            ingredients_with_amounts = to_list(item.get("ingredients_with_amounts", []))

            recipes.append(
                {
                    "title": title,
                    "time_minutes": max(1, time_minutes),
                    "difficulty": difficulty,
                    "servings": servings,
                    "ingredients_used": ingredients_used,
                    "ingredients_with_amounts": ingredients_with_amounts or ingredients_used,
                    "extra_needed": to_list(item.get("extra_needed", [])),
                    "steps": [str(s).strip() for s in steps if str(s).strip()],
                    "why_this_recipe": str(item.get("why_this_recipe", "")).strip(),
                }
            )

    if recipe_count is not None:
        recipes = recipes[:recipe_count]

    return {
        "recipes": recipes,
        "comment": str(data.get("comment", "")).strip(),
    }



def generate_recipes(
    products: list[dict[str, Any]],
    vegetarian: bool = False,
    max_time: int = 30,
    difficulty: str = "легко",
    servings: int = 2,
    recipe_count: int = 3,
    model: str | None = None,
) -> dict[str, Any]:
    """Генерирует запрошенное количество рецептов по распознанным продуктам."""
    product_names = products_to_names(products)
    recipe_count = _normalize_int(recipe_count, default=3, minimum=1, maximum=10)
    difficulty = _normalize_difficulty(difficulty)
    servings = _normalize_int(servings, default=2, minimum=1, maximum=10)

    model = model or get_groq_recipe_model()
    user_task = f"""
{RECIPE_GENERATION_PROMPT}

Список найденных продуктов: {', '.join(product_names) if product_names else 'продукты не найдены'}.

Фильтры пользователя:
- вегетарианское: {'да' if vegetarian else 'нет'}
- максимум времени: {max_time} минут
- сложность: {difficulty}
- порций: {servings}
- количество рецептов: {recipe_count}

Сгенерируй ровно {recipe_count} разных рецептов. Основу рецептов строй на найденных продуктах.
Фильтры обязательны:
- каждый рецепт должен иметь сложность "{difficulty}": {DIFFICULTY_REQUIREMENTS[difficulty]};
- каждый рецепт должен быть рассчитан ровно на {servings} порций;
- поле servings должно быть равно {servings};
- поле ingredients_with_amounts должно содержать количества ингредиентов на {servings} порций;
- шаги приготовления должны учитывать выбранное количество порций;
- время каждого рецепта не должно превышать {max_time} минут.
Если нужен продукт не из списка и это не базовый продукт (соль, перец, вода, масло, сахар),
укажи его в поле extra_needed.
Верни только JSON-объект, без текста до и после JSON.
""".strip()

    raw_text = groq_chat_json(prompt=user_task, image_path=None, model=model)
    data = extract_json_object(raw_text)
    return _normalize_recipes(
        data,
        recipe_count=recipe_count,
        requested_difficulty=difficulty,
        requested_servings=servings,
    )
