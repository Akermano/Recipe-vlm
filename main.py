from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.recipe_generator import generate_recipes
from src.utils import save_json
from src.vlm_recognizer import recognize_products


def print_products(result: dict) -> None:
    print("\n=== Распознанные продукты ===")
    for i, item in enumerate(result.get("products", []), start=1):
        name = item.get("name", "-")
        category = item.get("category", "-")
        confidence = item.get("confidence", "-")
        notes = item.get("notes", "")
        print(f"{i}. {name} | категория: {category} | уверенность: {confidence} | {notes}")
    print(f"\nКраткое описание: {result.get('summary', '')}")
    possible_missed = result.get("possible_missed", [])
    if possible_missed:
        print("Что могло быть пропущено:", ", ".join(possible_missed))


def print_recipes(result: dict) -> None:
    print("\n=== Предложенные рецепты ===")
    for i, recipe in enumerate(result.get("recipes", []), start=1):
        print(f"\n{i}. {recipe.get('title', 'Без названия')}")
        print(f"   Время: {recipe.get('time_minutes', '?')} мин")
        print(f"   Сложность: {recipe.get('difficulty', '-')}")
        print(f"   Порций: {recipe.get('servings', '-')}")
        print("   Используются:", ", ".join(recipe.get("ingredients_used", [])))
        amounts = recipe.get("ingredients_with_amounts", [])
        print("   Количества:", ", ".join(amounts) if amounts else "не указаны")
        extra = recipe.get("extra_needed", [])
        print("   Дополнительно:", ", ".join(extra) if extra else "не требуется")
        print("   Шаги:")
        for step_no, step in enumerate(recipe.get("steps", []), start=1):
            print(f"   {step_no}) {step}")
        print(f"   Комментарий: {recipe.get('why_this_recipe', '')}")
    if result.get("comment"):
        print("\nОбщий комментарий:", result["comment"])


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР3: рецепт по фото холодильника")
    parser.add_argument("--image", required=True, help="Путь к фото холодильника/полки")
    parser.add_argument("--vegetarian", action="store_true", help="Включить фильтр: вегетарианское")
    parser.add_argument("--time", type=int, default=30, help="Максимальное время готовки в минутах")
    parser.add_argument("--difficulty", default="легко", choices=["легко", "средне", "сложно"], help="Желаемая сложность")
    parser.add_argument("--servings", type=int, default=2, help="Количество порций")
    args = parser.parse_args()

    recognition = recognize_products(args.image)
    recipes = generate_recipes(
        products=recognition.get("products", []),
        vegetarian=args.vegetarian,
        max_time=args.time,
        difficulty=args.difficulty,
        servings=args.servings,
    )

    print_products(recognition)
    print_recipes(recipes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "image": str(Path(args.image)),
        "filters": {
            "vegetarian": args.vegetarian,
            "max_time": args.time,
            "difficulty": args.difficulty,
            "servings": args.servings,
        },
        "recognition": recognition,
        "recipes": recipes,
    }
    output_path = Path("outputs") / f"result_{timestamp}.json"
    save_json(output_path, output)
    print(f"\nРезультат сохранён в файл: {output_path}")


if __name__ == "__main__":
    main()
