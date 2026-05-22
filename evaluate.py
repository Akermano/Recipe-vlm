from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import products_to_names, split_products, top_product_names
from src.vlm_recognizer import recognize_products


def evaluate_row(image_path: Path, expected_products: str) -> dict:
    recognition = recognize_products(image_path)
    predicted_names = products_to_names(recognition.get("products", []))

    expected = split_products(expected_products)
    predicted = split_products(", ".join(predicted_names))

    found = expected & predicted
    missed = expected - predicted
    invented = predicted - expected

    precision = len(found) / len(predicted) if predicted else 0.0
    recall = len(found) / len(expected) if expected else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    top_three = top_product_names(recognition.get("products", []), limit=3)
    top_3_hit_rate = len(expected & set(top_three)) / len(top_three) if top_three else 0.0

    return {
        "image_file": image_path.name,
        "expected_products": ", ".join(sorted(expected)),
        "predicted_products": ", ".join(sorted(predicted)),
        "found": ", ".join(sorted(found)),
        "missed_by_model": ", ".join(sorted(missed)),
        "invented_by_model": ", ".join(sorted(invented)),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "top_3_hit_rate": round(top_3_hit_rate, 3),
        "summary": recognition.get("summary", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ручная оценка качества распознавания продуктов")
    parser.add_argument("--expected", default="data/eval_expected.csv", help="CSV с колонками image_file, expected_products")
    parser.add_argument("--images-dir", default="data/eval_images", help="Папка с 10–20 фото для оценки")
    parser.add_argument("--out", default="outputs/eval_results.csv", help="Куда сохранить результаты")
    args = parser.parse_args()

    expected_path = Path(args.expected)
    images_dir = Path(args.images_dir)
    out_path = Path(args.out)

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Не найден файл {expected_path}. Заполните data/eval_expected.csv по шаблону."
        )

    df = pd.read_csv(expected_path)
    required = {"image_file", "expected_products"}
    if not required.issubset(df.columns):
        raise ValueError("CSV должен содержать колонки: image_file, expected_products")

    rows = []
    for _, row in df.iterrows():
        image_path = images_dir / str(row["image_file"])
        if not image_path.exists():
            print(f"Пропуск: не найден файл {image_path}")
            continue
        print(f"Оцениваю: {image_path}")
        rows.append(evaluate_row(image_path, str(row["expected_products"])))

    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    if len(result) > 0:
        print("\n=== Итоговая оценка ===")
        print(f"Количество фото: {len(result)}")
        print(f"Средняя precision: {result['precision'].mean():.3f}")
        print(f"Средняя recall: {result['recall'].mean():.3f}")
        print(f"Средняя F1: {result['f1_score'].mean():.3f}")
        print(f"Средняя Top-3 hit rate: {result['top_3_hit_rate'].mean():.3f}")
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
