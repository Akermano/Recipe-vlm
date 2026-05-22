from __future__ import annotations

import tempfile
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.recipe_generator import generate_recipes
from src.utils import products_to_names, save_json, split_products, top_product_names
from src.vlm_recognizer import recognize_products

EVAL_EXPECTED_PATH = Path("data") / "eval_expected.csv"


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #eef3f2;
            color: #15242a;
        }

        [data-testid="stHeader"] {
            background: rgba(238, 243, 242, 0.92);
        }

        .main .block-container {
            max-width: 1360px;
            padding-top: 1.7rem;
            padding-bottom: 3.5rem;
        }

        section[data-testid="stSidebar"] {
            background: #10272e;
            border-right: 1px solid #1c434b;
        }

        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.9rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] .stCaption p {
            color: #f4f8f7;
        }

        .main h1,
        .main h2,
        .main h3,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3 {
            color: #0c3137 !important;
            letter-spacing: 0;
        }

        .main h1 {
            border-bottom: 1px solid #c7d7d5;
            font-size: 2.35rem;
            line-height: 1.15;
            margin-bottom: 1.1rem;
            padding-bottom: 0.9rem;
        }

        .main h2,
        .main h3 {
            margin-top: 0.45rem;
        }

        label p,
        .stMarkdown p,
        [data-testid="stCaptionContainer"] p {
            color: #2f444b;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: #ffffff;
            border: 1px dashed #277675;
            border-radius: 8px;
            min-height: 9rem;
            padding: 1rem;
        }

        div[data-testid="stFileUploaderDropzone"] button {
            background: #eef7f5;
            color: #0c3137 !important;
        }

        .stTextArea textarea,
        .stTextInput input,
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            background: #ffffff;
            border-color: #c7d7d5;
            border-radius: 8px;
            color: #15242a;
        }

        .stTextArea textarea:focus,
        .stTextInput input:focus,
        div[data-testid="stNumberInput"] input:focus {
            border-color: #176b67;
            box-shadow: 0 0 0 1px #176b67;
        }

        .stButton > button {
            background: #ffffff;
            min-height: 2.6rem;
            border-color: #176b67;
            border-radius: 6px;
            color: #0c3137 !important;
            font-weight: 600;
        }

        .stButton > button p,
        .stButton > button span {
            color: #0c3137 !important;
        }

        .stButton > button[kind="primary"] {
            background: #176b67;
            border-color: #176b67;
        }

        .stButton > button[kind="primary"],
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span {
            color: #ffffff !important;
        }

        .stButton > button:hover {
            background: #e5f1ef;
            border-color: #0f5653;
        }

        .stButton > button[kind="primary"]:hover {
            background: #125956;
            border-color: #125956;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #cbd9d7;
            border-left: 4px solid #d99836;
            border-radius: 8px;
            min-height: 6.2rem;
            padding: 0.85rem 1rem;
        }

        div[data-testid="stMetric"] label p {
            color: #486067;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #10272e;
        }

        div[data-testid="stDataFrame"],
        details[data-testid="stExpander"],
        div[data-testid="stVegaLiteChart"] {
            background: #ffffff;
            border: 1px solid #cbd9d7;
            border-radius: 8px;
            overflow: hidden;
        }

        details[data-testid="stExpander"] summary {
            background: #f7faf9;
            color: #10272e;
            font-weight: 600;
        }

        [data-testid="stImage"] img {
            background: #ffffff;
            border: 1px solid #cbd9d7;
            border-radius: 8px;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        hr {
            border-color: #c7d7d5;
            margin-top: 1.9rem;
            margin-bottom: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_recognition_frame(products: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(products)
    if frame.empty:
        return frame

    frame = frame.rename(
        columns={
            "name": "Продукт",
            "category": "Категория",
            "confidence": "Уверенность",
            "notes": "Комментарий",
        }
    )
    if "Уверенность" in frame:
        frame["Уверенность"] = frame["Уверенность"].map(lambda value: f"{float(value):.0%}")
    return frame


def get_eval_expected_products(image_name: str) -> str:
    if not EVAL_EXPECTED_PATH.exists():
        return ""

    expected_frame = pd.read_csv(EVAL_EXPECTED_PATH)
    required_columns = {"image_file", "expected_products"}
    if not required_columns.issubset(expected_frame.columns):
        return ""

    uploaded_name = Path(image_name).name.casefold()
    file_names = expected_frame["image_file"].map(lambda value: Path(str(value)).name.casefold())
    match = expected_frame.loc[file_names == uploaded_name, "expected_products"]
    if match.empty or pd.isna(match.iloc[0]):
        return ""
    return str(match.iloc[0]).strip()


def choose_manual_products(manual_products: str, image_name: str) -> tuple[str, str]:
    if manual_products.strip():
        return manual_products, "ручной ввод"

    eval_products = get_eval_expected_products(image_name)
    if eval_products:
        return eval_products, f"data/eval_expected.csv для {Path(image_name).name}"
    return "", "ручной список не найден"


def compare_products(
    manual_products: str,
    recognized_products: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    manual_names = split_products(manual_products)
    model_names = split_products(", ".join(products_to_names(recognized_products)))
    matched_names = manual_names & model_names

    rows: list[dict[str, str]] = []
    for product in sorted(manual_names | model_names):
        if product in matched_names:
            status = "Совпадает"
        elif product in manual_names:
            status = "Только вручную"
        else:
            status = "Только модель"

        rows.append(
            {
                "Продукт": product,
                "Вручную": "Да" if product in manual_names else "",
                "Модель": "Да" if product in model_names else "",
                "Статус": status,
            }
        )

    return pd.DataFrame(rows), {
        "manual": manual_names,
        "model": model_names,
        "matched": matched_names,
    }


def calculate_quality_metrics(
    comparison: dict[str, set[str]],
    products: list[dict[str, Any]],
) -> dict[str, float | list[str]]:
    manual_names = comparison["manual"]
    model_names = comparison["model"]
    matched_names = comparison["matched"]

    precision = len(matched_names) / len(model_names) if model_names else 0.0
    recall = len(matched_names) / len(manual_names) if manual_names else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    top_three = top_product_names(products, limit=3)
    top_three_matches = sorted(set(top_three) & manual_names)
    top_three_hit_rate = len(top_three_matches) / len(top_three) if top_three else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "top_3_hit_rate": top_three_hit_rate,
        "top_3_products": top_three,
        "top_3_matches": top_three_matches,
    }


def make_quality_metrics_frame(metrics: dict[str, float | list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Метрика": "Precision",
                "Значение": f"{metrics['precision']:.1%}",
                "Смысл": "Какая доля продуктов модели совпала с ручным списком",
            },
            {
                "Метрика": "Recall",
                "Значение": f"{metrics['recall']:.1%}",
                "Смысл": "Какая доля ручного списка найдена моделью",
            },
            {
                "Метрика": "F1 score",
                "Значение": f"{metrics['f1_score']:.1%}",
                "Смысл": "Баланс precision и recall",
            },
            {
                "Метрика": "Top-3 hit rate",
                "Значение": f"{metrics['top_3_hit_rate']:.1%}",
                "Смысл": "Совпадения среди 3 самых уверенных продуктов модели",
            },
        ]
    )


def make_metrics_chart_frame(metrics: dict[str, float | list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Метрика": ["Precision", "Recall", "F1 score", "Top-3"],
            "Значение": [
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["top_3_hit_rate"],
            ],
        }
    )


def make_comparison_chart_frame(comparison: dict[str, set[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Группа": ["Совпало", "Пропущено", "Только модель"],
            "Количество": [
                len(comparison["matched"]),
                len(comparison["manual"] - comparison["model"]),
                len(comparison["model"] - comparison["manual"]),
            ],
        }
    )


def make_breakdown_frame(
    comparison: dict[str, set[str]],
    metrics: dict[str, float | list[str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Группа": "Совпало",
                "Количество": len(comparison["matched"]),
                "Продукты": ", ".join(sorted(comparison["matched"])) or "-",
            },
            {
                "Группа": "Пропущено моделью",
                "Количество": len(comparison["manual"] - comparison["model"]),
                "Продукты": ", ".join(sorted(comparison["manual"] - comparison["model"])) or "-",
            },
            {
                "Группа": "Только модель",
                "Количество": len(comparison["model"] - comparison["manual"]),
                "Продукты": ", ".join(sorted(comparison["model"] - comparison["manual"])) or "-",
            },
            {
                "Группа": "Top-3 модели",
                "Количество": len(metrics["top_3_products"]),
                "Продукты": ", ".join(metrics["top_3_products"]) or "-",
            },
        ]
    )


def make_recipes_frame(recipes_result: dict[str, Any] | None) -> pd.DataFrame:
    if not recipes_result:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for number, recipe in enumerate(recipes_result.get("recipes", []), start=1):
        rows.append(
            {
                "Номер": number,
                "Название": recipe.get("title", ""),
                "Время, мин": recipe.get("time_minutes", ""),
                "Сложность": recipe.get("difficulty", ""),
                "Порций": recipe.get("servings", ""),
                "Используются": ", ".join(recipe.get("ingredients_used", [])),
                "Ингредиенты с количеством": ", ".join(recipe.get("ingredients_with_amounts", [])),
                "Дополнительно": ", ".join(recipe.get("extra_needed", [])),
                "Шаги": " | ".join(recipe.get("steps", [])),
                "Почему подходит": recipe.get("why_this_recipe", ""),
            }
        )
    return pd.DataFrame(rows)


def make_quality_conclusion(comparison: dict[str, set[str]], f1_score: float) -> str:
    matched_count = len(comparison["matched"])
    missed_count = len(comparison["manual"] - comparison["model"])
    extra_count = len(comparison["model"] - comparison["manual"])

    if f1_score >= 0.8:
        assessment = "распознавание хорошо совпадает с ручной проверкой"
    elif missed_count > extra_count:
        assessment = "главная ошибка сейчас - пропуски продуктов на фото"
    elif extra_count > missed_count:
        assessment = "модель чаще добавляет продукты, которых нет в ручном списке"
    else:
        assessment = "совпадение частичное, пропуски и лишние ответы сбалансированы"

    return (
        f"Вывод: {assessment}. Совпало: {matched_count}; "
        f"пропущено моделью: {missed_count}; только у модели: {extra_count}."
    )


def save_bar_chart_image(
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    output_path: Path,
    max_value: float | None = None,
) -> None:
    width, height = 1080, 620
    margin_left, margin_right = 120, 56
    margin_top, margin_bottom = 96, 132
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "#f7faf9")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=20, fill="#ffffff", outline="#c8d7d5")
    draw.text((margin_left, 52), title, fill="#123138")

    values = [float(value) for value in frame[value_column].tolist()]
    chart_max = max_value if max_value is not None else max(values + [1.0])
    chart_max = chart_max or 1.0
    tick_count = 4
    plot_bottom = margin_top + plot_height
    colors = ["#176b67", "#d99836", "#365b8c", "#b85a53"]

    for tick in range(tick_count + 1):
        tick_value = chart_max * tick / tick_count
        y = plot_bottom - plot_height * tick / tick_count
        draw.line((margin_left, y, width - margin_right, y), fill="#dce7e5", width=2)
        tick_label = f"{tick_value:.0%}" if chart_max <= 1.0 else f"{tick_value:.0f}"
        draw.text((40, y - 8), tick_label, fill="#51636a")

    bar_count = max(len(frame), 1)
    slot_width = plot_width / bar_count
    bar_width = min(136, slot_width * 0.55)
    ascii_labels = {
        "Совпало": "Matched",
        "Пропущено": "Missed",
        "Только модель": "Model only",
    }

    for index, row in frame.reset_index(drop=True).iterrows():
        value = float(row[value_column])
        center_x = margin_left + slot_width * (index + 0.5)
        x0 = center_x - bar_width / 2
        x1 = center_x + bar_width / 2
        y0 = plot_bottom - plot_height * value / chart_max
        visible_y0 = min(y0, plot_bottom - 4)
        draw.rounded_rectangle((x0, visible_y0, x1, plot_bottom), radius=10, fill=colors[index % len(colors)])

        value_label = f"{value:.1%}" if chart_max <= 1.0 else f"{value:.0f}"
        draw.text((x0, max(margin_top - 24, visible_y0 - 28)), value_label, fill="#123138")

        label = str(row[label_column])
        draw.text((x0, plot_bottom + 24), ascii_labels.get(label, label), fill="#34474f")

    draw.line((margin_left, plot_bottom, width - margin_right, plot_bottom), fill="#34474f", width=3)
    image.save(output_path)


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def export_output_report(
    recognition: dict[str, Any],
    comparison_frame: pd.DataFrame,
    comparison: dict[str, set[str]],
    products: list[dict[str, Any]],
    manual_source: str,
    recipes_result: dict[str, Any] | None,
    recipe_filters: dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("outputs") / f"streamlit_report_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_quality_metrics(comparison, products)
    metrics_frame = make_quality_metrics_frame(metrics)
    metrics_chart = make_metrics_chart_frame(metrics)
    comparison_chart = make_comparison_chart_frame(comparison)
    breakdown_frame = make_breakdown_frame(comparison, metrics)
    recognition_frame = make_recognition_frame(products)
    recipes_frame = make_recipes_frame(recipes_result)
    conclusion = make_quality_conclusion(comparison, float(metrics["f1_score"]))

    save_frame(report_dir / "recognized_products_table.csv", recognition_frame)
    save_frame(report_dir / "manual_vs_model_table.csv", comparison_frame)
    save_frame(report_dir / "quality_metrics_table.csv", metrics_frame)
    save_frame(report_dir / "quality_breakdown_table.csv", breakdown_frame)
    save_frame(report_dir / "quality_metrics_chart_data.csv", metrics_chart)
    save_frame(report_dir / "comparison_chart_data.csv", comparison_chart)
    if not recipes_frame.empty:
        save_frame(report_dir / "recipes_table.csv", recipes_frame)

    save_bar_chart_image(
        metrics_chart,
        label_column="Метрика",
        value_column="Значение",
        title="Recognition quality metrics",
        output_path=report_dir / "quality_metrics_chart.png",
        max_value=1.0,
    )
    save_bar_chart_image(
        comparison_chart,
        label_column="Группа",
        value_column="Количество",
        title="Manual list vs model output",
        output_path=report_dir / "manual_vs_model_chart.png",
    )

    (report_dir / "conclusion.txt").write_text(conclusion, encoding="utf-8")
    save_json(
        report_dir / "report_summary.json",
        {
            "recognition": recognition,
            "manual_products": sorted(comparison["manual"]),
            "manual_products_source": manual_source,
            "comparison": {
                "matched": sorted(comparison["matched"]),
                "manual_only": sorted(comparison["manual"] - comparison["model"]),
                "model_only": sorted(comparison["model"] - comparison["manual"]),
            },
            "metrics": {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "top_3_hit_rate": metrics["top_3_hit_rate"],
                "top_3_products": metrics["top_3_products"],
                "top_3_matches": metrics["top_3_matches"],
            },
            "conclusion": conclusion,
            "recipes": recipes_result,
            "filters": recipe_filters,
        },
    )
    return report_dir


def render_quality_report(
    comparison: dict[str, set[str]],
    products: list[dict[str, Any]],
    recognition: dict[str, Any],
    comparison_frame: pd.DataFrame,
    manual_source: str,
    recipes_result: dict[str, Any] | None,
    recipe_filters: dict[str, Any],
) -> None:
    st.divider()
    st.subheader("Метрики качества")

    if not comparison["manual"]:
        st.info("Для расчёта precision, recall и F1 нужен ручной список продуктов.")
        return

    metrics = calculate_quality_metrics(comparison, products)
    score_columns = st.columns(4)
    score_columns[0].metric("Precision", f"{metrics['precision']:.1%}")
    score_columns[1].metric("Recall", f"{metrics['recall']:.1%}")
    score_columns[2].metric("F1 score", f"{metrics['f1_score']:.1%}")
    score_columns[3].metric("Top-3 hit rate", f"{metrics['top_3_hit_rate']:.1%}")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.write("**Основные метрики**")
        metrics_chart = make_metrics_chart_frame(metrics)
        st.bar_chart(metrics_chart, x="Метрика", y="Значение", y_label="Доля")

    with chart_right:
        st.write("**Сравнение списков**")
        compare_chart = make_comparison_chart_frame(comparison)
        st.bar_chart(compare_chart, x="Группа", y="Количество", y_label="Продукты")

    table_left, table_right = st.columns([1.08, 0.92], gap="large")
    with table_left:
        st.write("**Таблица метрик**")
        st.dataframe(make_quality_metrics_frame(metrics), use_container_width=True, hide_index=True)

    with table_right:
        st.write("**Группы продуктов**")
        breakdown_frame = make_breakdown_frame(comparison, metrics)
        st.dataframe(breakdown_frame, use_container_width=True, hide_index=True)

    st.info(make_quality_conclusion(comparison, float(metrics["f1_score"])))

    if st.button("Экспортировать таблицы и графики в outputs", type="primary"):
        report_dir = export_output_report(
            recognition=recognition,
            comparison_frame=comparison_frame,
            comparison=comparison,
            products=products,
            manual_source=manual_source,
            recipes_result=recipes_result,
            recipe_filters=recipe_filters,
        )
        st.success(f"Отчёт сохранён: {report_dir}")


def render_recipe(recipe: dict[str, Any], recipe_number: int) -> None:
    title = recipe.get("title", "Рецепт")
    time_minutes = recipe.get("time_minutes", "?")
    servings = recipe.get("servings", "-")
    amounts = recipe.get("ingredients_with_amounts", [])
    with st.expander(f"{recipe_number}. {title} - {time_minutes} мин", expanded=recipe_number == 1):
        details_left, details_right = st.columns(2)
        with details_left:
            st.write("**Сложность:**", recipe.get("difficulty", "-"))
            st.write("**Порций:**", servings)
            st.write("**Используются:**", ", ".join(recipe.get("ingredients_used", [])) or "-")
        with details_right:
            extra = recipe.get("extra_needed", [])
            st.write("**Дополнительно:**", ", ".join(extra) if extra else "не требуется")
            st.write("**Почему подходит:**", recipe.get("why_this_recipe", "") or "-")

        st.write(f"**Ингредиенты на {servings} порций:**", ", ".join(amounts) or "-")
        st.write("**Шаги приготовления:**")
        for step_number, step in enumerate(recipe.get("steps", []), start=1):
            st.write(f"{step_number}. {step}")


st.set_page_config(page_title="ЛР3: рецепты по фото продуктов", page_icon="🍽️", layout="wide")
apply_page_styles()

with st.sidebar:
    st.header("Параметры рецепта")
    recipe_count = st.slider("Количество рецептов", min_value=1, max_value=10, value=3)
    vegetarian = st.checkbox("Вегетарианское", value=False)
    max_time = st.slider("Время готовки, минут", 5, 120, 30, step=5)
    difficulty = st.selectbox("Сложность", ["легко", "средне", "сложно"])
    servings = st.number_input("Порций", min_value=1, max_value=10, value=2, step=1)
    st.caption("GROQ_API_KEY должен быть задан в .env")

active_recipe_filters = {
    "recipe_count": recipe_count,
    "vegetarian": vegetarian,
    "max_time": max_time,
    "difficulty": difficulty,
    "servings": servings,
}

st.title("Рецепты по фото продуктов")

upload_column, manual_column = st.columns([1.05, 0.95], gap="large")
with upload_column:
    st.subheader("Фото")
    uploaded = st.file_uploader(
        "Загрузите фото холодильника или продуктовой полки",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        label_visibility="collapsed",
    )

with manual_column:
    st.subheader("Продукты вручную")
    manual_products = st.text_area(
        "Продукты вручную",
        key="manual_products",
        height=150,
        placeholder="яйца, молоко, сыр\nпомидоры",
        label_visibility="collapsed",
    )
    st.caption("Пустое поле использует эталон из eval_expected.csv, если имя фото найдено.")

if uploaded is None:
    st.info("Загрузите фото, чтобы начать распознавание.")
    st.stop()

image_bytes = uploaded.getvalue()
image_signature = sha256(image_bytes).hexdigest()
if st.session_state.get("image_signature") != image_signature:
    st.session_state["image_signature"] = image_signature
    st.session_state.pop("recognition", None)
    st.session_state.pop("recipes", None)
    st.session_state.pop("recipe_filters", None)

suffix = Path(uploaded.name).suffix or ".jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(image_bytes)
    image_path = Path(tmp.name)

st.divider()
st.subheader("Распознавание")

photo_column, recognition_column = st.columns([0.9, 1.1], gap="large")
with photo_column:
    st.image(str(image_path), use_container_width=True)

with recognition_column:
    if st.button("Распознать продукты", type="primary", use_container_width=True):
        with st.spinner("Распознаю продукты на фото..."):
            st.session_state["recognition"] = recognize_products(image_path)
            st.session_state.pop("recipes", None)
            st.session_state.pop("recipe_filters", None)

    recognition = st.session_state.get("recognition")
    if recognition:
        products = recognition.get("products", [])
        average_confidence = (
            sum(float(product.get("confidence", 0.0)) for product in products) / len(products)
            if products
            else 0.0
        )
        missed_products = recognition.get("possible_missed", [])
        model_metrics = st.columns(3)
        model_metrics[0].metric("Найдено", len(products))
        model_metrics[1].metric("Средняя уверенность", f"{average_confidence:.0%}" if products else "-")
        model_metrics[2].metric("Возможные пропуски", len(missed_products))

        recognition_frame = make_recognition_frame(products)
        if recognition_frame.empty:
            st.warning("Модель не нашла продукты на фото.")
        else:
            st.dataframe(recognition_frame, use_container_width=True, hide_index=True)

        if recognition.get("summary"):
            st.write("**Описание:**", recognition["summary"])
        if missed_products:
            st.write("**Возможные пропуски:**", ", ".join(missed_products))
    else:
        st.info("Результат распознавания появится здесь.")

if not recognition:
    st.stop()

products = recognition.get("products", [])
comparison_manual_products, comparison_source = choose_manual_products(manual_products, uploaded.name)
comparison_frame, comparison = compare_products(comparison_manual_products, products)

st.divider()
st.subheader("Сравнение списков")
comparison_metrics = st.columns(4)
comparison_metrics[0].metric("Вручную", len(comparison["manual"]))
comparison_metrics[1].metric("Модель", len(comparison["model"]))
comparison_metrics[2].metric("Совпадения", len(comparison["matched"]))
comparison_metrics[3].metric(
    "Покрытие ручного списка",
    f"{len(comparison['matched']) / len(comparison['manual']):.0%}" if comparison["manual"] else "-",
)

if not comparison["manual"]:
    st.info(
        "Введите список продуктов вручную или загрузите файл с именем из "
        "data/eval_expected.csv, чтобы увидеть сравнение с распознаванием."
    )
elif manual_products.strip():
    st.caption("Источник ручного списка: ручной ввод.")
else:
    st.caption(f"Источник ручного списка: {comparison_source}.")

if comparison_frame.empty:
    st.warning("Списки продуктов пока пусты.")
else:
    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

st.divider()
recipes_header, recipes_action = st.columns([0.68, 0.32], gap="large")
with recipes_header:
    st.subheader("Рецепты")
with recipes_action:
    generate_clicked = st.button("Сгенерировать рецепты", use_container_width=True)

if generate_clicked:
    with st.spinner("Генерирую рецепты..."):
        st.session_state["recipes"] = generate_recipes(
            products,
            vegetarian=vegetarian,
            max_time=max_time,
            difficulty=difficulty,
            servings=servings,
            recipe_count=recipe_count,
        )
        st.session_state["recipe_filters"] = {
            **active_recipe_filters,
        }

recipes_result = st.session_state.get("recipes")
report_filters = st.session_state.get("recipe_filters", active_recipe_filters)
if recipes_result:
    recipes = recipes_result.get("recipes", [])
    result_metrics = st.columns(4)
    result_metrics[0].metric("Запрошено рецептов", report_filters["recipe_count"])
    result_metrics[1].metric("Получено рецептов", len(recipes))
    result_metrics[2].metric("Сложность", report_filters["difficulty"])
    result_metrics[3].metric("Порций", report_filters["servings"])

    if recipes:
        for number, recipe in enumerate(recipes, start=1):
            render_recipe(recipe, number)
    else:
        st.warning("Модель не вернула рецепты.")

    if recipes_result.get("comment"):
        st.info(recipes_result["comment"])

render_quality_report(
    comparison=comparison,
    products=products,
    recognition=recognition,
    comparison_frame=comparison_frame,
    manual_source=comparison_source,
    recipes_result=recipes_result,
    recipe_filters=report_filters,
)
