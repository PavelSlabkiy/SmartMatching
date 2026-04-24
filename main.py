import argparse
import json
from pathlib import Path

from SmartMatching import SmartMatching


def read_json(path: Path) -> str:
    """Читает JSON-файл дерева и возвращает его как строку."""
    return path.read_text(encoding="utf-8")


def main() -> None:
    """Точка входа CLI для поиска всех совпадений между двумя деревьями."""
    parser = argparse.ArgumentParser(
        description="Поиск всех совпадений персон между двумя деревьями import.json."
    )
    parser.add_argument("--tree1", required=True, help="Путь к первому дереву.")
    parser.add_argument("--tree2", required=True, help="Путь ко второму дереву.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=90,
        help="Порог совпадения (по умолчанию: 90).",
    )
    parser.add_argument(
        "--exact-match",
        type=lambda value: str(value).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Режим exact match (по умолчанию: true). Значения: true/false.",
    )
    args = parser.parse_args()

    tree1_raw = read_json(Path(args.tree1))
    tree2_raw = read_json(Path(args.tree2))

    matcher = SmartMatching(
        tree1=tree1_raw,
        tree2=tree2_raw,
        threshold=args.threshold,
        exact_match=args.exact_match,
    )
    result = matcher.get_matches()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
