import argparse
import json
from pathlib import Path

from SmartMatching import SmartMatching


def read_json(path: Path) -> str:
    """Читает JSON-файл дерева и возвращает его как строку."""
    return path.read_text(encoding="utf-8")


def main() -> None:
    """Точка входа CLI для поиска лучшего совпадения между двумя деревьями."""
    parser = argparse.ArgumentParser(
        description="Поиск лучшего совпадения персон между двумя деревьями import.json."
    )
    parser.add_argument("--tree1", required=True, help="Путь к первому дереву.")
    parser.add_argument("--tree2", required=True, help="Путь ко второму дереву.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=90,
        help="Порог совпадения (по умолчанию: 90).",
    )
    args = parser.parse_args()

    tree1_raw = read_json(Path(args.tree1))
    tree2_raw = read_json(Path(args.tree2))

    matcher = SmartMatching(tree1=tree1_raw, tree2=tree2_raw, threshold=args.threshold)
    result = matcher.get_best_match()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
