import argparse
import json
from pathlib import Path

from SmartMatching import SmartMatching


def _read_json_file(path: Path) -> str:
    """
    Читает JSON-файл и возвращает его содержимое строкой.

    Возвращаем строку, чтобы парсинг и валидация были централизованы
    внутри класса SmartMatching.
    """
    return path.read_text(encoding="utf-8")


def main() -> None:
    """
    Точка входа тестового скрипта.

    Пример запуска:
    python3 run_match_test.py --tree1 ./import.json --tree2 ./import.json
    """
    parser = argparse.ArgumentParser(
        description="Тестовый запуск алгоритма SmartMatching для двух деревьев import.json."
    )
    parser.add_argument(
        "--tree1",
        required=True,
        help="Путь к первому дереву (JSON-массив персон формата import.json).",
    )
    parser.add_argument(
        "--tree2",
        required=True,
        help="Путь ко второму дереву (JSON-массив персон формата import.json).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=90,
        help="Порог score для принятия совпадения (по умолчанию: 90).",
    )
    args = parser.parse_args()

    tree1_path = Path(args.tree1)
    tree2_path = Path(args.tree2)

    tree1_raw = _read_json_file(tree1_path)
    tree2_raw = _read_json_file(tree2_path)

    matcher = SmartMatching(tree1=tree1_raw, tree2=tree2_raw, threshold=args.threshold)
    result = matcher.get_best_match()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
