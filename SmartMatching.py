from datetime import date
import json
import re

from rapidfuzz import fuzz


class SmartMatching:
    """
    Алгоритм поиска лучшего совпадения персон между двумя генеалогическими древами.

    Ожидаемый вход:
    - два дерева в формате `import.json` (каждое дерево — JSON-массив персон);
    - каждый элемент массива содержит поля `_id`, `treeId`, `name`, `surname`,
      `middleName`, `maidenName`, `birthdate`, `birthplace` (часть полей может отсутствовать).

    Выход:
    - один лучший мэтч (dict), если его score >= threshold;
    - пустой dict, если ни одна пара персон не прошла порог.
    """

    def __init__(self, tree1, tree2, threshold: int = 90):
        """
        Инициализирует сравнение двух деревьев.

        :param tree1: первое дерево в формате import.json (str или list[dict]).
        :param tree2: второе дерево в формате import.json (str или list[dict]).
        :param threshold: минимальный score пары, чтобы считаться валидным мэтчем.
        """
        self.threshold = threshold
        self.tree1 = self._parse_tree(tree1)
        self.tree2 = self._parse_tree(tree2)

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Нормализует строку для fuzzy-сравнения."""
        if not value:
            return ""
        value = value.lower()
        value = re.sub(r"[^\w\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _text_similarity(a: str, b: str) -> int:
        """Считает текстовую похожесть двух полей в шкале 0..100."""
        if not a or not b:
            return 70
        return fuzz.token_sort_ratio(
            SmartMatching._normalize_text(a),
            SmartMatching._normalize_text(b),
        )

    @staticmethod
    def _extract_oid(value):
        """Извлекает строковый id из Mongo-представления {'$oid': ...}."""
        if isinstance(value, dict):
            return value.get("$oid")
        return value

    @staticmethod
    def _first_value(value):
        """Берет первый элемент массива значений или возвращает само значение."""
        if isinstance(value, list):
            if not value:
                return None
            return value[0]
        return value

    @staticmethod
    def _birthdate_to_string(raw_birthdate):
        """
        Приводит birthdate из import.json к строке формата:
        YYYY, YYYY-MM или YYYY-MM-DD.
        """
        birth = SmartMatching._first_value(raw_birthdate)
        if not isinstance(birth, dict):
            return ""

        year = birth.get("year")
        month = birth.get("month")
        day = birth.get("day")

        if not year:
            return ""
        if not month:
            return f"{year}"
        if not day:
            return f"{year}-{month:02d}"
        return f"{year}-{month:02d}-{day:02d}"

    @staticmethod
    def _parse_date_range(value: str):
        """Преобразует частичную дату в диапазон дат для мягкого сравнения."""
        if not value:
            return None
        parts = value.split("-")
        try:
            year = int(parts[0])
            if len(parts) == 1:
                return date(year, 1, 1), date(year, 12, 31)
            month = int(parts[1])
            if len(parts) == 2:
                return date(year, month, 1), date(year, month, 28)
            day = int(parts[2])
            return date(year, month, day), date(year, month, day)
        except Exception:
            return None

    @staticmethod
    def _date_similarity(d1: str, d2: str) -> int:
        """Считает похожесть дат рождения с учетом неполных дат."""
        if not d1 or not d2:
            return 50

        r1 = SmartMatching._parse_date_range(d1)
        r2 = SmartMatching._parse_date_range(d2)
        if not r1 or not r2:
            return 50

        start1, end1 = r1
        start2, end2 = r2

        if start1 <= end2 and start2 <= end1:
            return 100
        if abs(start1.year - start2.year) <= 1:
            return 70
        return 0

    @staticmethod
    def _place_similarity(place1: str, place2: str) -> int:
        """Сравнивает места рождения по токенам и триграммам."""
        stop_words = {
            "г",
            "город",
            "с",
            "село",
            "деревня",
            "пос",
            "поселок",
            "рн",
            "район",
            "обл",
            "область",
            "край",
            "республика",
            "уезд",
            "волость",
        }

        def normalize(text: str):
            """Нормализация топонима: чистка и удаление служебных токенов."""
            if not text:
                return []
            text = SmartMatching._normalize_text(text)
            seen = set()
            out = []
            for token in text.split():
                if len(token) <= 2 or token in stop_words:
                    continue
                if token not in seen:
                    out.append(token)
                    seen.add(token)
            return out

        def trigrams(text: str):
            """Строит множество триграмм строки."""
            text = f"  {text} "
            return {text[i : i + 3] for i in range(len(text) - 2)}

        def trigram_jaccard(a: str, b: str):
            """Коэффициент Жаккара по триграммам."""
            ta = trigrams(a)
            tb = trigrams(b)
            if not ta or not tb:
                return 0.0
            return len(ta & tb) / len(ta | tb)

        tokens1 = normalize(place1)
        tokens2 = normalize(place2)
        if not tokens1 or not tokens2:
            return 50

        short_tokens, long_tokens = (
            (tokens1, tokens2) if len(tokens1) <= len(tokens2) else (tokens2, tokens1)
        )
        long_text = " ".join(long_tokens)
        scores = [trigram_jaccard(token, long_text) for token in short_tokens]
        if not scores:
            return 0

        best = max(scores)
        avg = sum(scores) / len(scores)
        final = best * 0.7 + avg * 0.3
        if final >= 0.75:
            return 100
        if final >= 0.55:
            return 80
        if final >= 0.35:
            return 60
        if final >= 0.2:
            return 40
        return 0

    @staticmethod
    def _person_to_index(person: dict) -> dict:
        """Приводит запись персоны к плоскому индексу полей для сравнения."""
        return {
            "name": SmartMatching._first_value(person.get("name")) or "",
            "middleName": SmartMatching._first_value(person.get("middleName")) or "",
            "lastName": SmartMatching._first_value(person.get("surname")) or "",
            "maidenName": SmartMatching._first_value(person.get("maidenName")) or "",
            "birthDate": SmartMatching._birthdate_to_string(person.get("birthdate")),
            "birthPlace": SmartMatching._first_value(person.get("birthplace")) or "",
        }

    @staticmethod
    def _parse_tree(raw_tree):
        """
        Парсит одно дерево в формате import.json.

        Поддерживаемый формат: JSON-массив персон (строка JSON или list[dict]).
        Валидация:
        - у каждой персоны должен быть `_id`;
        - у каждой персоны должен быть `treeId`;
        - все персоны должны принадлежать одному `treeId`.
        """
        if isinstance(raw_tree, str):
            raw_tree = json.loads(raw_tree)

        if not isinstance(raw_tree, list):
            raise ValueError("Ожидается массив персон (формат import.json).")

        people = {}
        tree_id = None
        for person in raw_tree:
            person_id = SmartMatching._extract_oid(person.get("_id"))
            if not person_id:
                raise ValueError("У персоны отсутствует поле _id.")

            person_tree_id = SmartMatching._extract_oid(person.get("treeId"))
            if not person_tree_id:
                raise ValueError(f"У персоны {person_id} отсутствует поле treeId.")

            if tree_id is None:
                tree_id = person_tree_id
            elif tree_id != person_tree_id:
                raise ValueError("Во входном массиве обнаружены персоны из разных деревьев.")

            people[str(person_id)] = person

        if tree_id is None:
            raise ValueError("Пустое дерево: не найдено ни одной персоны.")

        return {"id": tree_id, "people": people}

    def compare_idx2idx(self, idx1: dict, idx2: dict) -> float:
        """
        Рассчитывает итоговый score совпадения двух персон.

        Важно:
        - пол и статус жизни в скоринге не участвуют;
        - девичья фамилия учитывается только если присутствует у обеих персон.
        """
        score = 0.0
        weight_sum = 0.0

        def add(part_score, weight):
            nonlocal score, weight_sum
            score += part_score * weight
            weight_sum += weight

        add(self._text_similarity(idx1.get("lastName"), idx2.get("lastName")), 0.28)
        add(self._text_similarity(idx1.get("name"), idx2.get("name")), 0.24)

        if idx1.get("middleName") or idx2.get("middleName"):
            add(self._text_similarity(idx1.get("middleName"), idx2.get("middleName")), 0.10)

        add(self._date_similarity(idx1.get("birthDate"), idx2.get("birthDate")), 0.28)
        add(self._place_similarity(idx1.get("birthPlace"), idx2.get("birthPlace")), 0.10)

        # Девичья фамилия учитывается только при наличии в обеих записях.
        if idx1.get("maidenName") and idx2.get("maidenName"):
            add(self._text_similarity(idx1.get("maidenName"), idx2.get("maidenName")), 0.12)

        if weight_sum == 0:
            return 0.0
        return score / weight_sum

    def get_best_match(self) -> dict:
        """
        Ищет лучшую пару персон между двумя деревьями.

        :return: dict формата:
            {
              "score": float,
              "tree1": {"id": str, "personId": str, "person": {...}},
              "tree2": {"id": str, "personId": str, "person": {...}}
            }
            или {} если подходящих пар нет.
        """
        best = None
        people1 = self.tree1["people"]
        people2 = self.tree2["people"]

        for person1_id, person1 in people1.items():
            idx1 = self._person_to_index(person1)
            for person2_id, person2 in people2.items():
                idx2 = self._person_to_index(person2)
                score = self.compare_idx2idx(idx1, idx2)
                if score < self.threshold:
                    continue
                if best is None or score > best["score"]:
                    best = {
                        "score": round(score, 2),
                        "tree1": {
                            "id": self.tree1["id"],
                            "personId": person1_id,
                            "person": person1,
                        },
                        "tree2": {
                            "id": self.tree2["id"],
                            "personId": person2_id,
                            "person": person2,
                        },
                    }

        return best or {}
