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
    - все мэтчи со score >= threshold (от лучшего к худшему) + метаданные.
    """

    def __init__(self, tree1, tree2, threshold: int = 90, exact_match: bool = True):
        """
        Инициализирует сравнение двух деревьев.

        :param tree1: первое дерево в формате import.json (str или list[dict]).
        :param tree2: второе дерево в формате import.json (str или list[dict]).
        :param threshold: минимальный score пары, чтобы считаться валидным мэтчем.
        :param exact_match: режим строгого сравнения ключевых полей.
        """
        self.threshold = threshold
        self.exact_match = exact_match
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
    def _date_parts(value: str):
        """
        Возвращает кортеж частей даты (year, month, day).

        Для пустых/некорректных значений возвращает (None, None, None).
        """
        if not value:
            return None, None, None
        parts = value.split("-")
        try:
            year = int(parts[0]) if len(parts) >= 1 else None
            month = int(parts[1]) if len(parts) >= 2 else None
            day = int(parts[2]) if len(parts) >= 3 else None
            return year, month, day
        except Exception:
            return None, None, None

    @staticmethod
    def _date_exact_match(d1: str, d2: str) -> bool:
        """
        Проверяет exact match дат по правилу минимальной точности.

        Правила:
        - сравнение возможно только если в обеих датах есть год;
        - если хотя бы в одной дате задан только год, сравнивается только год;
        - если хотя бы в одной дате задан год+месяц, сравниваются год и месяц;
        - если обе даты полные, сравниваются год, месяц и день.
        """
        y1, m1, day1 = SmartMatching._date_parts(d1)
        y2, m2, day2 = SmartMatching._date_parts(d2)

        if not y1 or not y2:
            return False
        if y1 != y2:
            return False

        precision1 = 1 + int(m1 is not None) + int(day1 is not None)
        precision2 = 1 + int(m2 is not None) + int(day2 is not None)
        min_precision = min(precision1, precision2)

        if min_precision >= 2 and m1 != m2:
            return False
        if min_precision >= 3 and day1 != day2:
            return False
        return True

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

    def _is_exact_candidate(self, idx1: dict, idx2: dict) -> bool:
        """
        Проверяет, есть ли смысл сравнивать пару в exact-режиме.

        Требования:
        - фамилия, имя, отчество, дата и место рождения должны быть непустыми в обеих записях;
        - фамилия, имя и отчество должны совпадать строго (с нормализацией);
        - дата рождения должна совпадать по правилам exact-сравнения;
        - девичья фамилия сравнивается строго только при наличии в обеих записях.
        """
        required_fields = ("lastName", "name", "middleName", "birthDate", "birthPlace")
        for field in required_fields:
            if not idx1.get(field) or not idx2.get(field):
                return False

        if self._normalize_text(idx1["lastName"]) != self._normalize_text(idx2["lastName"]):
            return False
        if self._normalize_text(idx1["name"]) != self._normalize_text(idx2["name"]):
            return False
        if self._normalize_text(idx1["middleName"]) != self._normalize_text(idx2["middleName"]):
            return False

        if not self._date_exact_match(idx1["birthDate"], idx2["birthDate"]):
            return False

        maiden1 = idx1.get("maidenName")
        maiden2 = idx2.get("maidenName")
        if maiden1 and maiden2:
            if self._normalize_text(maiden1) != self._normalize_text(maiden2):
                return False

        return True

    def compare_idx2idx(self, idx1: dict, idx2: dict) -> float:
        """
        Рассчитывает итоговый score совпадения двух персон.

        Важно:
        - пол и статус жизни в скоринге не участвуют;
        - девичья фамилия учитывается только если присутствует у обеих персон.
        - в exact-режиме несовпавшие/пустые ключевые поля отсекают пару сразу.
        """
        if self.exact_match and not self._is_exact_candidate(idx1, idx2):
            return 0.0

        score = 0.0
        weight_sum = 0.0

        def add(part_score, weight):
            nonlocal score, weight_sum
            score += part_score * weight
            weight_sum += weight

        if self.exact_match:
            add(100, 0.28)  # Фамилия уже прошла exact-match
            add(100, 0.24)  # Имя уже прошло exact-match
        else:
            add(self._text_similarity(idx1.get("lastName"), idx2.get("lastName")), 0.28)
            add(self._text_similarity(idx1.get("name"), idx2.get("name")), 0.24)

        if idx1.get("middleName") or idx2.get("middleName"):
            if self.exact_match:
                add(100, 0.10)  # Отчество уже прошло exact-match
            else:
                add(self._text_similarity(idx1.get("middleName"), idx2.get("middleName")), 0.10)

        if self.exact_match:
            add(100, 0.28)  # Дата уже прошла exact-match
        else:
            add(self._date_similarity(idx1.get("birthDate"), idx2.get("birthDate")), 0.28)
        add(self._place_similarity(idx1.get("birthPlace"), idx2.get("birthPlace")), 0.10)

        # Девичья фамилия учитывается только при наличии в обеих записях.
        if idx1.get("maidenName") and idx2.get("maidenName"):
            if self.exact_match:
                add(100, 0.12)  # Девичья фамилия уже прошла exact-match
            else:
                add(self._text_similarity(idx1.get("maidenName"), idx2.get("maidenName")), 0.12)

        if weight_sum == 0:
            return 0.0
        return score / weight_sum

    def get_matches(self) -> dict:
        """
        Ищет все пары персон между двумя деревьями, прошедшие threshold.

        :return: dict формата:
            {
              "tree1Id": str,
              "tree2Id": str,
              "tree1Size": int,
              "tree2Size": int,
              "matchesCount": int,
              "matches": [
                {
                  "score": float,
                  "tree1": {"id": str, "personId": str, "person": {...}},
                  "tree2": {"id": str, "personId": str, "person": {...}}
                },
                ...
              ]
            }
        """
        matches = []
        people1 = self.tree1["people"]
        people2 = self.tree2["people"]

        for person1_id, person1 in people1.items():
            idx1 = self._person_to_index(person1)
            for person2_id, person2 in people2.items():
                idx2 = self._person_to_index(person2)
                score = self.compare_idx2idx(idx1, idx2)
                if score < self.threshold:
                    continue
                matches.append(
                    {
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
                )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return {
            "tree1Id": self.tree1["id"],
            "tree2Id": self.tree2["id"],
            "tree1Size": len(people1),
            "tree2Size": len(people2),
            "matchesCount": len(matches),
            "matches": matches,
        }
