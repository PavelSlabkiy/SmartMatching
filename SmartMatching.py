from rapidfuzz import fuzz
from datetime import date
import json
import heapq
import sys
import re

class SmartMatching:
    '''
    SmartMatching — поиск похожих людей во всех деревьях.
    Теперь возвращает список совпадений, и для каждого совпадения формируется отдельный people-фрагмент.
    '''
    def __init__(self, data, database, trashhold: int = 90, k: int = 1):
        self.data = data
        self.database = database
        self.trashhold = trashhold
        self.k = k

    # ----------------------------------------------------------------------

    def compare_idx2idx(self, idx1: dict, idx2: dict) -> float:
        # ---------- helpers (local) ----------

        def normalize_text(s: str) -> str:
            if not s:
                return ""
            s = s.lower()
            s = re.sub(r"[^\w\s]", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def text_similarity(a: str, b: str) -> int:
            if not a or not b:
                return 70
            return fuzz.token_sort_ratio(normalize_text(a), normalize_text(b))

        def parse_date_range(d: str):
            if not d:
                return None
            parts = d.split("-")
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

        def date_similarity(d1: str, d2: str) -> int:
            if not d1 or not d2:
                return 50

            r1 = parse_date_range(d1)
            r2 = parse_date_range(d2)

            if not r1 or not r2:
                return 50

            start1, end1 = r1
            start2, end2 = r2

            if start1 <= end2 and start2 <= end1:
                return 100

            if abs(start1.year - start2.year) <= 1:
                return 70

            return 0

        # ---------- new birthPlace similarity ----------

        def place_similarity(str1: str, str2: str) -> int:
            STOP_WORDS = {
                "г", "город", "с", "село", "деревня", "пос", "поселок",
                "рн", "район", "обл", "область", "край",
                "республика", "уезд", "волость"
            }

            def normalize(text: str) -> list[str]:
                if not text:
                    return []

                text = text.lower()
                text = re.sub(r"[^\w\s]", " ", text)
                tokens = text.split()

                seen = set()
                result = []
                for t in tokens:
                    if len(t) <= 2 or t in STOP_WORDS:
                        continue
                    if t not in seen:
                        seen.add(t)
                        result.append(t)
                return result

            def trigrams(s: str) -> set[str]:
                s = f"  {s} "
                return {s[i:i + 3] for i in range(len(s) - 2)}

            def trigram_jaccard(a: str, b: str) -> float:
                ta = trigrams(a)
                tb = trigrams(b)
                if not ta or not tb:
                    return 0.0
                return len(ta & tb) / len(ta | tb)

            tokens1 = normalize(str1)
            tokens2 = normalize(str2)

            if not tokens1 or not tokens2:
                return 50

            if len(tokens1) <= len(tokens2):
                short_tokens, long_tokens = tokens1, tokens2
            else:
                short_tokens, long_tokens = tokens2, tokens1

            long_text = " ".join(long_tokens)

            scores = []
            for token in short_tokens:
                scores.append(trigram_jaccard(token, long_text))

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

        # ---------- weighted scoring ----------

        score = 0.0
        weight_sum = 0.0

        def add(part_score, weight):
            nonlocal score, weight_sum
            score += part_score * weight
            weight_sum += weight

        # Фамилия
        add(text_similarity(idx1.get("lastName"), idx2.get("lastName")), 0.25)

        # Имя
        add(text_similarity(idx1.get("name"), idx2.get("name")), 0.20)

        # Отчество
        if idx1.get("middleName") or idx2.get("middleName"):
            add(text_similarity(idx1.get("middleName"), idx2.get("middleName")), 0.10)

        # Дата рождения
        add(date_similarity(idx1.get("birthDate"), idx2.get("birthDate")), 0.25)

        # Место рождения (улучшенное сравнение)
        add(place_similarity(idx1.get("birthPlace"), idx2.get("birthPlace")), 0.10)

        # Пол
        if idx1.get("gender") and idx2.get("gender"):
            add(100 if idx1["gender"] == idx2["gender"] else 0, 0.10)

        # Статус жизни
        if idx1.get("isAlive") is not None and idx2.get("isAlive") is not None:
            add(100 if str(idx1["isAlive"]) == str(idx2["isAlive"]) else 0, 0.05)

        if weight_sum == 0:
            return 0.0

        return score / weight_sum

    # ----------------------------------------------------------------------
    def get_oldest_generation_idx(self):
        data_json = json.loads(self.data)
        oldest_idx = []
        for person_id, person in data_json["people"].items():
            if person.get("fatherId") is None and person.get("motherId") is None:
                oldest_idx.append(person_id)
        return oldest_idx

    # ----------------------------------------------------------------------
    # Поиск совпадений во всех деревьях
    # ----------------------------------------------------------------------
    def parse_json(self):
        data_json = json.loads(self.data)
        database_json = json.loads(self.database)
        oldest_idx = self.get_oldest_generation_idx()

        scores_dict = {}

        for data_idx in oldest_idx:
            scores_list = []

            for tree_id, tree_data in database_json.get("tree_id", {}).items():
                people = tree_data.get("people", {})

                for db_id, db_person in people.items():

                    score = self.compare_idx2idx(data_json["people"][data_idx], db_person)
                    if score >= self.trashhold:
                        scores_list.append({
                            "data_id": data_idx,
                            "tree_id": tree_id,
                            "tree_owner": tree_data.get("tree_owner"),
                            "database_id": db_id,
                            "score": score
                        })

            scores_dict[data_idx] = scores_list

        return scores_dict

    # ----------------------------------------------------------------------
    def top_k_idx(self):
        k = self.k
        scores_dict = self.parse_json()

        pairs = [
            entry
            for _, entries in scores_dict.items()
            for entry in entries
        ]

        top_k = heapq.nlargest(k, pairs, key=lambda x: x["score"])
        return top_k

    # ----------------------------------------------------------------------
    # Формируем отдельный people-фрагмент для КАЖДОГО совпадения
    # ----------------------------------------------------------------------
    def get_older_generation_idx(self):
        top = self.top_k_idx()
        database_json = json.loads(self.database)

        matchedDataIds = list({t["data_id"] for t in top})

        results = []

        for match in top:
            tree_id = match["tree_id"]
            db_person_id = match["database_id"]

            people = database_json["tree_id"][tree_id]["people"]

            # если по какой-то причине нет — пропускаем
            if db_person_id not in people:
                continue

            # собираем предков именно для этого совпадения
            fragment_people = {}

            def collect_ancestors(pid):
                if pid is None or pid not in people:
                    return
                if pid in fragment_people:
                    return
                person = people[pid]
                fragment_people[pid] = person
                collect_ancestors(person.get("fatherId"))
                collect_ancestors(person.get("motherId"))

            collect_ancestors(db_person_id)

            # добавляем в общий список
            results.append({
                **match,
                "people": fragment_people
            })

        return {
            "matches": results,
            "matchedDataIds": matchedDataIds
        }


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    payload = sys.stdin.read()
    if not payload:
        print(json.dumps({}))
        sys.exit(0)

    obj = json.loads(payload)
    data = obj.get("data")
    db = obj.get("db")

    SM = SmartMatching(data, db, trashhold=90, k=5)
    out = SM.get_older_generation_idx()

    print(json.dumps(out))
