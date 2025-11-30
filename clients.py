import requests
from typing import Optional, Dict, Any, Union, List 
import json
import os
import re


def _normalize_query(q: str) -> str:
    q = q.replace('_', ' ').replace('-', ' ')
    q = q.lower().strip()
    q = re.sub(r'[^a-z0-9\s]', '', q)
    return q


class OpenFoodFactsClient:
    BASE = 'https://world.openfoodfacts.org/cgi/search.pl'
    CACHE_PATH = "nutrition_cache.json"

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self._cache = {}
        try:
            if os.path.exists(self.CACHE_PATH):
                with open(self.CACHE_PATH, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                    print(f"Loaded {len(self._cache)} items from cache")
        except Exception as e:
            print(f"Cache loading error (continuing with empty cache): {e}")
            self._cache = {}

    def _choose_search_term(self, query: str, synonyms: dict = None) -> str:  # ✅ Cambiar | por Union
        nq = _normalize_query(query)
        if synonyms:
            for k, vals in synonyms.items():
                if nq == _normalize_query(k):
                    return vals[0]
                for v in vals:
                    if nq == _normalize_query(v):
                        return vals[0]
        return nq

    def _save_cache(self):
        try:
            with open(self.CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    
    def search(self, query: str, page_size: int = 6, synonyms: dict = None) -> Optional[Dict[str, Any]]:
        term = self._choose_search_term(query, synonyms)
        params = {
            "search_terms": term,
            "search_simple": 1,
            "json": 1,
            "page_size": page_size,
        }
        try:
            r = requests.get(self.BASE, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data
        except requests.exceptions.RequestException as e:
            print(f"OpenFoodFacts API error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def get_best_nutriments(self, query: str) -> Optional[Dict[str, Any]]:
        nq = _normalize_query(query)
        if nq in self._cache:
            return self._cache[nq]

        data = self.search(query, synonyms=getattr(self, '_synonyms', None))
        if not data:
            return None
        products = data.get("products", [])
        if not products:
            return None

        for prod in products:
            nutriments = prod.get("nutriments")
            if nutriments:
                result = {
                    "product_name": prod.get("product_name", prod.get("generic_name", "")),
                    "brands": prod.get("brands"),
                    "nutriments": nutriments,
                    "nutrient_levels": prod.get("nutrient_levels"),
                    "serving_size": prod.get("serving_size"),
                }
                self._cache[nq] = result
                self._save_cache()
                return result

        self._cache[nq] = None
        self._save_cache()
        return None


class NutritionProvider:
    def __init__(self, synonyms: dict = None):
        self.synonyms = synonyms
        self.off = OpenFoodFactsClient()

    def get_nutrition(self, query: str):
        # use Open Food Facts
        if self.synonyms:
            self.off._synonyms = self.synonyms
        res = self.off.get_best_nutriments(query)
        return {"provider": "openfoodfacts", "result": res}

    def get_nutrition_for_labels(self, labels: list) -> dict:
        candidates = []
        for lab in labels:
            norm = _normalize_query(lab)
            candidates.append(norm)
            if self.synonyms:
                key = norm
                for k, vals in self.synonyms.items():
                    if _normalize_query(k) == key:
                        for v in vals:
                            candidates.append(_normalize_query(v))
                        break

        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        for q in unique_candidates:
            res = self.off.get_best_nutriments(q)
            if res:
                return {"provider": "openfoodfacts", "query": q, "result": res}

        return {"provider": None, "query": None, "result": None}

