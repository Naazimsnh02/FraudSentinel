"""
Tier-1 Card Fraud Scorer
Wraps cc_lgbm_model.txt + cc_lgbm_preproc.joblib from models/card_fraud/.
"""
import json, math, os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

# Default path — can override with env var CARD_SCORER_DIR
_DEFAULT_DIR = Path(__file__).parent.parent.parent / "models" / "card_fraud"
MODEL_DIR = Path(os.getenv("CARD_SCORER_DIR", _DEFAULT_DIR))

R = 6371.0
def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


class CardScorer:
    """
    score(tx_dict) → dict with risk_score, risk_level, route_to_llm, top_signals
    """

    # routing threshold from cc_lgbm_metrics.json (recall-0.90)
    DEFAULT_THRESHOLD = 0.9403985330442168

    FEATURES = [
        "log_amt", "amt", "hour", "dow", "is_night", "age", "geo_km",
        "log_city_pop", "mins_since_last", "tx_24h", "amt_24h", "tx_1h",
        "amt_over_p95", "amt_to_p95", "cat_fraud_rate",
        "category", "gender", "state",
    ]
    CAT_COLS = ["category", "gender", "state"]

    def __init__(self, model_dir: Path = MODEL_DIR):
        model_path  = model_dir / "cc_lgbm_model.txt"
        preproc_path = model_dir / "cc_lgbm_preproc.joblib"
        metrics_path = model_dir / "cc_lgbm_metrics.json"

        self.model   = lgb.Booster(model_file=str(model_path))
        preproc      = joblib.load(preproc_path)
        self.cat_p95 = preproc["cat_p95"]
        self.cat_rate = preproc["cat_rate"]

        # Load routing threshold from metrics file
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            self.threshold = m.get("routing_threshold", self.DEFAULT_THRESHOLD)
        else:
            self.threshold = self.DEFAULT_THRESHOLD

    def _featurize(self, tx: dict[str, Any]) -> pd.DataFrame:
        """Convert a raw transaction dict to a single-row feature DataFrame."""
        amt      = float(tx.get("amt", tx.get("amount", 0)))
        category = str(tx.get("category", "misc_net"))
        gender   = str(tx.get("gender", "M"))
        state    = str(tx.get("state", "CA"))
        hour     = int(tx.get("hour", 12))
        dow      = int(tx.get("dow", 0))
        lat      = float(tx.get("lat", 0))
        lon      = float(tx.get("long", tx.get("lon", 0)))
        m_lat    = float(tx.get("merch_lat", lat))
        m_lon    = float(tx.get("merch_long", tx.get("merch_lon", lon)))
        city_pop = float(tx.get("city_pop", 50000))
        dob_year = int(tx.get("dob_year", 1980))
        tx_24h   = int(tx.get("tx_24h", 0))
        amt_24h  = float(tx.get("amt_24h", amt))
        tx_1h    = int(tx.get("tx_1h", 0))
        mins_since_last = float(tx.get("mins_since_last", 99999))

        # Derived
        is_night  = 1 if (hour >= 22 or hour <= 4) else 0
        age       = max(0.0, 2025 - dob_year)
        geo_km    = _haversine(lat, lon, m_lat, m_lon)
        log_amt   = math.log1p(amt)
        log_city_pop = math.log1p(city_pop)

        p95 = self.cat_p95.get(category, np.median(list(self.cat_p95.values())))
        amt_over_p95 = 1 if amt > p95 else 0
        amt_to_p95   = amt / (p95 + 1e-6)
        cat_fraud_rate = self.cat_rate.get(category, 0.0)

        row = {
            "log_amt": log_amt, "amt": amt, "hour": hour, "dow": dow,
            "is_night": is_night, "age": age, "geo_km": geo_km,
            "log_city_pop": log_city_pop, "mins_since_last": mins_since_last,
            "tx_24h": tx_24h, "amt_24h": amt_24h, "tx_1h": tx_1h,
            "amt_over_p95": amt_over_p95, "amt_to_p95": amt_to_p95,
            "cat_fraud_rate": cat_fraud_rate,
            "category": category, "gender": gender, "state": state,
        }
        df = pd.DataFrame([row])
        for c in self.CAT_COLS:
            df[c] = df[c].astype("category")
        return df[self.FEATURES]

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.94:  return "CRITICAL"
        if score >= 0.75:  return "HIGH"
        if score >= 0.40:  return "MEDIUM"
        return "LOW"

    @staticmethod
    def _top_signals(tx: dict, score: float) -> list[str]:
        signals = []
        amt = float(tx.get("amt", tx.get("amount", 0)))
        hour = int(tx.get("hour", 12))
        tx_24h = int(tx.get("tx_24h", 0))
        amt_to_p95 = float(tx.get("amt_to_p95", 1.0))

        if amt_to_p95 > 2.0:
            signals.append(f"amount {amt_to_p95:.1f}× category 95th-percentile")
        if hour >= 22 or hour <= 4:
            signals.append("off-hours transaction")
        if tx_24h >= 5:
            signals.append(f"velocity spike: {tx_24h} txns in 24h")
        geo = float(tx.get("geo_km", 0))
        if geo > 500:
            signals.append(f"geo anomaly: {geo:.0f} km from home")
        mins = float(tx.get("mins_since_last", 99999))
        if mins < 5:
            signals.append(f"rapid repeat: {mins:.1f} min since last txn")
        cat = tx.get("category", "")
        if cat in ("misc_net", "shopping_net", "grocery_pos"):
            signals.append(f"high-risk merchant category: {cat}")
        if not signals:
            signals.append("moderate composite risk score")
        return signals[:4]

    def score(self, tx: dict[str, Any]) -> dict:
        df    = self._featurize(tx)
        prob  = float(self.model.predict(df)[0])
        level = self._risk_level(prob)
        route = prob >= self.threshold
        return {
            "risk_score":   round(prob, 4),
            "risk_level":   level,
            "route_to_llm": route,
            "top_signals":  self._top_signals(tx, prob),
            "threshold":    round(self.threshold, 4),
            "domain":       "card_fraud",
        }
