"""
Tier-1 AML Scorer
Wraps aml_lgbm_model.txt + aml_lgbm_preproc.joblib (tabular)
and aml_gnn.pt (graph GNN).
"""
import json, os, math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

_LGBM_DIR = Path(os.getenv("AML_LGBM_DIR",
    Path(__file__).parent.parent.parent / "models" / "aml"))
_GNN_DIR  = Path(os.getenv("AML_GNN_DIR",
    Path(__file__).parent.parent.parent / "models" / "aml"))

CAT_COLS = ["Receiving Currency", "Payment Currency", "Payment Format"]
FEATURES = [
    "hour", "dow", "log_paid", "log_recv", "amt_diff", "ccy_mismatch",
    "self_loop", "is_round", "same_bank",
    "snd_out_deg", "snd_in_deg", "snd_out_cnt", "snd_in_cnt", "snd_out_amt_mean",
    "rcv_out_deg", "rcv_in_deg", "rcv_in_cnt",
    "gather_scatter", "amt_to_snd_mean",
] + CAT_COLS


class AMLScorer:
    """
    score(tx_dict) → dict with risk_score, risk_level, route_to_llm, top_signals
    """
    DEFAULT_THRESHOLD = 1.768963856800349e-116  # recall-0.80 from metrics

    def __init__(self, lgbm_dir: Path = _LGBM_DIR):
        model_path   = lgbm_dir / "aml_lgbm_model.txt"
        preproc_path = lgbm_dir / "aml_lgbm_preproc.joblib"
        metrics_path = lgbm_dir / "aml_lgbm_metrics.json"

        self.model = lgb.Booster(model_file=str(model_path))
        preproc    = joblib.load(preproc_path)
        self.graph_feats = preproc.get("graph", {})

        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            self.threshold = m.get("routing_threshold", self.DEFAULT_THRESHOLD)
        else:
            self.threshold = self.DEFAULT_THRESHOLD

    def _get_graph_feat(self, key: str, account: str, default: float = 0.0) -> float:
        d = self.graph_feats.get(key, {})
        return float(d.get(account, default))

    def _featurize(self, tx: dict[str, Any]) -> pd.DataFrame:
        account  = str(tx.get("account",  tx.get("Account",  "UNK")))
        account1 = str(tx.get("account_1", tx.get("Account.1", "UNK")))
        paid     = float(tx.get("amount_paid",      tx.get("Amount Paid",      0)))
        recv     = float(tx.get("amount_received",  tx.get("Amount Received",  paid)))
        pay_curr = str(tx.get("payment_currency",   tx.get("Payment Currency", "USD")))
        rec_curr = str(tx.get("receiving_currency", tx.get("Receiving Currency","USD")))
        pay_fmt  = str(tx.get("payment_format",     tx.get("Payment Format",   "ACH")))
        from_bank = str(tx.get("from_bank", tx.get("From Bank", "BNKA")))
        to_bank   = str(tx.get("to_bank",   tx.get("To Bank",   "BNKB")))
        hour     = int(tx.get("hour", 12))
        dow      = int(tx.get("dow",  0))

        log_paid = math.log1p(paid)
        log_recv = math.log1p(recv)
        amt_diff = abs(paid - recv)
        ccy_mm   = 1 if pay_curr != rec_curr else 0
        self_loop = 1 if account == account1 else 0
        is_round  = 1 if (paid > 0 and paid % 100 == 0) else 0
        same_bank = 1 if from_bank == to_bank else 0

        snd_out_deg      = self._get_graph_feat("out_deg", account)
        snd_in_deg       = self._get_graph_feat("in_deg",  account)
        snd_out_cnt      = self._get_graph_feat("out_cnt", account)
        snd_in_cnt       = self._get_graph_feat("in_cnt",  account)
        snd_out_amt_mean = self._get_graph_feat("out_amt_mean", account)
        rcv_out_deg      = self._get_graph_feat("out_deg", account1)
        rcv_in_deg       = self._get_graph_feat("in_deg",  account1)
        rcv_in_cnt       = self._get_graph_feat("in_cnt",  account1)

        gather_scatter   = 1 if (snd_in_deg >= 5 and snd_out_deg >= 5) else 0
        amt_to_snd_mean  = paid / (snd_out_amt_mean + 1e-6)

        row = {
            "hour": hour, "dow": dow, "log_paid": log_paid, "log_recv": log_recv,
            "amt_diff": amt_diff, "ccy_mismatch": ccy_mm, "self_loop": self_loop,
            "is_round": is_round, "same_bank": same_bank,
            "snd_out_deg": snd_out_deg, "snd_in_deg": snd_in_deg,
            "snd_out_cnt": snd_out_cnt, "snd_in_cnt": snd_in_cnt,
            "snd_out_amt_mean": snd_out_amt_mean,
            "rcv_out_deg": rcv_out_deg, "rcv_in_deg": rcv_in_deg, "rcv_in_cnt": rcv_in_cnt,
            "gather_scatter": gather_scatter, "amt_to_snd_mean": amt_to_snd_mean,
            "Receiving Currency": rec_curr,
            "Payment Currency":   pay_curr,
            "Payment Format":     pay_fmt,
        }
        df = pd.DataFrame([row])
        for c in CAT_COLS:
            df[c] = df[c].astype("category")
        return df[FEATURES]

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.5:   return "CRITICAL"
        if score >= 0.1:   return "HIGH"
        if score >= 0.01:  return "MEDIUM"
        return "LOW"

    @staticmethod
    def _top_signals(tx: dict, score: float) -> list[str]:
        signals = []
        paid   = float(tx.get("amount_paid", tx.get("Amount Paid", 0)))
        cc_mm  = tx.get("payment_currency","USD") != tx.get("receiving_currency","USD")
        sl     = tx.get("account","") == tx.get("account_1","X")
        rnd    = paid > 0 and paid % 100 == 0

        snd_out = float(tx.get("snd_out_deg", 0))
        rcv_in  = float(tx.get("rcv_in_deg",  0))

        if cc_mm:
            signals.append(f"cross-currency: {tx.get('payment_currency','?')} → {tx.get('receiving_currency','?')}")
        if sl:
            signals.append("self-loop transfer (account round-trip)")
        if rnd:
            signals.append(f"round-number amount: {paid:,.0f}")
        if snd_out >= 20:
            signals.append(f"fan-out: sender has {int(snd_out)} unique counterparties")
        if rcv_in >= 20:
            signals.append(f"fan-in: receiver aggregates from {int(rcv_in)} sources")
        fmt = tx.get("payment_format", tx.get("Payment Format", ""))
        if fmt == "ACH":
            signals.append("ACH channel — common in structuring patterns")
        if not signals:
            signals.append("moderate composite AML risk score")
        return signals[:4]

    def score(self, tx: dict[str, Any]) -> dict:
        df   = self._featurize(tx)
        prob = float(self.model.predict(df)[0])
        level = self._risk_level(prob)
        route = bool(prob >= self.threshold)
        return {
            "risk_score":   round(prob, 6),
            "risk_level":   level,
            "route_to_llm": route,
            "top_signals":  self._top_signals(tx, prob),
            "threshold":    self.threshold,
            "domain":       "aml",
        }
