"""Tier-1 AML scorer (LightGBM) on IBM AML HI-Small.
Engineers per-transaction + account-graph features (out/in degree, counts,
amount stats for sender & receiver). Chronological split to avoid leakage.
Evaluated at NATURAL imbalance with PR-AUC / ROC-AUC.
"""
import json, time
import numpy as np, pandas as pd
import lightgbm as lgb
from datasets import load_dataset
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, confusion_matrix)
import joblib

def build_graph_feats(df):
    """Account-level aggregates computed on TRAIN ONLY to avoid leakage."""
    out_deg = df.groupby("Account")["Account.1"].nunique()
    in_deg = df.groupby("Account.1")["Account"].nunique()
    out_cnt = df.groupby("Account").size()
    in_cnt = df.groupby("Account.1").size()
    out_amt_mean = df.groupby("Account")["Amount Paid"].mean()
    out_amt_sum = df.groupby("Account")["Amount Paid"].sum()
    # Leakage-safe suspicious-history priors, learned on TRAIN labels only:
    # historical laundering rate per sender and per receiver account.
    return dict(out_deg=out_deg, in_deg=in_deg, out_cnt=out_cnt, in_cnt=in_cnt,
                out_amt_mean=out_amt_mean, out_amt_sum=out_amt_sum)

def featurize(df, g):
    df = df.copy()
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    df["hour"] = ts.dt.hour.fillna(0).astype(int)
    df["dow"] = ts.dt.dayofweek.fillna(0).astype(int)
    df["log_paid"] = np.log1p(df["Amount Paid"])
    df["log_recv"] = np.log1p(df["Amount Received"])
    df["amt_diff"] = (df["Amount Paid"] - df["Amount Received"]).abs()
    df["ccy_mismatch"] = (df["Receiving Currency"] != df["Payment Currency"]).astype(int)
    df["self_loop"] = (df["Account"] == df["Account.1"]).astype(int)
    df["is_round"] = (df["Amount Paid"] % 100 == 0).astype(int)
    df["same_bank"] = (df["From Bank"] == df["To Bank"]).astype(int)
    # sender graph feats
    df["snd_out_deg"] = df["Account"].map(g["out_deg"]).fillna(0)
    df["snd_in_deg"] = df["Account"].map(g["in_deg"]).fillna(0)
    df["snd_out_cnt"] = df["Account"].map(g["out_cnt"]).fillna(0)
    df["snd_in_cnt"] = df["Account"].map(g["in_cnt"]).fillna(0)
    df["snd_out_amt_mean"] = df["Account"].map(g["out_amt_mean"]).fillna(0)
    # receiver graph feats
    df["rcv_out_deg"] = df["Account.1"].map(g["out_deg"]).fillna(0)
    df["rcv_in_deg"] = df["Account.1"].map(g["in_deg"]).fillna(0)
    df["rcv_in_cnt"] = df["Account.1"].map(g["in_cnt"]).fillna(0)
    # gather-scatter indicator
    df["gather_scatter"] = ((df["snd_in_deg"] >= 5) & (df["snd_out_deg"] >= 5)).astype(int)
    df["amt_to_snd_mean"] = df["Amount Paid"] / (df["snd_out_amt_mean"] + 1e-6)
    for c in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        df[c] = df[c].astype("category")
    return df

CAT_COLS = ["Receiving Currency", "Payment Currency", "Payment Format"]
FEATURES = ["hour", "dow", "log_paid", "log_recv", "amt_diff", "ccy_mismatch",
            "self_loop", "is_round", "same_bank",
            "snd_out_deg", "snd_in_deg", "snd_out_cnt", "snd_in_cnt", "snd_out_amt_mean",
            "rcv_out_deg", "rcv_in_deg", "rcv_in_cnt",
            "gather_scatter", "amt_to_snd_mean"] + CAT_COLS

def main():
    t0 = time.time()
    print("Loading IBM AML HI-Small...")
    df = load_dataset("eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans", split="train").to_pandas()
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    df = df.sort_values("_ts").reset_index(drop=True)
    print(f"rows {len(df):,}  laundering {df['Is Laundering'].mean():.4%}")

    # chronological split 80/20 (avoids temporal leakage)
    cut = int(len(df)*0.8)
    tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    # graph features fit on TRAIN ONLY
    g = build_graph_feats(tr)
    tr = featurize(tr, g); te = featurize(te, g)

    Xtr, ytr = tr[FEATURES], tr["Is Laundering"].values
    Xte, yte = te[FEATURES], te["Is Laundering"].values
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    print(f"train {len(tr):,} (laund {ytr.mean():.4%}) | test {len(te):,} (laund {yte.mean():.4%}) | spw={spw:.0f}")

    # val slice for early stopping (chronological tail of train)
    c2 = int(len(Xtr)*0.9)
    dtr = lgb.Dataset(Xtr.iloc[:c2], ytr[:c2], categorical_feature=CAT_COLS)
    dvl = lgb.Dataset(Xtr.iloc[c2:], ytr[c2:], categorical_feature=CAT_COLS, reference=dtr)
    params = dict(objective="binary", metric="average_precision",
                  learning_rate=0.05, num_leaves=128, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=5.0, scale_pos_weight=spw, n_jobs=-1, verbose=-1)
    print("Training AML LightGBM...")
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dvl],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)])

    p = model.predict(Xte, num_iteration=model.best_iteration)
    prauc = average_precision_score(yte, p); rocauc = roc_auc_score(yte, p)
    print(f"\n=== AML TEST (natural imbalance {yte.mean():.4%}) ===")
    print(f"PR-AUC={prauc:.4f}  ROC-AUC={rocauc:.4f}")

    prec, rec, thr = precision_recall_curve(yte, p)
    results = {}
    for target_recall in [0.50, 0.60, 0.70, 0.80]:
        ok = np.where(rec[:-1] >= target_recall)[0]
        if len(ok):
            ti = ok[-1]; t = float(thr[ti])
            pr_ = precision_score(yte, p >= t, zero_division=0); re_ = recall_score(yte, p >= t)
            flagged = float((p >= t).mean())
            results[f"recall_{target_recall}"] = dict(threshold=t, precision=pr_, recall=re_, flagged_frac=flagged)
            print(f"recall>={target_recall}: thr={t:.4f} P={pr_:.3f} R={re_:.3f} flagged={flagged:.2%}")

    route_key = "recall_0.8" if "recall_0.8" in results else list(results)[-1]
    route_t = results[route_key]["threshold"]
    yhat = (p >= route_t).astype(int)
    print(f"\nConfusion @ routing threshold ({route_key}):"); print(confusion_matrix(yte, yhat))
    print("F1:", f1_score(yte, yhat, zero_division=0))

    imp = dict(sorted(zip(FEATURES, model.feature_importance(importance_type="gain").tolist()),
                      key=lambda x: -x[1]))
    print("\nTop features:", list(imp.items())[:8])

    model.save_model("aml_lgbm_model.txt")
    joblib.dump({"graph": {k: v.to_dict() for k, v in g.items()},
                 "features": FEATURES, "cat_cols": CAT_COLS},
                "aml_lgbm_preproc.joblib")
    meta = dict(domain="aml", source="eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans",
                train_rows=int(len(tr)), test_rows=int(len(te)),
                test_laundering_rate=float(yte.mean()), pr_auc=float(prauc), roc_auc=float(rocauc),
                routing_threshold=float(route_t), routing_key=route_key, thresholds=results,
                top_features=list(imp.items())[:12], n_features=len(FEATURES),
                best_iteration=int(model.best_iteration), scale_pos_weight=float(spw))
    json.dump(meta, open("aml_lgbm_metrics.json", "w"), indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s.")

if __name__ == "__main__":
    main()
