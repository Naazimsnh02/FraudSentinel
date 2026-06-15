"""Tier-1 real-time card-fraud scorer (LightGBM) on Sparkov.
Engineers behavioral/contextual features (velocity, geo, time, age) and trains
a gradient-boosted scorer. Evaluated at NATURAL imbalance with PR-AUC / ROC-AUC.
"""
import math, json, time
import numpy as np, pandas as pd
import lightgbm as lgb
from datasets import load_dataset
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, confusion_matrix)
import joblib

R = 6371.0
def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

CAT_COLS = ["category", "gender", "state"]

def featurize(df, fit_maps=None, cat_p95=None, cat_rate=None):
    df = df.sort_values(["cc_num", "unix_time"]).copy()
    # time features
    dt = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)
    # age
    df["age"] = (dt - pd.to_datetime(df["dob"])).dt.days / 365.25
    # geo distance home->merchant
    df["geo_km"] = haversine(df["lat"].values, df["long"].values,
                             df["merch_lat"].values, df["merch_long"].values)
    df["log_amt"] = np.log1p(df["amt"])
    df["log_city_pop"] = np.log1p(df["city_pop"])
    # velocity per card (vectorized rolling within 24h / 1h)
    df["prev_unix"] = df.groupby("cc_num")["unix_time"].shift(1)
    df["mins_since_last"] = (df["unix_time"] - df["prev_unix"]) / 60.0
    df["mins_since_last"] = df["mins_since_last"].fillna(99999).clip(upper=99999)
    # rolling counts using numpy two-pointer per card
    tx24 = np.zeros(len(df), int); amt24 = np.zeros(len(df)); tx1 = np.zeros(len(df), int)
    idx = 0
    for _, g in df.groupby("cc_num", sort=False):
        ut = g["unix_time"].values; am = g["amt"].values
        j24 = j1 = 0
        for i in range(len(g)):
            while ut[i] - ut[j24] > 86400: j24 += 1
            while ut[i] - ut[j1] > 3600: j1 += 1
            tx24[idx] = i - j24
            amt24[idx] = am[j24:i].sum()
            tx1[idx] = i - j1
            idx += 1
    df["tx_24h"] = tx24; df["amt_24h"] = amt24; df["tx_1h"] = tx1
    # category amount anomaly vs train norms
    if cat_p95 is None:
        cat_p95 = df.groupby("category")["amt"].quantile(0.95).to_dict()
        cat_rate = df.groupby("category")["is_fraud"].mean().to_dict()
    df["cat_p95"] = df["category"].map(cat_p95).fillna(np.median(list(cat_p95.values())))
    df["amt_over_p95"] = (df["amt"] > df["cat_p95"]).astype(int)
    df["amt_to_p95"] = df["amt"] / (df["cat_p95"] + 1e-6)
    df["cat_fraud_rate"] = df["category"].map(cat_rate).fillna(0.0)
    # categorical encodings
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df, cat_p95, cat_rate

FEATURES = ["log_amt", "amt", "hour", "dow", "is_night", "age", "geo_km",
            "log_city_pop", "mins_since_last", "tx_24h", "amt_24h", "tx_1h",
            "amt_over_p95", "amt_to_p95", "cat_fraud_rate"] + CAT_COLS

def main():
    t0 = time.time()
    print("Loading Sparkov splits...")
    tr = load_dataset("pointe77/credit-card-transaction", split="train").to_pandas()
    te = load_dataset("pointe77/credit-card-transaction", split="test").to_pandas()
    print(f"train {len(tr):,} (fraud {tr.is_fraud.mean():.4%}) | test {len(te):,} (fraud {te.is_fraud.mean():.4%})")

    tr, cat_p95, cat_rate = featurize(tr)
    te, _, _ = featurize(te, cat_p95=cat_p95, cat_rate=cat_rate)

    Xtr, ytr = tr[FEATURES], tr["is_fraud"].values
    Xte, yte = te[FEATURES], te["is_fraud"].values
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    print(f"scale_pos_weight = {spw:.1f}")

    # hold out a validation slice from train for early stopping
    n = len(Xtr); cut = int(n*0.9)
    Xt, yt = Xtr.iloc[:cut], ytr[:cut]
    Xv, yv = Xtr.iloc[cut:], ytr[cut:]

    params = dict(objective="binary", metric="average_precision",
                  learning_rate=0.05, num_leaves=64, max_depth=-1,
                  min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=5.0, scale_pos_weight=spw,
                  n_jobs=-1, verbose=-1)
    dtr = lgb.Dataset(Xt, yt, categorical_feature=CAT_COLS)
    dvl = lgb.Dataset(Xv, yv, categorical_feature=CAT_COLS, reference=dtr)
    print("Training card-fraud LightGBM...")
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dvl],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)])

    # evaluate at natural imbalance on official test split
    p = model.predict(Xte, num_iteration=model.best_iteration)
    prauc = average_precision_score(yte, p)
    rocauc = roc_auc_score(yte, p)
    print(f"\n=== CARD TEST (natural imbalance {yte.mean():.4%}) ===")
    print(f"PR-AUC={prauc:.4f}  ROC-AUC={rocauc:.4f}")

    # choose routing threshold: high-recall to NOT miss fraud (Tier-1 must catch, LLM filters FPs)
    prec, rec, thr = precision_recall_curve(yte, p)
    results = {}
    for target_recall in [0.80, 0.85, 0.90, 0.95]:
        # smallest threshold achieving >= target recall
        ok = np.where(rec[:-1] >= target_recall)[0]
        if len(ok):
            ti = ok[-1]
            t = float(thr[ti])
            pr_ = precision_score(yte, p >= t); re_ = recall_score(yte, p >= t)
            flagged = float((p >= t).mean())
            results[f"recall_{target_recall}"] = dict(threshold=t, precision=pr_, recall=re_, flagged_frac=flagged)
            print(f"recall>={target_recall}: thr={t:.4f} P={pr_:.3f} R={re_:.3f} flagged={flagged:.2%}")

    # default routing threshold = target recall 0.90
    route_t = results["recall_0.9"]["threshold"]
    yhat = (p >= route_t).astype(int)
    print("\nConfusion @ routing threshold (recall~0.90):")
    print(confusion_matrix(yte, yhat))
    print("F1:", f1_score(yte, yhat))

    # feature importance
    imp = dict(sorted(zip(FEATURES, model.feature_importance(importance_type="gain").tolist()),
                      key=lambda x: -x[1]))
    print("\nTop features:", list(imp.items())[:8])

    model.save_model("cc_lgbm_model.txt")
    joblib.dump({"cat_p95": cat_p95, "cat_rate": cat_rate, "features": FEATURES,
                 "cat_cols": CAT_COLS}, "cc_lgbm_preproc.joblib")
    meta = dict(domain="card_fraud", source="pointe77/credit-card-transaction",
                train_rows=int(len(tr)), test_rows=int(len(te)),
                test_fraud_rate=float(yte.mean()), pr_auc=float(prauc), roc_auc=float(rocauc),
                routing_threshold=float(route_t), thresholds=results,
                top_features=list(imp.items())[:12], n_features=len(FEATURES),
                best_iteration=int(model.best_iteration), scale_pos_weight=float(spw))
    json.dump(meta, open("cc_lgbm_metrics.json", "w"), indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s. Saved model + preproc + metrics.")

if __name__ == "__main__":
    main()
