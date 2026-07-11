import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
from scipy.optimize import linprog
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, make_scorer
import warnings
warnings.filterwarnings('ignore')

import pypsa

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BUS      = "DE0 1"
CARRIERS = ["solar", "onwind"]

NETWORK_FILES = {
    2023: "base_s_50_elec_2023.nc",
    2024: "base_s_50_elec_2024.nc",
    2025: "base_s_50_elec_2025.nc",
}

HORIZON    = 24
STEP       = 24
N_WINDOWS  = 365
META_SPLIT = 0

CAPACITY_MWH = 1.5
POWER_MW     = 0.25
ETA          = 0.90

USE_LEARNED_THRESHOLDS = True

TSCV = TimeSeriesSplit(n_splits=5)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_network_data(network_files, bus, carriers):
    dfs = {}
    for year, path in network_files.items():
        n  = pypsa.Network(path)
        mp = n.buses_t.marginal_price[bus]
        nl = n.loads_t.p[bus]

        gen_series = {}
        for carrier in carriers:
            gens  = n.generators[(n.generators.carrier == carrier) & (n.generators.bus == bus)].index
            p_nom = n.generators.loc[gens, "p_nom"]
            gen_series[f"{carrier}_gen_mw"] = (n.generators_t.p_max_pu[gens] * p_nom).sum(axis=1)

        dfs[year] = pd.DataFrame({
            "load":          nl,
            **gen_series,
            "month":         mp.index.month,
            "hour":          mp.index.hour,
            "day_of_week":   mp.index.dayofweek,
            "marginal_price": mp,
            "year":          year,
        })
        print(f"Loaded {year}: {len(dfs[year])} rows")

    df = pd.concat(dfs.values()).sort_index()
    print(df.shape)
    print(df["year"].value_counts().sort_index())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    for col in ["hour", "day_of_week", "month"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["price_lag1"]   = df["marginal_price"].shift(1)
    df["price_lag24"]  = df["marginal_price"].shift(24)
    df["price_lag168"] = df["marginal_price"].shift(168)
    df["price_roll_mean_24"]    = df["marginal_price"].shift(1).rolling(24).mean()
    df["price_roll_std_24"]     = df["marginal_price"].shift(1).rolling(24).std()
    df["price_roll_mean_168"]   = df["marginal_price"].shift(1).rolling(168).mean()
    df["price_smooth_trend_12"] = df["marginal_price"].shift(1).rolling(12).mean()

    df["load_lag1"]   = df["load"].shift(1)
    df["load_lag2"]   = df["load"].shift(2)
    df["load_lag3"]   = df["load"].shift(3)
    df["load_lag6"]   = df["load"].shift(6)
    df["load_lag12"]  = df["load"].shift(12)
    df["load_lag24"]  = df["load"].shift(24)
    df["load_lag48"]  = df["load"].shift(48)
    df["load_lag168"] = df["load"].shift(168)
    df["load_roll_mean_24"]  = df["load"].shift(1).rolling(24).mean()
    df["load_roll_std_24"]   = df["load"].shift(1).rolling(24).std()
    df["load_roll_mean_168"] = df["load"].shift(1).rolling(168).mean()

    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

reg_features = [
    "hour", "day_of_week", "month", "load", "solar_gen_mw", "onwind_gen_mw",
    "price_lag1", "price_lag24", "price_lag168",
    "price_roll_mean_24", "price_roll_std_24", "load_lag1",
    "price_smooth_trend_12",
]
load_features = [
    "hour", "day_of_week", "month",
    "solar_gen_mw", "onwind_gen_mw",
    "load_lag1", "load_lag2", "load_lag3", "load_lag6",
    "load_lag12", "load_lag24", "load_lag48", "load_lag168",
    "load_roll_mean_24", "load_roll_std_24", "load_roll_mean_168",
]
threshold_features = [
    "price_roll_std_24", "price_roll_mean_24", "price_roll_mean_168",
    "solar_gen_mw", "onwind_gen_mw", "load_lag24",
    "hour", "day_of_week", "month",
]

reg_idx  = {f: i for i, f in enumerate(reg_features)}
load_idx = {f: i for i, f in enumerate(load_features)}
thr_idx  = {f: i for i, f in enumerate(threshold_features)}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_load_model(X, y):
    param_grid = {
        "n_estimators":     [100, 200, 300, 500, 800],
        "max_depth":        [3, 4, 5, 6, 8, 10],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, tree_method="hist"),
        param_distributions=param_grid,
        n_iter=20, scoring="neg_mean_absolute_error",
        cv=TSCV, n_jobs=17, random_state=42, verbose=1,
    )
    search.fit(X.to_numpy(), y.to_numpy())
    print(f"Best Load Params: {search.best_params_}")
    return search.best_estimator_


def train_quantile_price_model(X, y, quantile_alpha, label=""):
    pinball_scorer = make_scorer(mean_pinball_loss, alpha=quantile_alpha, greater_is_better=False)
    param_grid = {
        "n_estimators":     [50, 100, 200, 400],
        "max_depth":        [3, 4, 5, 6],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [5, 10, 20],
        "reg_alpha":        [0.0, 0.1, 1.0, 5.0],
        "reg_lambda":       [1.0, 5.0, 10.0, 20.0],
    }
    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:quantileerror", quantile_alpha=quantile_alpha,
                     random_state=42, tree_method="hist"),
        param_distributions=param_grid,
        n_iter=20, scoring=pinball_scorer,
        cv=TSCV, n_jobs=17, random_state=42, verbose=1,
    )
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    search.fit(X_arr, y_arr)
    print(f"Best {label} Params: {search.best_params_}")
    return search.best_estimator_


def train_price_models(X_train, y_train):
    print("=== Training Quantile Price Models (q10 / q50 / q90) ===")
    q50 = train_quantile_price_model(X_train, y_train, 0.50, "q50 Price (median)")

    y_pred_q50     = q50.predict(X_train.to_numpy())
    train_residuals = y_train.to_numpy() - y_pred_q50

    print("--- Training q10 model on negative residual offsets ---")
    q10 = train_quantile_price_model(X_train, train_residuals, 0.05, "q10 Price Offset")

    print("--- Training q90 model on positive residual offsets ---")
    q90 = train_quantile_price_model(X_train, train_residuals, 0.95, "q90 Price Offset")

    return q50, q10, q90


def train_threshold_models(df, test_start_idx):
    def _label_one_window(start):
        actual_prices = df["marginal_price"].iloc[start:start + HORIZON].values
        best_rev, best_cp, best_dp = -np.inf, 25, 75
        for cp in range(5, 50, 5):
            for dp in range(55, 95, 5):
                rev = lmp_arbitrage_dispatch(actual_prices, charge_pct=cp, discharge_pct=dp)["revenue"].sum()
                if rev > best_rev:
                    best_rev, best_cp, best_dp = rev, cp, dp
        row = make_threshold_row(df, start)
        return row, best_cp, best_dp

    print("=== Building threshold training data ===")
    thr_starts  = list(range(168, test_start_idx - HORIZON, STEP))
    results_thr = Parallel(n_jobs=-1)(delayed(_label_one_window)(s) for s in thr_starts)

    thr_X_rows, thr_y_cp, thr_y_dp = zip(*results_thr)
    thr_X    = np.array(thr_X_rows)
    thr_y_cp = np.array(thr_y_cp, dtype=float)
    thr_y_dp = np.array(thr_y_dp, dtype=float)
    print(f"Threshold training samples: {len(thr_X)}")

    thr_param_grid = {
        "n_estimators":     [50, 100, 200, 300, 500],
        "max_depth":        [3, 4, 5, 6, 8],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    print("=== Training XGBoost threshold predictors ===")
    TSCV_thr = TimeSeriesSplit(n_splits=5)

    charge_model = RandomizedSearchCV(
        XGBRegressor(random_state=42, tree_method="hist"),
        param_distributions=thr_param_grid,
        n_iter=20, scoring="neg_mean_absolute_error",
        cv=TSCV_thr, n_jobs=17, random_state=42, verbose=1,
    )
    charge_model.fit(thr_X, thr_y_cp)
    print(f"Best charge-threshold params: {charge_model.best_params_}")

    discharge_model = RandomizedSearchCV(
        XGBRegressor(random_state=42, tree_method="hist"),
        param_distributions=thr_param_grid,
        n_iter=20, scoring="neg_mean_absolute_error",
        cv=TSCV_thr, n_jobs=17, random_state=42, verbose=1,
    )
    discharge_model.fit(thr_X, thr_y_dp)
    print(f"Best discharge-threshold params: {discharge_model.best_params_}")

    return charge_model, discharge_model

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ROW BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def make_load_row(df, pred_idx, pred_time, load_history):
    row = np.zeros(len(load_features))
    row[load_idx["hour"]]          = pred_time.hour
    row[load_idx["day_of_week"]]   = pred_time.dayofweek
    row[load_idx["month"]]         = pred_time.month
    row[load_idx["solar_gen_mw"]]  = df["solar_gen_mw"].iloc[pred_idx]
    row[load_idx["onwind_gen_mw"]] = df["onwind_gen_mw"].iloc[pred_idx]
    n = len(load_history)
    for lag, name in [(1,"load_lag1"),(2,"load_lag2"),(3,"load_lag3"),
                      (6,"load_lag6"),(12,"load_lag12"),(24,"load_lag24"),
                      (48,"load_lag48"),(168,"load_lag168")]:
        row[load_idx[name]] = load_history[n - lag] if n >= lag else load_history[0]
    row[load_idx["load_roll_mean_24"]]  = np.mean(load_history[-24:])
    row[load_idx["load_roll_std_24"]]   = np.std(load_history[-24:]) if len(load_history[-24:]) > 1 else 0
    row[load_idx["load_roll_mean_168"]] = np.mean(load_history[-168:])
    return row


def make_reg_row(df, pred_idx, pred_time, price_history, pred_load, prev_pred_load):
    row = np.zeros(len(reg_features))
    row[reg_idx["hour"]]          = pred_time.hour
    row[reg_idx["day_of_week"]]   = pred_time.dayofweek
    row[reg_idx["month"]]         = pred_time.month
    row[reg_idx["load"]]          = pred_load
    row[reg_idx["solar_gen_mw"]]  = df["solar_gen_mw"].iloc[pred_idx]
    row[reg_idx["onwind_gen_mw"]] = df["onwind_gen_mw"].iloc[pred_idx]
    row[reg_idx["load_lag1"]]     = prev_pred_load
    n = len(price_history)
    row[reg_idx["price_lag1"]]   = price_history[n-1]   if n >= 1   else price_history[0]
    row[reg_idx["price_lag24"]]  = price_history[n-24]  if n >= 24  else price_history[0]
    row[reg_idx["price_lag168"]] = price_history[n-168] if n >= 168 else price_history[0]
    win24 = price_history[-24:]
    win12 = price_history[-12:]
    row[reg_idx["price_roll_mean_24"]]    = np.mean(win24)
    row[reg_idx["price_roll_std_24"]]     = np.std(win24) if len(win24) > 1 else 0
    row[reg_idx["price_smooth_trend_12"]] = np.mean(win12)
    return row


def make_threshold_row(df, start_idx):
    row = np.zeros(len(threshold_features))
    ref = df.iloc[start_idx]
    for feat in threshold_features:
        if feat in df.columns:
            row[thr_idx[feat]] = ref[feat]
    return row


def predict_thresholds(df, start_idx, charge_model, discharge_model):
    row = make_threshold_row(df, start_idx).reshape(1, -1)
    cp  = float(charge_model.best_estimator_.predict(row)[0])
    dp  = float(discharge_model.best_estimator_.predict(row)[0])
    cp  = np.clip(cp, 5, 45)
    dp  = np.clip(dp, 55, 95)
    if dp <= cp + 5:
        dp = cp + 10
    return cp, dp

# ══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def lmp_arbitrage_dispatch(prices, capacity_mwh=CAPACITY_MWH, power_mw=POWER_MW, eta=ETA,
                           charge_pct=25, discharge_pct=75,
                           charge_signal=None, discharge_signal=None):
    prices = np.asarray(prices, dtype=float)
    cs = np.asarray(charge_signal,    dtype=float) if charge_signal    is not None else prices
    ds = np.asarray(discharge_signal, dtype=float) if discharge_signal is not None else prices

    p_low  = np.percentile(cs, charge_pct)
    p_high = np.percentile(ds, discharge_pct)
    sqrt_eta = np.sqrt(eta)
    soc, rows = 0.5 * capacity_mwh, []

    for i, price in enumerate(prices):
        action, revenue = "idle", 0.0
        if cs[i] <= p_low and soc < capacity_mwh:
            action   = "charge"
            energy   = min(power_mw, (capacity_mwh - soc) / sqrt_eta)
            soc     += energy * sqrt_eta
            revenue  = -price * energy
        elif ds[i] >= p_high and soc > 0:
            action   = "discharge"
            energy   = min(power_mw, soc * sqrt_eta)
            soc     -= energy / sqrt_eta
            revenue  = price * energy
        rows.append({"action": action, "revenue": revenue, "soc": soc})

    return pd.DataFrame(rows)




def lp_oracle(prices, capacity_mwh=CAPACITY_MWH, power_mw=POWER_MW, eta=ETA):
    T      = len(prices)
    prices = np.asarray(prices, dtype=float)
    soc_0  = 0.5 * capacity_mwh

    c      = np.concatenate([prices, -prices])
    bounds = [(0.0, power_mw)] * (2 * T)

    A_ub, b_ub = [], []
    for t in range(T):
        row = np.zeros(2 * T)
        row[:t+1]    =  np.sqrt(eta)
        row[T:T+t+1] = -1.0 / np.sqrt(eta)
        A_ub.append(row.copy()); b_ub.append(capacity_mwh - soc_0)
        A_ub.append(-row);       b_ub.append(soc_0)

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if res.success:
        return max(np.dot(res.x[T:], prices) - np.dot(res.x[:T], prices), 0.0)
    return 0.0

# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

def ranking_accuracy(pred_prices, actual_prices, k=6):
    pred_cheap   = set(np.argsort(pred_prices)[:k])
    actual_cheap = set(np.argsort(actual_prices)[:k])
    pred_exp     = set(np.argsort(pred_prices)[-k:])
    actual_exp   = set(np.argsort(actual_prices)[-k:])
    return (len(pred_cheap & actual_cheap) / k + len(pred_exp & actual_exp) / k) / 2


def forecast_window(df, start_idx, load_model, q50_model, q10_model, q90_model):
    price_history = list(df["marginal_price"].iloc[:start_idx])
    load_history  = list(df["load"].iloc[:start_idx])
    rows = []

    for h in range(HORIZON):
        pred_idx  = start_idx + h
        pred_time = df.index[pred_idx]

        load_row  = make_load_row(df, pred_idx, pred_time, load_history)
        pred_load = float(load_model.predict(load_row.reshape(1, -1))[0])
        prev_load = load_history[-1]

        reg_row    = make_reg_row(df, pred_idx, pred_time, price_history, pred_load, prev_load).reshape(1, -1)
        pred_q50   = float(q50_model.predict(reg_row)[0])
        pred_q10   = pred_q50 + min(0.0, float(q10_model.predict(reg_row)[0]))
        pred_q90   = pred_q50 + max(0.0, float(q90_model.predict(reg_row)[0]))

        rows.append({
            "timestamp":  pred_time,
            "pred_price": pred_q50,
            "pred_q10":   pred_q10,
            "pred_q90":   pred_q90,
            "actual":     df["marginal_price"].iloc[pred_idx],
            "pred_load":  pred_load,
        })
        price_history.append(pred_q50)
        load_history.append(pred_load)

    return pd.DataFrame(rows).set_index("timestamp")


def generate_price_windows(df, test_start_idx, load_model, q50_model, q10_model, q90_model):
    print("=== Generating quantile price forecasts (all windows) ===")
    price_windows = []
    for i in range(N_WINDOWS):
        start_idx = test_start_idx + i * STEP
        if start_idx + HORIZON > len(df):
            break
        w = forecast_window(df, start_idx, load_model, q50_model, q10_model, q90_model)
        price_windows.append({
            "start_time": w.index[0],
            "window_df":  w,
            "mae":        mean_absolute_error(w["actual"], w["pred_price"]),
        })
    return price_windows

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════


def replay_at_actual_prices(w, strategies):
    rev      = {k: 0.0              for k in strategies}
    soc      = {k: 0.5 * CAPACITY_MWH for k in strategies}
    sqrt_eta = np.sqrt(ETA)

    for h in range(len(w)):
        p_actual = w["actual"].values[h]
        for k, disp in strategies.items():
            act = disp.iloc[h]["action"]
            if act == "charge" and soc[k] < CAPACITY_MWH:
                energy   = min(POWER_MW, (CAPACITY_MWH - soc[k]) / sqrt_eta)
                soc[k]  += energy * sqrt_eta
                rev[k]  -= p_actual * energy
            elif act == "discharge" and soc[k] > 0:
                energy   = min(POWER_MW, soc[k] * sqrt_eta)
                soc[k]  -= energy / sqrt_eta
                rev[k]  += p_actual * energy
    return rev


def run_backtest(df, test_price_windows, thr_charge_model, thr_discharge_model, test_start_idx):
    print("=== Running integrated quantile price → LMP arbitrage dispatch ===")
    combined_results = []

    for pw in test_price_windows:
        w         = pw["window_df"]
        start_idx = df.index.get_loc(w.index[0])
        charge_pct, discharge_pct = predict_thresholds(df, start_idx, thr_charge_model, thr_discharge_model)

        pred_dispatch = lmp_arbitrage_dispatch(
            w["pred_price"].values, charge_pct=charge_pct, discharge_pct=discharge_pct,
            charge_signal=w["pred_q10"].values, discharge_signal=w["pred_q90"].values,
        )
        ml_fixed_dispatch = lmp_arbitrage_dispatch(
            w["pred_price"].values, charge_pct=25, discharge_pct=75,
            charge_signal=w["pred_q10"].values, discharge_signal=w["pred_q90"].values,
        )
        naive_signal  = df["marginal_price"].iloc[start_idx - len(w):start_idx].values if start_idx >= len(w) else w["pred_price"].values
        naive_dispatch = lmp_arbitrage_dispatch(naive_signal, charge_pct=25, discharge_pct=75)
        ml_actual_dispatch = lmp_arbitrage_dispatch(
            w["actual"].values, charge_pct=charge_pct, discharge_pct=discharge_pct,
        )

        strategies = {
            "ml":       pred_dispatch,
            "ml_fixed": ml_fixed_dispatch,
            "naive":    naive_dispatch,
            "ml_actual": ml_actual_dispatch,
        }
        rev = replay_at_actual_prices(w, strategies)

        oracle_revenue = lp_oracle(w["actual"].values)
        capture_ratio  = (min(rev["ml"] / oracle_revenue, 1.0) * 100) if oracle_revenue > 1e-6 else 0.0

        rank_corr, _ = spearmanr(w["pred_price"].values, w["actual"].values)
        rank_acc     = ranking_accuracy(w["pred_price"].values, w["actual"].values)
        pred_spread  = w["pred_q90"].values - w["pred_q10"].values

        combined_results.append({
            "start_time":        pw["start_time"],
            "price_mae":         pw["mae"],
            "realized_revenue":  rev["ml"],
            "naive_revenue":     rev["naive"],
            "ml_fixed_revenue":  rev["ml_fixed"],
            "ml_actual_revenue": rev["ml_actual"],
            "oracle_revenue":    oracle_revenue,
            "capture_ratio":     capture_ratio,
            "charge_pct":        charge_pct,
            "discharge_pct":     discharge_pct,
            "window_df":         w,
            "pred_dispatch":     pred_dispatch,
            "naive_dispatch":    naive_dispatch,
            "rank_corr":         rank_corr,
            "rank_acc":          rank_acc,
            "avg_q_spread":      np.mean(pred_spread),
            "actual_spread":     w["actual"].max() - w["actual"].min(),
        })

    return combined_results

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(combined_results):
    ml_revs    = np.array([r["realized_revenue"]  for r in combined_results])
    naive_revs = np.array([r["naive_revenue"]     for r in combined_results])
    fix_revs   = np.array([r["ml_fixed_revenue"]  for r in combined_results])
    act_revs   = np.array([r["ml_actual_revenue"] for r in combined_results])

    total_oracle = np.sum([r["oracle_revenue"] for r in combined_results])

    def safe_stat(a, b):
        try:
            w_stat, p_w = wilcoxon(a - b)
            t_stat, p_t = ttest_rel(a, b)
        except Exception:
            w_stat = p_w = t_stat = p_t = float("nan")
        return p_w, p_t

    def sig_label(p):
        if np.isnan(p):  return "n/a"
        if p < 0.001:    return "p < 0.001 ***"
        if p < 0.01:     return f"p = {p:.3f} **"
        if p < 0.05:     return f"p = {p:.3f} *"
        return                   f"p = {p:.3f} (not significant)"

    p_w_naive, p_t_naive = safe_stat(ml_revs, naive_revs)
    p_w_fix,   p_t_fix   = safe_stat(ml_revs, fix_revs)
    p_w_act,   p_t_act   = safe_stat(act_revs, ml_revs)

    total_ml     = ml_revs.sum()
    total_naive  = naive_revs.sum()
    total_ml_fix = fix_revs.sum()
    total_ml_act = act_revs.sum()

    # avg_capture       = np.mean([r["capture_ratio"] for r in combined_results])
    capture_ml_learned = (total_ml / total_oracle * 100) if total_oracle > 0 else 0
    capture_ml_fixed  = (total_ml_fix / total_oracle * 100) if total_oracle > 0 else 0
    capture_ml_actual = (total_ml_act / total_oracle * 100) if total_oracle > 0 else 0
    capture_naive     = (total_naive  / total_oracle * 100) if total_oracle > 0 else 0

    avg_q_spread      = np.mean([r["avg_q_spread"]  for r in combined_results])
    avg_act_spread    = np.mean([r["actual_spread"] for r in combined_results])
    compression_ratio = avg_q_spread / avg_act_spread if avg_act_spread > 0 else 0

    rank_arr     = [r["rank_corr"] for r in combined_results if not np.isnan(r["rank_corr"])]
    rank_acc_arr = [r["rank_acc"]  for r in combined_results]

    lift_from_fixed = ((total_ml - total_ml_fix) / total_ml_fix * 100) if total_ml_fix != 0 else 0
    lift_over_naive = ((total_ml - total_naive)   / total_naive  * 100) if total_naive  != 0 else 0
    forecast_loss   = total_ml_act - total_ml

    avg_mae = np.mean([r["price_mae"] for r in combined_results])

    print(f"\n{'═'*65}")
    print(f"   BESS LMP Arbitrage Backtest — Quantile XGBoost")
    print(f"   ({len(combined_results)} test windows, 1 window = 24 h)")
    print(f"{'═'*65}")
    print(f"\n   FORECAST QUALITY")
    print(f"{'─'*65}")
    print(f"   Avg q50 MAE:                        {avg_mae:>10.2f}  €/MWh")
    print(f"   Avg Rank Correlation (Spearman ρ):  {np.mean(rank_arr):>10.3f}")
    print(f"   Avg Ranking Accuracy (top/bot 25%): {np.mean(rank_acc_arr):>10.1%}")
    print(f"   Quantile Spread (q90−q10 predicted):{avg_q_spread:>10.2f}  €/MWh")
    print(f"   Actual Price Range (mean):          {avg_act_spread:>10.2f}  €/MWh")
    print(f"   Spread Coverage Ratio:              {compression_ratio:>10.3f}")
    print(f"   (1.0 = perfect; <1.0 = under-dispersed; >1.0 = over-dispersed)")
    print(f"\n   FINANCIAL PERFORMANCE")
    print(f"{'─'*65}")
    print(f"   {'Strategy':<40} {'Revenue (€)':>10}  {'Capture':>7}")
    print(f"   {'─'*57}")
    print(f"   {'LP Oracle (perfect foresight)':<40} {total_oracle:>10.3f}  {'100.0%':>7}")
    print(f"   {'ML + Learned Thresholds':<40} {total_ml:>10.3f}  {capture_ml_learned:>6.1f}%")
    print(f"   {'ML + Fixed Thresholds (p25/p75)':<40} {total_ml_fix:>10.3f}  {capture_ml_fixed:>6.1f}%")
    print(f"   {'ML Logic on Actual Prices':<40} {total_ml_act:>10.3f}  {capture_ml_actual:>6.1f}%")
    print(f"   {'Naive Persistence Baseline':<40} {total_naive:>10.3f}  {capture_naive:>6.1f}%")
    print(f"\n   COMPARATIVE LIFTS")
    print(f"{'─'*65}")
    print(f"   ML over Naive Baseline:             {lift_over_naive:>+10.2f} %")
    print(f"   Threshold Learning over Fixed:      {lift_from_fixed:>+10.2f} %")
    print(f"   Opportunity Cost of Forecast Error: {forecast_loss:>10.3f}  €")
    print(f"\n   STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank / paired t-test)")
    print(f"{'─'*65}")
    print(f"   ML Learned vs Naive:")
    print(f"      Wilcoxon: {sig_label(p_w_naive)}")
    print(f"      t-test:   {sig_label(p_t_naive)}")
    print(f"   ML Learned vs ML Fixed (threshold learning value):")
    print(f"      Wilcoxon: {sig_label(p_w_fix)}")
    print(f"      t-test:   {sig_label(p_t_fix)}")
    print(f"   ML Actual vs ML Learned (cost of forecast error):")
    print(f"      Wilcoxon: {sig_label(p_w_act)}")
    print(f"      t-test:   {sig_label(p_t_act)}")
    print(f"\n   INTERPRETATION")
    print(f"{'─'*65}")

    print(f"k=6: {np.mean([ranking_accuracy(r['window_df']['pred_price'].values, r['window_df']['actual'].values, k=6) for r in combined_results]):.1%}")
    print(f"k=4: {np.mean([ranking_accuracy(r['window_df']['pred_price'].values, r['window_df']['actual'].values, k=4) for r in combined_results]):.1%}")
    print(f"k=8: {np.mean([ranking_accuracy(r['window_df']['pred_price'].values, r['window_df']['actual'].values, k=8) for r in combined_results]):.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

action_colors = {"charge": "#2196F3", "discharge": "#F44336", "idle": "#B0BEC5"}


def plot_overview(df, combined_results, test_start_idx):
    fig, (ax_price, ax_rev, ax_cap) = plt.subplots(3, 1, figsize=(22, 20))

    full_actual = df["marginal_price"].iloc[test_start_idx + META_SPLIT * STEP:]
    ax_price.plot(full_actual.index, full_actual.values, color="black", lw=1.8, alpha=0.55, label="Actual LMP")
    for res in combined_results:
        wdf = res["window_df"]
        ax_price.fill_between(wdf.index, wdf["pred_q10"], wdf["pred_q90"], color="royalblue", alpha=0.04)
        ax_price.plot(wdf.index, wdf["pred_price"], color="royalblue", alpha=0.18, lw=1.2)
    last = combined_results[-1]["window_df"]
    ax_price.fill_between(last.index, last["pred_q10"], last["pred_q90"], color="crimson", alpha=0.15, label="Latest q10–q90 interval")
    ax_price.plot(last.index, last["pred_price"], color="crimson", lw=2.5, label="Latest q50 Forecast")
    ax_price.set_title("Sliding 24 h LMP Forecast — Quantile XGBoost (q10/q50/q90)", fontsize=16, fontweight="bold")
    ax_price.set_ylabel("LMP (€/MWh)", fontsize=13)
    ax_price.legend(fontsize=12)
    ax_price.grid(True, linestyle="--", alpha=0.4)
    ax_price.tick_params(labelsize=11)

    n_res = len(combined_results)
    x     = np.arange(n_res)
    step  = max(1, n_res // 20)
    tick_idx = np.arange(0, n_res, step)
    labels   = [combined_results[i]["start_time"].strftime("%m-%d") for i in tick_idx]
    w_bar    = 0.28
    ax_rev.bar(x - w_bar, [r["oracle_revenue"]  for r in combined_results], w_bar, label="Oracle (LP)",           color="#90A4AE")
    ax_rev.bar(x,         [r["naive_revenue"]    for r in combined_results], w_bar, label="Naive (actual prices)", color="#42A5F5")
    ax_rev.bar(x + w_bar, [r["realized_revenue"] for r in combined_results], w_bar, label="ML Quantile dispatch",  color="#26A69A")
    ax_rev.axhline(0, color="black", lw=0.8)
    ax_rev.set_ylabel("Revenue (€)", fontsize=13)
    ax_rev.set_title("BESS Arbitrage Revenue — Oracle vs Naive vs ML Quantile Dispatch", fontsize=16, fontweight="bold")
    ax_rev.set_xticks(tick_idx)
    ax_rev.set_xticklabels(labels, fontsize=10)
    ax_rev.legend(fontsize=12)
    ax_rev.grid(axis="y", linestyle="--", alpha=0.4)
    ax_rev.tick_params(labelsize=11)

    cap_vals = [r["capture_ratio"] for r in combined_results]
    mae_vals = [r["price_mae"]     for r in combined_results]
    ax_cap2  = ax_cap.twinx()
    ax_cap.plot(x, cap_vals, color="#EF5350", marker="o", markersize=3, lw=2, label="Capture Ratio (%)")
    ax_cap.fill_between(x, cap_vals, alpha=0.12, color="#EF5350")
    ax_cap.axhline(100, color="#90A4AE", lw=1, linestyle="--", label="Oracle = 100%")
    ax_cap.set_ylabel("Capture Ratio (%)", color="#EF5350", fontsize=13)
    ax_cap.set_ylim(0, 115)
    ax_cap2.plot(x, mae_vals, color="#5C6BC0", marker="s", markersize=3, lw=1.5, linestyle=":", label="q50 MAE (€/MWh)")
    ax_cap2.set_ylabel("Price MAE (€/MWh)", color="#5C6BC0", fontsize=13)
    ax_cap.set_title("ML Capture Ratio vs Oracle  |  MAE overlay", fontsize=16, fontweight="bold")
    ax_cap.set_xticks(tick_idx)
    ax_cap.set_xticklabels(labels, fontsize=10)
    lines1, labs1 = ax_cap.get_legend_handles_labels()
    lines2, labs2 = ax_cap2.get_legend_handles_labels()
    ax_cap.legend(lines1 + lines2, labs1 + labs2, fontsize=12)
    ax_cap.grid(True, linestyle="--", alpha=0.4)
    ax_cap.tick_params(labelsize=11)
    ax_cap2.tick_params(labelsize=11)

    plt.tight_layout()
    plt.show()


def plot_dispatch_detail(combined_results, n_detail=15):
    detail_results = combined_results[-n_detail:]
    fig, axes = plt.subplots(len(detail_results), 1, figsize=(16, 5 * len(detail_results)))
    if len(detail_results) == 1:
        axes = [axes]

    for i, res in enumerate(detail_results):
        wdf  = res["window_df"]
        disp = res["pred_dispatch"]
        ax   = axes[i]
        ax2  = ax.twinx()
        hrs  = np.arange(len(wdf))

        ax2.fill_between(hrs, wdf["pred_q10"].values, wdf["pred_q90"].values, color="royalblue", alpha=0.15, label="q10–q90 interval")
        ax2.plot(hrs, wdf["actual"].values,     color="black",   lw=1.5, alpha=0.6, label="Actual LMP")
        ax2.plot(hrs, wdf["pred_price"].values, color="crimson", lw=1.5, linestyle=":", label="q50 Forecast")
        ax2.set_ylabel("€/MWh", color="gray", fontsize=11)
        ax2.tick_params(labelsize=10)

        bar_c = [action_colors[a] for a in disp["action"]]
        ax.bar(hrs, disp["soc"].values, color=bar_c, alpha=0.72)
        ax.set_ylim(0, CAPACITY_MWH * 1.15)
        ax.set_ylabel("State of Charge (MWh)", fontsize=11)
        ax.set_xticks(hrs)
        ax.set_xticklabels([t.strftime("%d %b %H:%M") for t in wdf.index], rotation=45, fontsize=8)
        ax.tick_params(labelsize=10)

        patches = [mpatches.Patch(color=c, label=a.capitalize()) for a, c in action_colors.items()]
        h2, l2  = ax2.get_legend_handles_labels()
        ax.legend(handles=patches + h2, fontsize=10, loc="upper left")

        thr_str = f"p{res['charge_pct']:.0f}/p{res['discharge_pct']:.0f} (learned)" if USE_LEARNED_THRESHOLDS else "p25/p75 (fixed)"
        ax.set_title(
            f"Window {len(combined_results) - len(detail_results) + i + 1}  |  "
            f"{res['start_time'].strftime('%Y-%m-%d')}  |  "
            f"q50 MAE = {res['price_mae']:.1f} €/MWh  |  "
            f"ML Revenue = {res['realized_revenue']:.3f} €  |  "
            f"Oracle = {res['oracle_revenue']:.3f} €  |  "
            f"Capture = {res['capture_ratio']:.1f}%  |  "
            f"Thresholds: {thr_str}",
            fontsize=11,
        )
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_mae_vs_capture(combined_results):
    mae_arr      = np.array([r["price_mae"]     for r in combined_results])
    capture_arr  = np.array([r["capture_ratio"] for r in combined_results])
    rank_arr     = np.array([r["rank_corr"]     for r in combined_results])

    r_mae,  p_mae  = pearsonr(mae_arr,  capture_arr)
    r_rank, p_rank = pearsonr(rank_arr, capture_arr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.scatter(mae_arr, capture_arr, color="#90A4AE", s=25, alpha=0.6)
    m, b = np.polyfit(mae_arr, capture_arr, 1)
    xl = np.linspace(mae_arr.min(), mae_arr.max(), 100)
    ax1.plot(xl, m*xl + b, color="#EF5350", lw=2, label=f"r = {r_mae:+.3f}  (p = {p_mae:.3f})")
    ax1.set_xlabel("q50 MAE (€/MWh)", fontsize=13)
    ax1.set_ylabel("Capture Ratio (%)", fontsize=13)
    ax1.set_title("MAE vs Capture Ratio\n(standard metric)", fontweight="bold", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.tick_params(labelsize=11)

    ax2.scatter(rank_arr, capture_arr, color="#42A5F5", s=25, alpha=0.6)
    m2, b2 = np.polyfit(rank_arr, capture_arr, 1)
    xl2 = np.linspace(rank_arr.min(), rank_arr.max(), 100)
    ax2.plot(xl2, m2*xl2 + b2, color="#EF5350", lw=2, label=f"r = {r_rank:+.3f}  (p = {p_rank:.4f})")
    ax2.set_xlabel("Price Rank Correlation (Spearman ρ)", fontsize=13)
    ax2.set_ylabel("Capture Ratio (%)", fontsize=13)
    ax2.set_title("Rank Correlation vs Capture Ratio\n(economically relevant metric)", fontweight="bold", fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.tick_params(labelsize=11)

    plt.suptitle("MAE vs Rank Correlation as Dispatch Performance Predictors", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_capture_ratio(combined_results):
    dates        = [r["start_time"] for r in combined_results]
    capture_vals = [r["capture_ratio"] for r in combined_results]
    rolling_cap  = pd.Series(capture_vals).rolling(14, center=True).mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.scatter(dates, capture_vals, color="#EF5350", s=8, alpha=0.4, label="Daily Capture Ratio")
    ax.plot(dates, rolling_cap, color="#B71C1C", lw=2.5, label="14-day Rolling Average")
    ax.axhline(np.mean(capture_vals), color="#90A4AE", lw=1.5, linestyle="--", label=f"Mean = {np.mean(capture_vals):.1f}%")
    ax.axhline(100, color="black", lw=0.8, linestyle=":", alpha=0.4)
    ax.set_title("ML Capture Ratio Over Time (2025)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Capture Ratio (%)", fontsize=13)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylim(0, 115)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig("figure5_capture_ratio.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_dispatch_heatmap(combined_results):
    charge_matrix    = np.zeros((24, 12))
    discharge_matrix = np.zeros((24, 12))
    count_matrix     = np.zeros((24, 12))

    for r in combined_results:
        month = r["start_time"].month - 1
        disp  = r["pred_dispatch"]
        wdf   = r["window_df"]
        for h in range(len(disp)):
            hour   = wdf.index[h].hour
            action = disp.iloc[h]["action"]
            count_matrix[hour, month] += 1
            if action == "charge":
                charge_matrix[hour, month] += 1
            elif action == "discharge":
                discharge_matrix[hour, month] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        charge_freq    = np.where(count_matrix > 0, charge_matrix    / count_matrix, 0)
        discharge_freq = np.where(count_matrix > 0, discharge_matrix / count_matrix, 0)

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

    hm1 = sns.heatmap(charge_freq, ax=ax1, cmap="Blues", vmin=0, vmax=1,
                      xticklabels=months, yticklabels=[f"{h:02d}:00" for h in range(24)],
                      cbar_kws={"label": "Charge Frequency"})
    hm1.collections[0].colorbar.ax.tick_params(labelsize=11)
    hm1.collections[0].colorbar.set_label("Charge Frequency", fontsize=12)
    ax1.set_xlabel("Month", fontsize=13)
    ax1.set_ylabel("Hour of Day", fontsize=13)
    ax1.tick_params(labelsize=11)

    hm2 = sns.heatmap(discharge_freq, ax=ax2, cmap="Reds", vmin=0, vmax=1,
                      xticklabels=months, yticklabels=[f"{h:02d}:00" for h in range(24)],
                      cbar_kws={"label": "Discharge Frequency"})
    hm2.collections[0].colorbar.ax.tick_params(labelsize=11)
    hm2.collections[0].colorbar.set_label("Discharge Frequency", fontsize=12)
    ax2.set_xlabel("Month", fontsize=13)
    ax2.set_ylabel("Hour of Day", fontsize=13)
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig("figure6_dispatch_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.show()

def plot_threshold_heatmap(combined_results):
    res    = combined_results[len(combined_results) // 2]
    prices = res["window_df"]["actual"].values
    date   = res["start_time"].strftime("%Y-%m-%d")

    charge_pcts    = list(range(10, 50, 5))
    discharge_pcts = list(range(55, 100, 5))
    rev_matrix     = np.full((len(charge_pcts), len(discharge_pcts)), np.nan)

    for i, cp in enumerate(charge_pcts):
        for j, dp in enumerate(discharge_pcts):
            if dp > cp + 5:
                rev_matrix[i, j] = lmp_arbitrage_dispatch(
                    prices, charge_pct=cp, discharge_pct=dp
                )["revenue"].sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(rev_matrix, aspect="auto", origin="lower",
                   cmap="RdYlGn", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Revenue (€)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    ax.set_xticks(range(len(discharge_pcts)))
    ax.set_xticklabels([f"p{d}" for d in discharge_pcts], rotation=45, fontsize=11)
    ax.set_yticks(range(len(charge_pcts)))
    ax.set_yticklabels([f"p{c}" for c in charge_pcts], fontsize=11)
    ax.set_xlabel("Discharge Threshold Percentile", fontsize=13)
    ax.set_ylabel("Charge Threshold Percentile", fontsize=13)

    best = np.unravel_index(np.nanargmax(rev_matrix), rev_matrix.shape)
    ax.plot(best[1], best[0], "r*", markersize=16,
            label=f"Optimal: charge=p{charge_pcts[best[0]]}, discharge=p{discharge_pcts[best[1]]} (€{rev_matrix[best]:.3f})")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("threshold_heatmap.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_dispatch_seasonal(combined_results):
    summer = next((r for r in combined_results if r["start_time"].month in [6, 7]), combined_results[len(combined_results) // 3])
    winter = next((r for r in combined_results if r["start_time"].month in [12, 1]), combined_results[0])

    action_colors = {"charge": "#2196F3", "discharge": "#F44336", "idle": "#B0BEC5"}
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    for ax, res, season in zip(axes, [summer, winter], ["Summer", "Winter"]):
        wdf  = res["window_df"]
        disp = res["pred_dispatch"]
        hrs  = np.arange(len(wdf))
        ax2  = ax.twinx()

        ax2.fill_between(hrs, wdf["pred_q10"], wdf["pred_q90"], color="royalblue", alpha=0.18, label="q10–q90 interval")
        ax2.plot(hrs, wdf["actual"],     color="black",   lw=2,   label="Actual LMP")
        ax2.plot(hrs, wdf["pred_price"], color="crimson", lw=1.5, linestyle="--", label="q50 Forecast")
        ax2.set_ylabel("LMP (€/MWh)", color="dimgray", fontsize=12)
        ax2.tick_params(labelsize=10)

        ax.bar(hrs, disp["soc"], color=[action_colors[a] for a in disp["action"]], alpha=0.75)
        ax.set_ylim(0, CAPACITY_MWH * 1.2)
        ax.set_ylabel("State of Charge (MWh)", fontsize=12)
        ax.set_xticks(hrs)
        ax.set_xticklabels([wdf.index[h].strftime("%H:%M") for h in hrs], rotation=45, fontsize=8)
        ax.set_xlabel("Hour", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2)

        patches = [mpatches.Patch(color=c, label=a.capitalize()) for a, c in action_colors.items()]
        h2, l2  = ax2.get_legend_handles_labels()
        ax.legend(handles=patches + h2, fontsize=11, loc="upper left", ncol=3)
        ax.set_title(
            f"{season} {res['start_time'].strftime('%Y-%m-%d')}  |  "
            f"q50 MAE = {res['price_mae']:.1f} €/MWh  |  "
            f"ML Revenue = {res['realized_revenue']:.2f} €  |  "
            f"Oracle = {res['oracle_revenue']:.2f} €  |  "
            f"Capture = {res['capture_ratio']:.1f}%",
            fontweight="bold", fontsize=13
        )

    plt.tight_layout()
    plt.savefig("dispatch_seasonal.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_load_forecast_seasonal(combined_results, df):
    summer = next((r for r in combined_results if r["start_time"].month in [6, 7]), combined_results[len(combined_results) // 3])
    winter = next((r for r in combined_results if r["start_time"].month in [12, 1]), combined_results[0])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # ← vertical
    for ax, (season, res) in zip(axes, [("Summer", summer), ("Winter", winter)]):
        wdf         = res["window_df"]
        hrs         = np.arange(len(wdf))
        actual_load = df["load"].reindex(wdf.index).values
        tick_pos    = hrs[::4]

        ax.plot(hrs, actual_load,             color="black",   lw=2,   label="Actual Load")
        ax.plot(hrs, wdf["pred_load"].values, color="#FF7043", lw=1.8, linestyle="--", label="Predicted Load")
        ax.fill_between(hrs, actual_load, wdf["pred_load"].values, color="#FF7043", alpha=0.10)
        ax.text(0.98, 0.95, f"MAE = {mean_absolute_error(actual_load, wdf['pred_load'].values):.1f} MW",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_ylabel("Load (MW)", fontsize=13)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([wdf.index[h].strftime("%H:%M") for h in tick_pos], rotation=45, fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig("load_forecast_seasonal.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_lmp_forecast_seasonal(combined_results):
    summer = next((r for r in combined_results if r["start_time"].month in [6, 7]), combined_results[len(combined_results) // 3])
    winter = next((r for r in combined_results if r["start_time"].month in [12, 1]), combined_results[0])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # ← vertical
    for ax, (season, res) in zip(axes, [("Summer", summer), ("Winter", winter)]):
        wdf      = res["window_df"]
        hrs      = np.arange(len(wdf))
        tick_pos = hrs[::4]
        spread   = np.mean(wdf["pred_q90"].values - wdf["pred_q10"].values)

        ax.fill_between(hrs, wdf["pred_q10"], wdf["pred_q90"], color="royalblue", alpha=0.18, label="q10–q90 interval")
        ax.plot(hrs, wdf["pred_q10"], color="royalblue", lw=1.0, linestyle=":", alpha=0.7)
        ax.plot(hrs, wdf["pred_q90"], color="royalblue", lw=1.0, linestyle=":", alpha=0.7, label="q10 / q90 bounds")
        ax.plot(hrs, wdf["pred_price"], color="crimson", lw=1.8, linestyle="--", label="q50 Forecast")
        ax.plot(hrs, wdf["actual"],     color="black",   lw=2,                   label="Actual LMP")
        ax.text(0.98, 0.95, f"q50 MAE = {res['price_mae']:.1f} €/MWh\nAvg spread = {spread:.1f} €/MWh",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_ylabel("LMP (€/MWh)", fontsize=13)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([wdf.index[h].strftime("%H:%M") for h in tick_pos], rotation=45, fontsize=10)
        ax.legend(fontsize=12, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig("lmp_forecast_seasonal.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_acf_lag_selection(df, test_start_idx):
    price_lags = [1, 24, 168]
    load_lags  = [1, 2, 3, 6, 12, 24, 48, 168]

    train_price = df["marginal_price"].iloc[:test_start_idx]
    train_load  = df["load"].iloc[:test_start_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    plot_acf(train_price, lags=200, ax=ax1, color="#EF5350", alpha=0.05, zero=False)
    ax1.set_title("Autocorrelation Function Marginal Price (LMP)", fontsize=15, fontweight="bold")
    ax1.set_xlabel("Lag (hours)", fontsize=13)
    ax1.set_ylabel("ACF", fontsize=13)
    ax1.tick_params(labelsize=11)
    for lag in price_lags:
        ax1.axvline(lag, color="#1565C0", linestyle="--", alpha=0.6, lw=1.2)
    ax1.axvline(price_lags[0], color="#1565C0", linestyle="--", alpha=0.6, lw=1.2, label=f"Selected lags: {price_lags}")
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)

    plot_acf(train_load, lags=200, ax=ax2, color="#42A5F5", alpha=0.05, zero=False)
    ax2.set_title("Autocorrelation Function Load", fontsize=15, fontweight="bold")
    ax2.set_xlabel("Lag (hours)", fontsize=13)
    ax2.set_ylabel("ACF", fontsize=13)
    ax2.tick_params(labelsize=11)
    for lag in load_lags:
        ax2.axvline(lag, color="#1565C0", linestyle="--", alpha=0.6, lw=1.2)
    ax2.axvline(load_lags[0], color="#1565C0", linestyle="--", alpha=0.6, lw=1.2, label=f"Selected lags: {load_lags}")
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("acf_lag_selection.pdf", bbox_inches="tight", dpi=300)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Load and prepare data
    df = load_network_data(NETWORK_FILES, BUS, CARRIERS)
    df = engineer_features(df)

    test_start_idx = len(df) - (N_WINDOWS * STEP) - HORIZON

    # 2. Train models
    print("=== Training Load Model ===")
    load_model = train_load_model(df[load_features].iloc[:test_start_idx], df["load"].iloc[:test_start_idx])

    q50_model, q10_model, q90_model = train_price_models(
        df[reg_features].iloc[:test_start_idx],
        df["marginal_price"].iloc[:test_start_idx],
    )

    thr_charge_model, thr_discharge_model = train_threshold_models(df, test_start_idx)

    # 3. Generate forecasts and run backtest
    price_windows      = generate_price_windows(df, test_start_idx, load_model, q50_model, q10_model, q90_model)
    test_price_windows = price_windows[META_SPLIT:]
    combined_results   = run_backtest(df, test_price_windows, thr_charge_model, thr_discharge_model, test_start_idx)

    # 4. Summary
    compute_summary(combined_results)

    # 5. Plots
    plot_overview(df, combined_results, test_start_idx)
    plot_dispatch_detail(combined_results)
    plot_mae_vs_capture(combined_results)
    plot_capture_ratio(combined_results)
    plot_dispatch_heatmap(combined_results)
    plot_threshold_heatmap(combined_results)
    plot_dispatch_seasonal(combined_results)
    plot_load_forecast_seasonal(combined_results, df)
    plot_lmp_forecast_seasonal(combined_results)
    # plot_acf_lag_selection(df, test_start_idx)