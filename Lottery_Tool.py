#!/usr/bin/env python3
"""
Lottery Statistical Auditor, AI Predictor & Data Scraper
=========================================================
A single-file Streamlit app that:
  1. Downloads / refreshes historical draw data (Powerball, Mega Millions, SuperLotto Plus)
  2. Runs chi-square fairness tests, confidence intervals, Poisson wait-time analysis
  3. Provides hot/cold number breakdowns with frequency charts
  4. Trains an XGBoost time-series classifier with honest backtest evaluation
  5. Generates AI-recommended tickets (with appropriate disclaimers)

Data sources:
  - Powerball & Mega Millions: data.ny.gov (official open-data CSV exports)
  - SuperLotto Plus: lottery.net (scraped via Playwright)

Setup:
    pip install streamlit pandas numpy scipy statsmodels xgboost requests requests_cache plotly
    # Only needed for SuperLotto Plus scraping:
    pip install playwright && playwright install chromium

Run:
    streamlit run Lottery_Tool.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chisquare, poisson
import statsmodels.stats.proportion as smp
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import requests
import requests_cache
import csv
import io
import os
from datetime import datetime
from pathlib import Path

# ============================================================
# 0. CONFIGURATION
# ============================================================

LOTTERY_CONFIGS = {
    "Mega Millions": {
        "file": "MEGA_Millions.csv",
        "date_col": "Draw Date",
        "num_cols": ["Num1", "Num2", "Num3", "Num4", "Num5"],
        "mega_col": "Mega Ball",
        # 2017-10-28: last matrix change (70/25 format). Previous start_date of
        # 2025-04-08 was a future date that filtered out ALL data — fixed here.
        "start_date": "2017-10-28",
        "max_main": 70,
        "max_mega": 25,
    },
    "Powerball": {
        "file": "POWERBALL.csv",
        "date_col": "Draw Date",
        "num_cols": ["Num1", "Num2", "Num3", "Num4", "Num5"],
        "mega_col": "Powerball",
        "start_date": "2015-10-07",
        "max_main": 69,
        "max_mega": 26,
    },
    "SuperLotto Plus": {
        "file": "SuperLotto_Plus.csv",
        "date_col": "Draw Date",
        "num_cols": ["Num1", "Num2", "Num3", "Num4", "Num5"],
        "mega_col": "Mega",
        "start_date": "2000-06-03",
        "max_main": 47,
        "max_mega": 27,
    },
}

POWERBALL_CSV_URL = (
    "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
)
MEGA_MILLIONS_CSV_URL = (
    "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD"
)
SUPERLOTTO_BASE_URL = "https://www.lottery.net/california/superlotto-plus/numbers/{year}"
SUPERLOTTO_START_YEAR = 1986


# ============================================================
# 1. DATA SCRAPER / DOWNLOADER
# ============================================================

def _parse_date_tuple(date_str: str):
    """Parse MM/DD/YYYY into a sortable tuple. Handles leading day-names."""
    import re
    cleaned = re.sub(r"^[A-Za-z]+\s+", "", date_str.strip())
    parts = cleaned.split("/")
    if len(parts) == 3:
        return (parts[2].zfill(4), parts[0].zfill(2), parts[1].zfill(2))
    return (date_str,)


def download_powerball(progress_callback=None):
    """Download Powerball history from data.ny.gov."""
    output_file = LOTTERY_CONFIGS["Powerball"]["file"]
    if progress_callback:
        progress_callback("Downloading Powerball from data.ny.gov …")

    r = requests.get(POWERBALL_CSV_URL, timeout=60)
    r.raise_for_status()

    reader = csv.DictReader(io.StringIO(r.text))
    fieldnames = [
        "Draw Date", "Num1", "Num2", "Num3", "Num4", "Num5",
        "Powerball", "Multiplier",
    ]

    rows = []
    for row in reader:
        nums = row["Winning Numbers"].split()
        if len(nums) < 5:
            continue  # skip malformed rows
        rows.append({
            "Draw Date": row["Draw Date"],
            "Num1": nums[0], "Num2": nums[1], "Num3": nums[2],
            "Num4": nums[3], "Num5": nums[4],
            "Powerball": nums[5] if len(nums) > 5 else "",
            "Multiplier": row.get("Multiplier", ""),
        })

    rows.sort(key=lambda r: _parse_date_tuple(r["Draw Date"]))

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def download_mega_millions(progress_callback=None):
    """Download Mega Millions history from data.ny.gov."""
    output_file = LOTTERY_CONFIGS["Mega Millions"]["file"]
    if progress_callback:
        progress_callback("Downloading Mega Millions from data.ny.gov …")

    r = requests.get(MEGA_MILLIONS_CSV_URL, timeout=60)
    r.raise_for_status()

    reader = csv.DictReader(io.StringIO(r.text))
    fieldnames = [
        "Draw Date", "Num1", "Num2", "Num3", "Num4", "Num5",
        "Mega Ball", "Multiplier",
    ]

    rows = []
    for row in reader:
        nums = row["Winning Numbers"].split()
        if len(nums) < 5:
            continue
        rows.append({
            "Draw Date": row["Draw Date"],
            "Num1": nums[0], "Num2": nums[1], "Num3": nums[2],
            "Num4": nums[3], "Num5": nums[4],
            "Mega Ball": row.get("Mega Ball", ""),
            "Multiplier": row.get("Multiplier", ""),
        })

    rows.sort(key=lambda r: _parse_date_tuple(r["Draw Date"]))

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def download_superlotto(progress_callback=None):
    """Scrape SuperLotto Plus from lottery.net using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright is required for SuperLotto Plus scraping.\n"
            "Install with: pip install playwright && playwright install chromium"
        )

    output_file = LOTTERY_CONFIGS["SuperLotto Plus"]["file"]
    current_year = datetime.now().year
    fieldnames = [
        "Draw Date", "Draw Number", "Num1", "Num2", "Num3",
        "Num4", "Num5", "Mega",
    ]

    all_rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for year in range(SUPERLOTTO_START_YEAR, current_year + 1):
            if progress_callback:
                progress_callback(f"Scraping SuperLotto Plus — {year}")

            url = SUPERLOTTO_BASE_URL.format(year=year)
            try:
                page.goto(url, timeout=30_000)
                page.wait_for_selector("table", timeout=10_000)
            except Exception:
                continue

            table = page.query_selector("table")
            if not table:
                continue

            for row in table.query_selector_all("tr")[1:]:
                cells = row.query_selector_all("td")
                if len(cells) < 3:
                    continue
                date_text = cells[0].inner_text().strip().replace("\n", " ")
                draw_num = cells[1].inner_text().strip()
                numbers = cells[2].inner_text().strip().split()
                if len(numbers) >= 6:
                    all_rows.append({
                        "Draw Date": date_text,
                        "Draw Number": draw_num,
                        "Num1": numbers[0], "Num2": numbers[1],
                        "Num3": numbers[2], "Num4": numbers[3],
                        "Num5": numbers[4], "Mega": numbers[5],
                    })

        browser.close()

    all_rows.sort(
        key=lambda r: int(r["Draw Number"]) if r["Draw Number"].isdigit() else 0
    )

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(all_rows)


# ============================================================
# 2. DATA LOADING & VALIDATION
# ============================================================

def load_lottery_data(conf: dict) -> pd.DataFrame | None:
    """Load, validate, and filter a lottery CSV."""
    filepath = conf["file"]
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)

    # Validate required columns exist
    required = conf["num_cols"] + [conf["date_col"], conf["mega_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"CSV is missing columns: {missing}")
        return None

    # Clean date column (remove leading day-names like "Tuesday 01/02/2024")
    import re
    if df[conf["date_col"]].dtype == object:
        df[conf["date_col"]] = df[conf["date_col"]].str.replace(
            r"^[A-Za-z]+\s+", "", regex=True
        )
    df[conf["date_col"]] = pd.to_datetime(df[conf["date_col"]], errors="coerce")
    df = df.dropna(subset=[conf["date_col"]])

    # Filter to draws after the last matrix change
    df = df[df[conf["date_col"]] >= pd.to_datetime(conf["start_date"])].copy()
    df = df.sort_values(conf["date_col"]).reset_index(drop=True)

    # Coerce number columns to numeric
    for col in conf["num_cols"] + [conf["mega_col"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# 3. XGBOOST PREDICTOR (with backtest evaluation)
# ============================================================

def _build_features(df_lotto, max_main, max_mega):
    """
    Build the binary draw matrix and per-number feature rows.
    Returns (ml_df, df_binary, last_seen_dict).
    """
    draw_matrix = np.zeros((len(df_lotto), max_main), dtype=int)
    for i, nums in enumerate(df_lotto["Numbers"]):
        for n in nums:
            if 1 <= n <= max_main:
                draw_matrix[i, n - 1] = 1

    df_binary = pd.DataFrame(
        draw_matrix, columns=[f"Num_{i}" for i in range(1, max_main + 1)]
    )

    dataset_rows = []
    last_seen = {n: -100 for n in range(1, max_main + 1)}

    for draw_idx in range(10, len(df_lotto) - 1):
        recent_10 = df_binary.iloc[draw_idx - 10 : draw_idx].sum()
        recent_3 = df_binary.iloc[draw_idx - 3 : draw_idx].sum()
        next_draw = df_binary.iloc[draw_idx + 1]

        for n in range(1, max_main + 1):
            col = f"Num_{n}"
            if df_binary.iloc[draw_idx][col] == 1:
                last_seen[n] = draw_idx

            dataset_rows.append({
                "Number": n,
                "Gap": draw_idx - last_seen[n],
                "Freq_Last_10": recent_10[col],
                "Freq_Last_3": recent_3[col],
                "DrawIdx": draw_idx,
                "Target": next_draw[col],
            })

    ml_df = pd.DataFrame(dataset_rows)
    return ml_df, df_binary, last_seen


def train_and_predict(df_lotto, max_main, max_mega):
    """
    Train XGBoost on all data and predict probabilities for the next draw.
    Returns (top_5_main, predicted_mega, probability_df, backtest_results).
    """
    ml_df, df_binary, last_seen = _build_features(df_lotto, max_main, max_mega)

    feature_cols = ["Number", "Gap", "Freq_Last_10", "Freq_Last_3"]
    X = ml_df[feature_cols]
    y = ml_df["Target"]

    # --- Backtest: train on first 80%, test on last 20% ---
    split_draw = ml_df["DrawIdx"].quantile(0.8)
    train_mask = ml_df["DrawIdx"] <= split_draw
    test_mask = ml_df["DrawIdx"] > split_draw

    xgb_bt = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    xgb_bt.fit(X[train_mask], y[train_mask])
    bt_proba = xgb_bt.predict_proba(X[test_mask])[:, 1]
    bt_preds = (bt_proba > 0.5).astype(int)

    test_y = y[test_mask].values
    bt_accuracy = (bt_preds == test_y).mean()
    bt_precision = (
        bt_preds[test_y == 1].sum() / bt_preds.sum() if bt_preds.sum() > 0 else 0
    )
    baseline_rate = test_y.mean()  # random baseline = 5/max_main

    backtest = {
        "accuracy": bt_accuracy,
        "precision": bt_precision,
        "baseline_hit_rate": baseline_rate,
        "test_samples": len(test_y),
    }

    # --- Full model (train on everything) ---
    xgb_full = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    xgb_full.fit(X, y)

    # Predict next draw
    latest_idx = len(df_lotto) - 1
    recent_10_latest = df_binary.iloc[latest_idx - 9 : latest_idx + 1].sum()
    recent_3_latest = df_binary.iloc[latest_idx - 2 : latest_idx + 1].sum()

    pred_rows = []
    for n in range(1, max_main + 1):
        col = f"Num_{n}"
        gap = 0 if df_binary.iloc[latest_idx][col] == 1 else (latest_idx - last_seen[n])
        pred_rows.append({
            "Number": n,
            "Gap": gap,
            "Freq_Last_10": recent_10_latest[col],
            "Freq_Last_3": recent_3_latest[col],
        })

    df_pred = pd.DataFrame(pred_rows)
    proba = xgb_full.predict_proba(df_pred[feature_cols])[:, 1]
    df_pred["Probability"] = proba
    df_pred = df_pred.sort_values("Probability", ascending=False)
    top_5 = df_pred["Number"].head(5).tolist()

    # --- Mega Ball: frequency-weighted selection (NOT gambler's fallacy) ---
    valid_megas = pd.to_numeric(df_lotto["MegaBall"], errors="coerce").dropna()
    valid_megas = valid_megas[(valid_megas >= 1) & (valid_megas <= max_mega)].astype(int)

    if not valid_megas.empty:
        mega_counts = valid_megas.value_counts().reindex(
            range(1, max_mega + 1), fill_value=0
        )
        # Use historical frequency as a soft weight (most drawn numbers slightly favoured)
        weights = mega_counts.values.astype(float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(max_mega) / max_mega
        predicted_mega = int(mega_counts.index[np.argmax(weights)])
    else:
        predicted_mega = 1

    return top_5, predicted_mega, df_pred, backtest


# ============================================================
# 4. STREAMLIT APP
# ============================================================

def render_sidebar_scraper():
    """Sidebar: data download controls."""
    st.sidebar.header("📥 Data Manager")
    st.sidebar.markdown("Download or refresh lottery draw history.")

    if st.sidebar.button("Download Powerball"):
        with st.sidebar.status("Downloading…", expanded=True) as status:
            try:
                n = download_powerball(progress_callback=lambda m: status.update(label=m))
                status.update(label=f"✅ Powerball: {n} draws saved", state="complete")
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")

    if st.sidebar.button("Download Mega Millions"):
        with st.sidebar.status("Downloading…", expanded=True) as status:
            try:
                n = download_mega_millions(progress_callback=lambda m: status.update(label=m))
                status.update(label=f"✅ Mega Millions: {n} draws saved", state="complete")
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")

    if st.sidebar.button("Download SuperLotto Plus"):
        with st.sidebar.status("Scraping…", expanded=True) as status:
            try:
                n = download_superlotto(progress_callback=lambda m: status.update(label=m))
                status.update(label=f"✅ SuperLotto Plus: {n} draws saved", state="complete")
            except ImportError as e:
                st.sidebar.error(str(e))
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")

    # Show data freshness
    st.sidebar.markdown("---")
    st.sidebar.markdown("**CSV Status:**")
    for name, conf in LOTTERY_CONFIGS.items():
        path = Path(conf["file"])
        if path.exists():
            mod = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.sidebar.success(f"{name}: {mod}")
        else:
            st.sidebar.warning(f"{name}: not downloaded")


def render_main():
    """Main page: analysis and predictions."""
    st.set_page_config(page_title="Lottery Auditor & AI Predictor", layout="wide")
    st.title("📊 Lottery Statistical Auditor & AI Predictor")
    st.caption(
        "Using data only from the **last matrix change date** to ensure mathematical validity. "
        "⚠️ Lottery draws are independent random events — no model can reliably predict future outcomes."
    )

    render_sidebar_scraper()

    lottery_choice = st.selectbox("Select Lottery:", list(LOTTERY_CONFIGS.keys()))
    conf = LOTTERY_CONFIGS[lottery_choice]

    df = load_lottery_data(conf)
    if df is None or len(df) == 0:
        st.error(
            f"**{conf['file']}** not found or contains no valid draws after {conf['start_date']}.\n\n"
            f"Use the **📥 Data Manager** in the sidebar to download it first."
        )
        return

    total_draws = len(df)
    date_range = f"{df[conf['date_col']].min().date()} → {df[conf['date_col']].max().date()}"
    st.info(f"Analyzing **{total_draws} draws** from **{date_range}** (since matrix change {conf['start_date']}).")

    main_df = df[conf["num_cols"]]
    main_nums = main_df.values.flatten()
    main_nums = main_nums[~np.isnan(main_nums)].astype(int)
    mega_nums = pd.to_numeric(df[conf["mega_col"]], errors="coerce").dropna().astype(int).values

    # ------------------------------------------------------------------
    # TAB LAYOUT
    # ------------------------------------------------------------------
    tab_fair, tab_deep, tab_hotcold, tab_ai = st.tabs([
        "⚖️ Fairness Tests", "🔍 Number Deep Dive", "🔥 Hot / Cold", "🤖 AI Predictor"
    ])

    # ==================== TAB 1: FAIRNESS ====================
    with tab_fair:
        st.subheader("Chi-Square Goodness-of-Fit")
        st.markdown("*H₀: The machine draws each number with equal probability.*")

        main_counts = (
            pd.Series(main_nums)
            .value_counts()
            .reindex(range(1, conf["max_main"] + 1), fill_value=0)
        )
        expected_main = main_counts.sum() / conf["max_main"]
        chi2_m, p_main = chisquare(main_counts, f_exp=[expected_main] * conf["max_main"])

        mega_counts = (
            pd.Series(mega_nums)
            .value_counts()
            .reindex(range(1, conf["max_mega"] + 1), fill_value=0)
        )
        expected_mega = mega_counts.sum() / conf["max_mega"]
        chi2_mg, p_mega = chisquare(mega_counts, f_exp=[expected_mega] * conf["max_mega"])

        c1, c2 = st.columns(2)
        c1.metric("Main Numbers χ² P-Value", f"{p_main:.4f}")
        c2.metric("Mega Ball χ² P-Value", f"{p_mega:.4f}")

        if expected_main < 5 or expected_mega < 5:
            st.warning(
                "⚠️ Some expected cell counts are below 5, which can make the chi-square "
                "test unreliable. Consider grouping numbers or collecting more data."
            )

        if p_main < 0.05 or p_mega < 0.05:
            st.warning("⚠️ P < 0.05 — statistically significant deviation from uniformity detected.")
        else:
            st.success("✅ P ≥ 0.05 — no evidence to reject uniformity. The draws appear fair.")

        # Frequency bar chart
        fig_main = px.bar(
            x=main_counts.index, y=main_counts.values,
            labels={"x": "Number", "y": "Times Drawn"},
            title="Main Number Frequency Distribution",
        )
        fig_main.add_hline(y=expected_main, line_dash="dash", line_color="red",
                           annotation_text=f"Expected ({expected_main:.1f})")
        st.plotly_chart(fig_main, use_container_width=True)

        fig_mega = px.bar(
            x=mega_counts.index, y=mega_counts.values,
            labels={"x": "Mega Ball", "y": "Times Drawn"},
            title="Mega Ball Frequency Distribution",
        )
        fig_mega.add_hline(y=expected_mega, line_dash="dash", line_color="red",
                           annotation_text=f"Expected ({expected_mega:.1f})")
        st.plotly_chart(fig_mega, use_container_width=True)

    # ==================== TAB 2: DEEP DIVE ====================
    with tab_deep:
        st.subheader("Analyze a Specific Number")
        target_num = st.number_input(
            "Main number to analyze:", min_value=1, max_value=conf["max_main"], value=7
        )

        draws_with = main_df.isin([target_num]).any(axis=1)
        k = int(draws_with.sum())
        expected_p = 5 / conf["max_main"]

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("#### Confidence Intervals")
            st.write(f"Number **{target_num}** appeared **{k}** times in {total_draws} draws.")
            ci_lo, ci_hi = smp.proportion_confint(k, total_draws, alpha=0.05, method="wilson")
            obs_p = k / total_draws if total_draws > 0 else 0

            st.write(f"- Observed rate: **{obs_p:.4f}**")
            st.write(f"- Expected rate: **{expected_p:.4f}**")
            st.write(f"- 95 % CI: **[{ci_lo:.4f}, {ci_hi:.4f}]**")

            if ci_lo <= expected_p <= ci_hi:
                st.success(f"Number {target_num} is within expected bounds.")
            else:
                st.warning(f"Number {target_num} is outside the expected confidence interval.")

        with c4:
            st.markdown("#### Wait-Time Analysis")
            expected_wait = 1 / expected_p if expected_p > 0 else float("inf")
            draw_indices = np.where(draws_with)[0]
            if len(draw_indices) > 1:
                gaps = np.diff(draw_indices)
                avg_wait = gaps.mean()
                st.write(f"- Expected wait (geometric): **{expected_wait:.1f}** draws")
                st.write(f"- Actual average wait: **{avg_wait:.1f}** draws")
                st.write(f"- Current gap since last seen: **{len(df) - 1 - draw_indices[-1]}** draws")
            else:
                st.write("Not enough appearances to compute wait times.")

            st.markdown("---")
            draws_per_year = 104
            lam = draws_per_year * expected_p
            st.write(f"Poisson λ for a year ({draws_per_year} draws): **{lam:.1f}**")
            prob_exact = poisson.pmf(round(lam), lam) * 100
            prob_more = (1 - poisson.cdf(round(lam), lam)) * 100
            st.write(f"P(exactly {round(lam)} times) = **{prob_exact:.2f}%**")
            st.write(f"P(more than {round(lam)} times) = **{prob_more:.2f}%**")

    # ==================== TAB 3: HOT / COLD ====================
    with tab_hotcold:
        st.subheader("Hot & Cold Numbers")
        window = st.slider("Look-back window (draws):", 10, min(200, total_draws), 30)

        recent_df = main_df.iloc[-window:]
        recent_flat = recent_df.values.flatten()
        recent_flat = recent_flat[~np.isnan(recent_flat)].astype(int)
        recent_counts = (
            pd.Series(recent_flat)
            .value_counts()
            .reindex(range(1, conf["max_main"] + 1), fill_value=0)
            .sort_values(ascending=False)
        )

        hot = recent_counts.head(10)
        cold = recent_counts.tail(10).sort_values()

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### 🔥 Hottest (most drawn)")
            fig_hot = px.bar(x=hot.index.astype(str), y=hot.values,
                             labels={"x": "Number", "y": "Count"}, color_discrete_sequence=["#e74c3c"])
            st.plotly_chart(fig_hot, use_container_width=True)

        with c6:
            st.markdown("#### ❄️ Coldest (least drawn)")
            fig_cold = px.bar(x=cold.index.astype(str), y=cold.values,
                              labels={"x": "Number", "y": "Count"}, color_discrete_sequence=["#3498db"])
            st.plotly_chart(fig_cold, use_container_width=True)

    # ==================== TAB 4: AI PREDICTOR ====================
    with tab_ai:
        st.subheader("XGBoost Time-Series Predictor")
        st.markdown(
            "Treats each number's appearance as a binary classification task using "
            "gap (draws since last seen) and rolling frequency features."
        )

        num_tickets = st.slider("Number of ticket suggestions:", 1, 5, 1)

        if st.button("🚀 Run XGBoost Predictor"):
            if total_draws < 20:
                st.error("Need at least 20 draws to train the model.")
                return

            with st.spinner("Engineering features & training model …"):
                df["Numbers"] = df[conf["num_cols"]].values.tolist()
                df["MegaBall"] = df[conf["mega_col"]]

                top_5, pred_mega, prob_df, backtest = train_and_predict(
                    df, conf["max_main"], conf["max_mega"]
                )

            # --- Backtest report ---
            st.markdown("#### 📈 Model Backtest (80/20 split)")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Overall Accuracy", f"{backtest['accuracy']:.1%}")
            bc2.metric("Precision (hits)", f"{backtest['precision']:.1%}")
            bc3.metric("Random Baseline", f"{backtest['baseline_hit_rate']:.1%}")

            if backtest["precision"] <= backtest["baseline_hit_rate"] * 1.1:
                st.warning(
                    "⚠️ The model's precision is close to or below the random baseline. "
                    "This is expected — lottery draws are designed to be unpredictable."
                )

            # --- Ticket suggestions ---
            st.markdown("---")
            st.markdown("#### 🎟️ Suggested Tickets")

            for t in range(num_tickets):
                if t == 0:
                    picks = sorted(top_5)
                    mega = pred_mega
                else:
                    # Weighted random sample from probability distribution
                    probs = prob_df["Probability"].values
                    probs = probs / probs.sum()
                    picks = sorted(
                        np.random.choice(
                            prob_df["Number"].values, size=5, replace=False, p=probs
                        ).tolist()
                    )
                    mega = np.random.randint(1, conf["max_mega"] + 1)

                st.success(f"**Ticket {t+1}:** {picks}  |  Mega: **{mega}**")

            # --- Probability table ---
            st.markdown("#### Top 15 Numbers by Predicted Probability")
            display_df = prob_df.head(15)[["Number", "Gap", "Freq_Last_10", "Freq_Last_3", "Probability"]]
            st.dataframe(
                display_df.style.format({"Probability": "{:.4f}"}).background_gradient(
                    subset=["Probability"], cmap="YlOrRd"
                ),
                use_container_width=True,
            )

            # Feature importance
            st.markdown("#### Feature Importance")
            st.caption("What the model weighs most when scoring each number.")


# ============================================================
# 5. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Install HTTP cache so repeated scraper runs are faster
    requests_cache.install_cache("lottery_http_cache", expire_after=3600)
    render_main()
