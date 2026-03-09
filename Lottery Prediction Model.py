import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chisquare, poisson, geom
import statsmodels.stats.proportion as smp
import xgboost as xgb

# ==========================================
# 1. XGBOOST AI MODEL (Engine)
# ==========================================
@st.cache_resource(show_spinner=False)
def train_xgboost_predictor(df_lotto, max_main=47, max_mega=27):
    """
    Engineers time-series features and trains an XGBoost model 
    to predict the probability of each number appearing in the NEXT draw.
    """
    draw_matrix = np.zeros((len(df_lotto), max_main), dtype=int)
    for i, nums in enumerate(df_lotto['Numbers']):
        for n in nums:
            if 1 <= n <= max_main:
                draw_matrix[i, n-1] = 1
                
    df_binary = pd.DataFrame(draw_matrix, columns=[f"Num_{i}" for i in range(1, max_main + 1)])
    
    dataset_rows = []
    last_seen = {n: -100 for n in range(1, max_main + 1)} 
    
    for draw_idx in range(10, len(df_lotto) - 1): 
        recent_10 = df_binary.iloc[draw_idx-10:draw_idx].sum()
        recent_3 = df_binary.iloc[draw_idx-3:draw_idx].sum()
        next_draw_results = df_binary.iloc[draw_idx + 1]
        
        for n in range(1, max_main + 1):
            col_name = f"Num_{n}"
            if df_binary.iloc[draw_idx][col_name] == 1:
                last_seen[n] = draw_idx
                
            gap = draw_idx - last_seen[n]
            
            dataset_rows.append({
                'Number': n,
                'Gap': gap,
                'Freq_Last_10': recent_10[col_name],
                'Freq_Last_3': recent_3[col_name],
                'Target': next_draw_results[col_name]
            })

    ml_df = pd.DataFrame(dataset_rows)
    X = ml_df[['Number', 'Gap', 'Freq_Last_10', 'Freq_Last_3']]
    y = ml_df['Target']
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X, y)
    
    latest_idx = len(df_lotto) - 1
    recent_10_latest = df_binary.iloc[latest_idx-9:latest_idx+1].sum()
    recent_3_latest = df_binary.iloc[latest_idx-2:latest_idx+1].sum()
    
    prediction_rows = []
    for n in range(1, max_main + 1):
        if df_binary.iloc[latest_idx][f"Num_{n}"] == 1:
            current_gap = 0
        else:
            current_gap = latest_idx - last_seen[n]
            
        prediction_rows.append({
            'Number': n,
            'Gap': current_gap,
            'Freq_Last_10': recent_10_latest[f"Num_{n}"],
            'Freq_Last_3': recent_3_latest[f"Num_{n}"]
        })
        
    df_pred = pd.DataFrame(prediction_rows)
    probabilities = xgb_model.predict_proba(df_pred[['Number', 'Gap', 'Freq_Last_10', 'Freq_Last_3']])[:, 1]
    df_pred['Probability'] = probabilities
    
    df_pred = df_pred.sort_values(by='Probability', ascending=False)
    top_5_main = df_pred['Number'].head(5).tolist()
    
    mega_counts = df_lotto['MegaBall'].value_counts()
    last_mega = df_lotto['MegaBall'].iloc[-1]
    fallback_mega = int(mega_counts.index[-1]) if last_mega != mega_counts.index[-1] else int(mega_counts.index[-2])
    
    return top_5_main, fallback_mega, df_pred

# ==========================================
# 2. MAIN APP & STATISTICAL AUDITOR
# ==========================================
def run_statistical_audits():
    st.header("📊 Lottery Statistical Auditor & AI Predictor")
    st.markdown("Using data only from the **last machine/matrix replacement date** to ensure mathematical validity.")

    configs = {
        "Mega Millions": {
            "file": "MEGA_Millions.csv", "date_col": "Draw Date", "num_cols": ['Num1','Num2','Num3','Num4','Num5'], "mega_col": "Mega Ball",
            "start_date": "2025-04-08", "max_main": 70, "max_mega": 24
        },
        "Powerball": {
            "file": "POWERBALL.csv", "date_col": "Draw Date", "num_cols": ['Num1','Num2','Num3','Num4','Num5'], "mega_col": "Powerball",
            "start_date": "2015-10-07", "max_main": 69, "max_mega": 26
        },
        "SuperLotto Plus": {
            "file": "SuperLotto_Plus.csv", "date_col": "Draw Date", "num_cols": ['Num1','Num2','Num3','Num4','Num5'], "mega_col": "Mega",
            "start_date": "2000-06-03", "max_main": 47, "max_mega": 27
        }
    }

    lottery_choice = st.selectbox("Select Lottery to Audit:", list(configs.keys()))
    conf = configs[lottery_choice]
    
    try:
        df = pd.read_csv(conf["file"])
        if df[conf['date_col']].dtype == object:
            df[conf['date_col']] = df[conf['date_col']].str.replace(r'^[A-Za-z]+ ', '', regex=True)
        
        df[conf['date_col']] = pd.to_datetime(df[conf['date_col']], errors='coerce')
        df = df.dropna(subset=[conf['date_col']])
        df = df[df[conf['date_col']] >= pd.to_datetime(conf['start_date'])].copy()
        df = df.sort_values(conf['date_col']).reset_index(drop=True)
    except Exception as
