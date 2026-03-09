import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chisquare, poisson
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
    
    # ==========================================
    # Mega Ball Prediction (Strictly Bounded)
    # ==========================================
    # Force data to be numeric and strictly within the valid range for this lottery
    valid_megas = pd.to_numeric(df_lotto['MegaBall'], errors='coerce').dropna()
    valid_megas = valid_megas[(valid_megas >= 1) & (valid_megas <= max_mega)].astype(int)
    
    if not valid_megas.empty:
        mega_counts = valid_megas.value_counts()
        last_mega = valid_megas.iloc[-1]
        
        # Check if there are valid numbers that have NEVER been drawn yet
        drawn_set = set(mega_counts.index)
        all_set = set(range(1, max_mega + 1))
        never_drawn = list(all_set - drawn_set)
        
        if never_drawn:
            # If a valid number has never been drawn, it's the most "due". Pick it.
            fallback_mega = sorted(never_drawn)[0] 
        else:
            # Otherwise, pick the least frequent valid number
            fallback_mega = int(mega_counts.index[-1])
            # If the least frequent was literally just drawn last week, pick the second least frequent
            if fallback_mega == last_mega and len(mega_counts) > 1:
                fallback_mega = int(mega_counts.index[-2])
    else:
        fallback_mega = 1 # Absolute safety fallback
    
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
    except Exception as e:
        st.error(f"Error loading {conf['file']}. Make sure the CSV is in the same folder.")
        return

    total_draws = len(df)
    st.info(f"Analyzing **{total_draws} draws** since the machine was replaced on **{conf['start_date']}**.")

    main_df = df[conf['num_cols']].apply(pd.to_numeric, errors='coerce')
    main_nums = main_df.values.flatten()
    main_nums = main_nums[~np.isnan(main_nums)]
    mega_nums = pd.to_numeric(df[conf['mega_col']], errors='coerce').dropna().values

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Fairness Hypothesis Testing")
        st.markdown("*Null Hypothesis (H0): The machine draws balls uniformly.*")
        
        main_counts = pd.Series(main_nums).value_counts().reindex(range(1, conf['max_main'] + 1), fill_value=0)
        expected_main_val = sum(main_counts) / conf['max_main']
        chi2_main, p_main = chisquare(main_counts, f_exp=[expected_main_val] * conf['max_main'])
        
        mega_counts = pd.Series(mega_nums).value_counts().reindex(range(1, conf['max_mega'] + 1), fill_value=0)
        expected_mega_val = sum(mega_counts) / conf['max_mega']
        chi2_mega, p_mega = chisquare(mega_counts, f_exp=[expected_mega_val] * conf['max_mega'])

        st.metric("Main Numbers P-Value", f"{p_main:.4f}")
        st.metric("Mega Ball P-Value", f"{p_mega:.4f}")
        
        if p_main < 0.05 or p_mega < 0.05:
            st.warning("⚠️ **P-Value < 0.05:** The machine shows statistically significant bias!")
        else:
            st.success("✅ **P-Value > 0.05:** We cannot reject the null hypothesis. The machine is fair.")

    st.divider()
    st.subheader(f"Deep Dive: Analyze a Specific Number")
    target_num = st.number_input("Enter a Main Number to analyze:", min_value=1, max_value=conf['max_main'], value=7)
    
    draws_with_target = main_df.isin([target_num]).any(axis=1)
    k = draws_with_target.sum()
    expected_p = 5 / conf['max_main']

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Confidence Intervals")
        st.markdown(f"Number **{target_num}** was drawn **{k} times** out of {total_draws} draws.")
        ci_low, ci_high = smp.proportion_confint(k, total_draws, alpha=0.05, method='wilson')
        
        st.write(f"- **Observed Probability:** {k/total_draws:.4f}")
        st.write(f"- **Expected Probability:** {expected_p:.4f}")
        st.write(f"- **95% Confidence Interval:** [{ci_low:.4f}, {ci_high:.4f}]")
        
        if ci_low <= expected_p <= ci_high:
            st.success(f"Number {target_num}'s appearances are within expectations.")
        else:
            st.warning(f"Number {target_num} is appearing outside expected bounds!")

    with col4:
        st.markdown("#### Poisson & Geometric Wait Times")
        expected_wait = 1 / expected_p
        if k > 1:
            draw_indices = np.where(draws_with_target)[0]
            avg_actual_wait = np.mean(np.diff(draw_indices))
            st.write(f"- **Geometric Expected Wait Time:** {expected_wait:.1f} draws")
            st.write(f"- **Actual Average Wait Time:** {avg_actual_wait:.1f} draws")

        st.markdown("---")
        draws_in_year = 104
        lambda_poisson = draws_in_year * expected_p
        st.write(f"In a standard year ({draws_in_year} draws), we expect to see this number **{lambda_poisson:.1f} times**.")
        
        prob_exact = poisson.pmf(int(lambda_poisson), lambda_poisson) * 100
        prob_more = (1 - poisson.cdf(int(lambda_poisson), lambda_poisson)) * 100
        st.write(f"Odds of drawing it EXACTLY {int(lambda_poisson)} times: **{prob_exact:.2f}%**")
        st.write(f"Odds of drawing it MORE than {int(lambda_poisson)} times: **{prob_more:.2f}%**")

    # ==========================================
    # 3. XGBOOST UI COMPONENT
    # ==========================================
    st.divider()
    st.subheader("🤖 AI Prediction: XGBoost Machine Learning")
    st.markdown("This model treats lottery draws as a Time-Series Classification problem. It calculates wait times and rolling frequencies to predict the most probable upcoming numbers.")
    
    if st.button("Run XGBoost AI Predictor"):
        with st.spinner("Engineering features and training XGBoost model..."):
            
            # 1. Format the DataFrame so the AI can read the lists of numbers
            df['Numbers'] = df[conf['num_cols']].values.tolist()
            df['MegaBall'] = df[conf['mega_col']]
            
            # 2. Run the AI function
            top_5, predicted_mega, prob_df = train_xgboost_predictor(
                df_lotto=df, 
                max_main=conf['max_main'], 
                max_mega=conf['max_mega']
            )
            
            # 3. Show Results
            st.success(f"### 🏆 AI Recommended Ticket: {sorted(top_5)} | Mega Ball: {predicted_mega}")
            
            st.markdown("#### Top 10 Numbers by Machine Learning Probability")
            st.dataframe(prob_df.head(10).style.format({'Probability': '{:.4f}'}))
            
            st.info("💡 **What is the AI looking at?** The model determined that the **Gap (Wait Time)** and **Frequency in the last 10 draws** were the most critical features in determining if a number would be drawn.")

# ==========================================
# 4. EXECUTE APP
# ==========================================
run_statistical_audits()
