import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import cloudscraper

st.set_page_config(page_title="Lottery EV Optimizer", layout="wide")

# --- 1. DATA EXTRACTION (FREE CLOUDSCRAPER) ---

def fetch_free_lottery_data(game_id=8, pages=10):
    """Fetches CA Lottery data for FREE using CloudScraper to bypass Cloudflare."""
    base_url = "https://www.calottery.com/api/DrawGameApi/DrawGamePastDrawResults/{game_id}/{page}/20"
    all_draws = []
    
    # Initialize the free Cloudflare bypasser
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for page in range(1, pages + 1):
        status_text.text(f"Scraping page {page} of {pages} (using free CloudScraper)...")
        target_url = base_url.format(game_id=game_id, page=page)
        
        try:
            # We use scraper.get() instead of requests.get()
            response = scraper.get(target_url, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                draws = data.get('PreviousDraws', [])
                
                if not draws: 
                    st.info(f"Reached the end of available data at page {page}.")
                    break
                
                for draw in draws:
                    raw_nums = draw.get('WinningNumbers', [])
                    if raw_nums:
                        if isinstance(raw_nums[0], dict):
                            nums = [int(n.get('Number', 0)) for n in raw_nums]
                        else:
                            nums = [int(n) for n in raw_nums]
                        
                        all_draws.append({
                            'DrawNumber': draw.get('DrawNumber'),
                            'Date': draw.get('DrawDate'),
                            'Numbers': nums[:-1],
                            'MegaBall': nums[-1]
                        })
            else:
                st.warning(f"Failed on page {page}. Status code: {response.status_code}")
                break
                
        except Exception as e:
            st.error(f"CloudScraper connection failed on page {page}: {e}")
            break
            
        progress_bar.progress(page / pages)
        
    status_text.text("Free scraping complete!")
    return pd.DataFrame(all_draws)

# --- 2. DATA MANAGEMENT ---

@st.cache_data
def load_data():
    """Tries to load scraped CSV. Falls back to static manual data if unavailable."""
    if os.path.exists("historical_draws.csv"):
        # Load the CSV. We need to evaluate the 'Numbers' string back into a list.
        df = pd.read_csv("historical_draws.csv")
        df['Numbers'] = df['Numbers'].apply(ast.literal_eval)
        return df, "Loaded from historical_draws.csv"
    
    # Fallback Data
    manual_data = [
        [1,19,24, 38, 46, 3], [8, 9, 22, 33, 44, 24], [5, 13, 14, 26, 42, 20], 
        [3, 5, 7, 8, 46, 17], [4, 19, 22, 33, 43, 12], [2, 15, 17, 22, 38, 18],
        [6, 24, 29, 32, 41, 13], [3, 6, 14, 26, 38, 2], [6, 14, 15, 33, 37, 14],
        [10, 13, 18, 20, 40, 15], [4, 18, 20, 23, 39, 3], [4, 13, 22, 29, 39, 26],
        [1, 4, 14, 17, 32, 17], [6, 16, 17, 27, 42, 13], [5, 35, 37, 42, 46, 24],
        [5, 10, 13, 22, 33, 25], [6, 9, 33, 37, 40, 22], [9, 14, 21, 42, 43, 24],
        [1, 13, 22, 31, 37, 18], [1, 3, 4, 17, 40, 9], [7, 16, 17, 33, 35, 21],
        [10, 19, 30, 33, 42, 14], [6, 10, 30, 42, 43, 19], [4, 11, 19, 23, 24, 27]
    ]
    all_draws = [{'Numbers': draw[:5], 'MegaBall': draw[5]} for draw in manual_data]
    return pd.DataFrame(all_draws), "Loaded from Manual Fallback Data"


# --- 3. BAYESIAN & GAME THEORY OPERATIONS ---

@st.cache_resource
def run_mcmc_simulation(df_lotto):
    """Runs the PyMC MCMC simulation to find mechanical biases."""
    all_winning_nums = [n for sublist in df_lotto['Numbers'] for n in sublist]
    observed_counts = np.bincount(all_winning_nums, minlength=48)[1:]
    
    with pm.Model() as lottery_model:
        latent_theta = pm.Normal("latent_theta", mu=0, sigma=1, shape=47)
        theta = pm.Deterministic("theta", pm.math.softmax(latent_theta))
        pm.Multinomial("results", n=np.sum(observed_counts), p=theta, observed=observed_counts)
        trace = pm.sample(draws=1000, tune=1000, target_accept=0.90, chains=2, random_seed=42, progressbar=False)
        
    mega_counts = np.bincount(df_lotto['MegaBall'], minlength=28)[1:]
    with pm.Model() as mega_model:
        latent_mega = pm.Normal("latent_mega", mu=0, sigma=1, shape=27)
        theta_mega = pm.Deterministic("theta_mega", pm.math.softmax(latent_mega))
        pm.Multinomial("res", n=len(df_lotto), p=theta_mega, observed=mega_counts)
        mega_trace = pm.sample(1000, tune=1000, target_accept=0.90, chains=2, progressbar=False)
        
    return trace, mega_trace

def categorize_numbers(trace, n_numbers=47):
    fair_prob = 1 / n_numbers
    summary = az.summary(trace, var_names=["theta"], hdi_prob=0.95)
    summary['deviation'] = summary['mean'] - fair_prob
    summary['snr'] = summary['deviation'] / summary['sd']
    summary['lotto_num'] = np.arange(1, n_numbers + 1)
    
    hot = summary[summary['snr'] > 0.8]['lotto_num'].tolist()
    cold = summary[summary['snr'] < -1.0]['lotto_num'].tolist()
    neutral = summary[(summary['snr'] <= 0.8) & (summary['snr'] >= -1.0)]['lotto_num'].tolist()
    
    return {'hot': hot, 'cold': cold, 'neutral': neutral}, summary

def get_crowd_avoidance_scores(n_numbers=47, n_agents=10000):
    weights = np.ones(n_numbers + 1)
    weights[1:32] *= 5.0  # Birthday Bias
    for n in [7, 11, 3, 8]: weights[n] *= 2.0 # Lucky numbers
        
    probs = weights[1:] / np.sum(weights[1:])
    crowd_picks = np.random.choice(np.arange(1, n_numbers + 1), size=n_agents * 5, p=probs)
    
    counts = dict(zip(*np.unique(crowd_picks, return_counts=True)))
    avoidance_scores = {n: 1.0 / counts.get(n, 1) for n in range(1, n_numbers + 1)}
    
    avg_score = np.mean(list(avoidance_scores.values()))
    return {n: score / avg_score for n, score in avoidance_scores.items()}

def generate_nash_equilibrium_tickets(mcmc_stats, crowd_scores, top_megas, num_tickets=10):
    final_tickets = []
    weights = {n: (2.0 if n in mcmc_stats['hot'] else 1.5 if n in mcmc_stats['cold'] else 1.0) * (crowd_scores[n] ** 0.5) for n in range(1, 48)}
    
    population = list(weights.keys())
    w_list = list(weights.values())
    
    while len(final_tickets) < num_tickets:
        ticket = sorted(random.choices(population, weights=w_list, k=5))
        if len(set(ticket)) != 5: continue
            
        if 106 <= sum(ticket) <= 135 and len([n for n in ticket if n % 2 != 0]) in [2, 3]:
            if ticket not in [t[0] for t in final_tickets]:
                final_tickets.append((ticket, random.choice(top_megas)))
    return final_tickets

def calculate_multi_mega_ev(tickets, mega_picks, main_trace, mega_trace, crowd_scores, nominal_jackpot=25_000_000):
    """
    Advanced EV Calculator where each ticket can have its own unique Mega Ball.
    Runs 2000 simulations per ticket to calculate the expected return.
    """
    theta_main = main_trace.posterior['theta'].values.reshape(-1, 47)
    theta_mega = mega_trace.posterior['theta_mega'].values.reshape(-1, 27)
    
    # Calculate mean probabilities for fast simulation
    mean_main = theta_main.mean(axis=0)
    mean_mega = theta_mega.mean(axis=0)
    
    results = []
    
    for t_idx, ticket in enumerate(tickets):
        current_mega = mega_picks[t_idx]
        
        # 1. PARIMUTUEL ADJUSTMENT
        sharing_risk = np.mean([1.0 / crowd_scores.get(n, 1.0) for n in ticket + [current_mega]])
        estimated_winners = max(1.0, sharing_risk * 1.8) 
        adj_jackpot = nominal_jackpot / estimated_winners
        
        # 2. FAST SIMULATION
        winnings = 0
        ticket_indices = [n-1 for n in ticket]
        
        sim_draws = np.random.choice(47, size=(2000, 5), p=mean_main)
        sim_megas = np.random.choice(27, size=2000, p=mean_mega)
        
        for i in range(2000):
            matches = len(set(ticket_indices).intersection(sim_draws[i]))
            mega_hit = (sim_megas[i] == (current_mega - 1))
            
            # Tally parimutuel prizes based on CA SuperLotto structure
            if matches == 5 and mega_hit: winnings += adj_jackpot
            elif matches == 5: winnings += 24000
            elif matches == 4 and mega_hit: winnings += 1400
            elif matches == 4: winnings += 117
            elif matches == 3 and mega_hit: winnings += 55
            elif matches in [2, 3]: winnings += 10
            elif mega_hit: winnings += 1
                
        results.append({
            'Ticket': str(ticket),
            'Mega': current_mega,
            'Est. Winners': round(estimated_winners, 2),
            'Adj. Jackpot': f"${adj_jackpot/1e6:.1f}M",
            'EV': round((winnings / 2000) - 1.00, 4)
        })
        
    return pd.DataFrame(results).sort_values(by='EV', ascending=False)


# --- 4. STREAMLIT UI ---

st.title("🎯 Operations Research: Lottery EV Optimizer")
st.markdown("Applying Bayesian MCMC and Game Theory (Nash Equilibrium) to optimize lottery number selection.")

# Sidebar - Data Fetching
st.sidebar.header("📡 Data Controls")

if st.sidebar.button("Fetch Latest Data (FREE)"):
    with st.spinner("Bypassing Cloudflare..."):
        scraped_df = fetch_free_lottery_data(pages=5)
        if not scraped_df.empty:
            scraped_df.to_csv("historical_draws.csv", index=False)
            st.sidebar.success(f"Successfully scraped and saved {len(scraped_df)} draws!")
            load_data.clear()
        else:
            st.sidebar.error("Scraping failed or returned empty data.")

# Load the data into the app
df_lotto, data_source_msg = load_data()
st.sidebar.info(f"Source: {data_source_msg}\nTotal Draws: {len(df_lotto)}")

# Main Interface
if st.button("Run Advanced EV Simulation"):
    with st.spinner("Running PyMC MCMC Sampling (This may take a minute)..."):
        trace, mega_trace = run_mcmc_simulation(df_lotto)
        mcmc_stats, summary = categorize_numbers(trace)
        crowd_scores = get_crowd_avoidance_scores()
        
        mega_summary = az.summary(mega_trace, var_names=["theta_mega"])
        mega_summary['snr'] = (mega_summary['mean'] - (1/27)) / mega_summary['sd']
        mega_weights = {i: max(0.1, 1.0 + mega_summary.iloc[i-1]['snr']) * crowd_scores.get(i, 1.0) for i in range(1, 28)}
        top_megas = sorted(mega_weights, key=mega_weights.get, reverse=True)[:3]

    st.subheader("📊 MCMC Heatmap: Statistical Imbalances")
    fig, ax = plt.subplots(figsize=(10, 5))
    expected_prob = 1 / 47
    deviations = ((summary['mean'].values - expected_prob) / expected_prob) * 100
    heatmap_data = np.full(48, np.nan)
    heatmap_data[:47] = deviations
    sns.heatmap(heatmap_data.reshape((6, 8)), annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.write("🔥 **Hot Numbers (Signal > Noise):**", mcmc_stats['hot'])
    with col2:
        st.write("❄️ **Cold Numbers (Mean Reversion):**", mcmc_stats['cold'])

    st.subheader("🧠 Nash Equilibrium Optimized Tickets (10 Generated)")
    st.markdown("These tickets balance **mechanical bias** (from MCMC) with **crowd avoidance** (from Game Theory).")
    
    # Generate 10 tickets
    nash_tickets = generate_nash_equilibrium_tickets(mcmc_stats, crowd_scores, top_megas, num_tickets=10)
    
    # Display tickets as a formatted block
    for i, (ticket, mega) in enumerate(nash_tickets, 1):
        st.success(f"**Ticket {i}:** {ticket}  |  **Mega Ball:** {mega}  |  *(Sum: {sum(ticket)})*")
        
    st.subheader("📈 Expected Value (EV) Analysis")
    st.markdown("Running 2,000 simulations per ticket to calculate Expected Value (Return on $1 ticket). Higher EV = mathematically smarter play.")
    
    # Extract the tickets and megas into separate lists for the EV function
    just_tickets = [t[0] for t in nash_tickets]
    just_megas = [t[1] for t in nash_tickets]
    
    with st.spinner("Simulating tickets..."):
        df_ev = calculate_multi_mega_ev(
            tickets=just_tickets,
            mega_picks=just_megas,
            main_trace=trace,
            mega_trace=mega_trace,
            crowd_scores=crowd_scores
        )
        
        # Display the EV table
        st.dataframe(df_ev, use_container_width=True)
else:
    st.info("👈 Click 'Fetch Latest Data (FREE)' in the sidebar to scrape the CA Lottery, or click 'Run Advanced EV Simulation' to evaluate the current dataset.")