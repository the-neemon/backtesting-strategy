import streamlit as st
import pandas as pd
import math
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Jolly Gold 2 Strategy",
    layout="wide"
)

# ==========================================
# 2. STRATEGY LOGIC
# ==========================================

def clean_numeric(val):
    if isinstance(val, str):
        return float(val.replace(',', ''))
    return float(val)

def get_ceiled_gap(price, percentage):
    gap_val = price * (percentage / 100) 
    return math.ceil(gap_val / 100) * 100

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.title() 
        
        cols_to_clean = ['Open', 'High', 'Low', 'Close']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        expiry_col = [c for c in df.columns if 'Expiry' in c]
        if expiry_col:
            df['Expiry Date'] = pd.to_datetime(df[expiry_col[0]])
        else:
            st.error("No 'Expiry Date' column found in CSV!")
            return None

        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def run_simulation(df, start_date, end_date, lots, gaps, single_cycle_mode=False):
    
    # Find start index
    try:
        current_idx = df[df['Date'] >= pd.to_datetime(start_date)].index[0]
    except IndexError:
        return pd.DataFrame(), pd.DataFrame(), 0

    end_dt_obj = pd.to_datetime(end_date)
    
    grand_ledger = []
    cycle_summaries = []
    total_profit = 0
    cycle_count = 0
    next_entry_price = None
    
    # Progress Bar
    progress_bar = st.progress(0)
    
    while True:
        # Check Limits
        if current_idx >= len(df):
            break
        if not single_cycle_mode and df.iloc[current_idx]['Date'] > end_dt_obj:
            break
            
        # --- CYCLE LOGIC ---
        subset = df.iloc[current_idx:].reset_index(drop=True)
        
        position_open = False
        current_leg = 0
        total_lots = 0
        total_cost = 0
        avg_price = 0
        last_buy_day_close = 0
        cycle_ledger = []
        cycle_res = None
        
        for i, row in subset.iterrows():
            original_idx = current_idx + i
            date = row['Date']
            expiry = row['Expiry Date']
            high, low, close, open_p = row['High'], row['Low'], row['Close'], row['Open']

            # 1. ENTRY (Leg 1)
            if not position_open:
                if i == 0:
                    current_leg = 0
                    qty = lots[current_leg]
                    
                    # Logic: If continuous mode and not first cycle, use calculated price
                    if next_entry_price and not single_cycle_mode:
                        buy_price = next_entry_price
                        note = "Cycle Restart"
                    else:
                        buy_price = high
                        note = "Start High"
                    
                    total_lots = qty
                    total_cost = qty * buy_price
                    avg_price = buy_price
                    last_buy_day_close = close
                    position_open = True
                    
                    cycle_ledger.append({
                        'Date': date, 'Action': 'BUY', 'Leg': 'Leg 1', 'Qty': qty, 
                        'Price': buy_price, 'AvgPrice': avg_price, 'Status': note,
                        'Profit': 0, 'Cycle': cycle_count + 1
                    })
                continue
            
            # 2. EXIT (Target)
            target = avg_price * 1.01
            if high >= target:
                pnl = (target - avg_price) * total_lots
                cycle_ledger.append({
                    'Date': date, 'Action': 'SELL', 'Leg': 'Target', 'Qty': total_lots, 
                    'Price': target, 'AvgPrice': avg_price, 'Status': 'Profit Exit',
                    'Profit': pnl, 'Cycle': cycle_count + 1
                })
                cycle_res = {'end_idx': original_idx, 'pnl': pnl, 'reason': 'Target Hit', 'exit_price': target}
                break
                
            # 3. EXIT (Expiry)
            if date >= expiry:
                exit_p = avg_price if high >= avg_price else close
                status = "Expiry (NPNL)" if high >= avg_price else "Expiry (Loss)"
                pnl = (exit_p - avg_price) * total_lots
                
                cycle_ledger.append({
                    'Date': date, 'Action': 'SELL', 'Leg': 'Expiry', 'Qty': total_lots, 
                    'Price': exit_p, 'AvgPrice': avg_price, 'Status': status,
                    'Profit': pnl, 'Cycle': cycle_count + 1
                })
                cycle_res = {'end_idx': original_idx, 'pnl': pnl, 'reason': status, 'exit_price': exit_p}
                break
                
            # 4. NEXT LEGS
            if current_leg < 4:
                next_leg = current_leg + 1
                gap_pct = gaps[next_leg]
                gap_avg = get_ceiled_gap(avg_price, gap_pct)
                gap_close = get_ceiled_gap(last_buy_day_close, gap_pct)
                trigger = min(avg_price - gap_avg, last_buy_day_close - gap_close)
                
                if low <= trigger:
                    buy_price = open_p if open_p < trigger else trigger
                    qty = lots[next_leg]
                    total_cost += qty * buy_price
                    total_lots += qty
                    avg_price = total_cost / total_lots
                    last_buy_day_close = close
                    current_leg += 1
                    
                    cycle_ledger.append({
                        'Date': date, 'Action': 'BUY', 'Leg': f'Leg {current_leg+1}', 'Qty': qty, 
                        'Price': buy_price, 'AvgPrice': avg_price, 'Status': "Limit/Gap",
                        'Profit': 0, 'Cycle': cycle_count + 1
                    })

        # --- END OF CYCLE HANDLING ---
        grand_ledger.extend(cycle_ledger)
        
        if cycle_res:
            cycle_count += 1
            total_profit += cycle_res['pnl']
            cycle_summaries.append({
                'Cycle': cycle_count,
                'Start Date': cycle_ledger[0]['Date'].date(),
                'End Date': cycle_ledger[-1]['Date'].date(),
                'Outcome': cycle_res['reason'],
                'Profit': cycle_res['pnl']
            })
            
            # If Single Mode, STOP here
            if single_cycle_mode:
                break

            # Prepare next loop (Continuous Mode)
            current_idx = cycle_res['end_idx']
            next_entry_price = cycle_res['exit_price'] + 5
            
            # Skip forward if today is expiry to avoid loop
            if df.iloc[current_idx]['Date'] >= df.iloc[current_idx]['Expiry Date']:
                current_idx += 1
        else:
            # Data ran out
            break
            
    progress_bar.empty()
    return pd.DataFrame(grand_ledger), pd.DataFrame(cycle_summaries), total_profit

# ==========================================
# 3. FRONTEND UI LAYOUT
# ==========================================

# --- SIDEBAR: Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Position Sizing")
    c1, c2 = st.columns(2)
    l1 = c1.number_input("Leg 1 Lots", value=6)
    l2 = c2.number_input("Leg 2 Lots", value=4)
    l3 = c1.number_input("Leg 3 Lots", value=6)
    l4 = c2.number_input("Leg 4 Lots", value=6)
    l5 = c1.number_input("Leg 5 Lots", value=6)
    
    st.subheader("Gap Settings (%)")
    g2 = st.number_input("Gap Leg 2 (%)", value=1.0, step=0.1)
    g3 = st.number_input("Gap Leg 3 (%)", value=1.5, step=0.1)
    g4 = st.number_input("Gap Leg 4 (%)", value=2.0, step=0.1)
    g5 = st.number_input("Gap Leg 5 (%)", value=2.5, step=0.1)

    lots = [l1, l2, l3, l4, l5]
    gaps = [0, g2, g3, g4, g5]

# --- MAIN PAGE ---
st.title("Jolly Gold 2 Strategy")
st.write("Upload your Commodity Data (CSV) to begin backtesting.")

uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.divider()
        st.subheader("Simulation Settings")
        
        # Mode Selection
        mode = st.radio("Select Mode", ["Single Cycle", "Continuous Backtest"])
        
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        
        end_date = max_date # Default
        if mode == "Continuous Backtest":
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        if st.button("Run Simulation", type="primary"):
            
            # Determine flags
            is_single = (mode == "Single Cycle")
            
            # Run Logic
            ledger_df, summary_df, total_pnl = run_simulation(df, start_date, end_date, lots, gaps, single_cycle_mode=is_single)
            
            if not summary_df.empty:
                # --- DASHBOARD METRICS ---
                st.divider()
                st.subheader("Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Profit", f"{total_pnl:,.2f}")
                m2.metric("Total Cycles", len(summary_df))
                
                wins = summary_df[summary_df['Profit'] > 0]
                win_rate = (len(wins) / len(summary_df)) * 100
                m3.metric("Win Rate", f"{win_rate:.1f}%")
                
                avg_trade = total_pnl / len(summary_df)
                m4.metric("Avg Profit/Cycle", f"{avg_trade:,.2f}")
                
                # --- CHARTS (Only for Continuous) ---
                if not is_single:
                    st.subheader("Cumulative Profit Curve")
                    summary_df['Cumulative PnL'] = summary_df['Profit'].cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=summary_df['End Date'], 
                        y=summary_df['Cumulative PnL'],
                        mode='lines+markers',
                        name='Equity',
                        line=dict(color='#00CC96', width=3)
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- DATA TABS ---
                tab1, tab2 = st.tabs(["Cycle Summary", "Detailed Ledger"])
                
                with tab1:
                    st.dataframe(summary_df.style.format({"Profit": "{:,.2f}"}), use_container_width=True)
                
                with tab2:
                    st.dataframe(ledger_df.style.format({"Price": "{:,.2f}", "AvgPrice": "{:,.2f}", "Profit": "{:,.2f}"}), use_container_width=True)
                    
                    # Download Button
                    csv = ledger_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Full Ledger CSV", data=csv, file_name="jolly_gold_results.csv", mime='text/csv')
            
            else:
                st.warning("No cycles completed in the selected period.")