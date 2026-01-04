import streamlit as st
import pandas as pd
import math
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Systematic Strategy Backtester",
    layout="wide"
)

# ==========================================
# 2. STRATEGY LOGIC
# ==========================================

def clean_numeric(val):
    if isinstance(val, str):
        return float(val.replace(',', '').strip())
    return float(val)

def get_ceiled_gap(price, percentage):
    gap_val = price * (percentage / 100) 
    return math.ceil(gap_val / 100) * 100

@st.cache_data
def load_data(uploaded_files):
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    all_dfs = []
    
    for uploaded_file in uploaded_files:
        try:
            filename = uploaded_file.name.lower()
            temp_dfs = []

            # --- EXCEL ---
            if filename.endswith('.xlsx'):
                xls_data = pd.read_excel(uploaded_file, engine='openpyxl', sheet_name=None)
                temp_dfs.extend(xls_data.values()) 
            elif filename.endswith('.xls'):
                try:
                    xls_data = pd.read_excel(uploaded_file, engine='xlrd', sheet_name=None)
                    temp_dfs.extend(xls_data.values())
                except Exception:
                    uploaded_file.seek(0)
                    tables = pd.read_html(uploaded_file)
                    temp_dfs.extend(tables)
            # --- CSV ---
            elif filename.endswith('.csv'):
                temp_dfs.append(pd.read_csv(uploaded_file))

            # --- CLEAN ---
            for df in temp_dfs:
                df.columns = df.columns.astype(str).str.title().str.strip()
                all_dfs.append(df)

        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {e}")
            continue

    if not all_dfs:
        return None

    try:
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Numeric Conversion
        cols_to_clean = ['Open', 'High', 'Low', 'Close']
        for col in cols_to_clean:
            if col in full_df.columns and full_df[col].dtype == 'object':
                full_df[col] = full_df[col].apply(clean_numeric)
        
        # --- ROBUST DATE PARSING ---
        if 'Date' in full_df.columns:
            full_df['Date'] = full_df['Date'].astype(str).str.strip()
            
            # Priority 1: ISO Format (2025-10-03)
            iso_dates = pd.to_datetime(full_df['Date'], format='%Y-%m-%d', errors='coerce')
            
            # Priority 2: Standard Format (30 Apr 2021)
            std_dates = pd.to_datetime(full_df['Date'], format='%d %b %Y', errors='coerce')
            
            # Priority 3: Standard Fallback
            fallback_dates = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')
            
            full_df['Date'] = iso_dates.fillna(std_dates).fillna(fallback_dates)
            full_df = full_df.dropna(subset=['Date'])
        else:
            return None

        # --- ROBUST EXPIRY PARSING ---
        expiry_col = [c for c in full_df.columns if 'Expiry' in c]
        if expiry_col:
            col_name = expiry_col[0]
            full_df[col_name] = full_df[col_name].astype(str).str.strip()
            
            iso_exp = pd.to_datetime(full_df[col_name], format='%Y-%m-%d', errors='coerce')
            std_exp = pd.to_datetime(full_df[col_name], format='%d%b%Y', errors='coerce')
            fallback_exp = pd.to_datetime(full_df[col_name], dayfirst=True, errors='coerce')
            
            full_df['Expiry Date'] = iso_exp.fillna(std_exp).fillna(fallback_exp)
            full_df = full_df.dropna(subset=['Expiry Date'])
        else:
            return None

        # Sort by Date
        full_df = full_df.sort_values('Date', ascending=True).reset_index(drop=True)
        return full_df

    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None

def run_simulation(df, start_date, end_date, lots, gaps, single_cycle_mode=False):
    # Filter Date Range
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    # Group by Date to handle multiple expiries per day
    daily_groups = {k: v for k, v in df[mask].groupby('Date')}
    unique_dates = sorted(daily_groups.keys())
    
    grand_ledger = []
    cycle_summaries = []
    total_profit = 0
    cycle_count = 0
    next_entry_price = None
    max_legs = len(lots)
    
    # 10x Margin/Multiplier
    MULTIPLIER = 10 
    
    position_open = False
    active_expiry = None
    current_leg = 0
    total_lots = 0
    avg_price = 0
    last_buy_day_close = 0
    cycle_ledger = []
    
    progress_bar = st.progress(0)
    total_days = len(unique_dates)
    
    for day_idx, current_date in enumerate(unique_dates):
        if day_idx % 10 == 0: progress_bar.progress(day_idx / total_days)
        todays_contracts = daily_groups[current_date]
        
        # ==========================================
        # 1. NEW CYCLE ENTRY (Rule: <=7th -> +2M, >7th -> +3M)
        # ==========================================
        if not position_open:
            offset_months = 2 if current_date.day <= 7 else 3
            target_date = current_date + relativedelta(months=offset_months)
            
            # Find contract matching Target Month/Year
            matching_contract = todays_contracts[
                (todays_contracts['Expiry Date'].dt.month == target_date.month) &
                (todays_contracts['Expiry Date'].dt.year == target_date.year)
            ]
            
            if matching_contract.empty:
                continue
                
            row = matching_contract.iloc[0]
            active_expiry = row['Expiry Date'] # LOCK EXPIRY
            
            current_leg = 0
            qty = lots[current_leg]
            
            if next_entry_price and not single_cycle_mode:
                buy_price = next_entry_price
                note = "Cycle Restart"
            else:
                buy_price = row['High']
                note = "Start High"
            
            total_lots = qty
            avg_price = buy_price
            last_buy_day_close = row['Close']
            position_open = True
            
            cycle_ledger = [{
                'Date': current_date, 'Action': 'BUY', 'Leg': 'Leg 1', 'Qty': qty, 
                'Price': buy_price, 'AvgPrice': avg_price, 'Status': note,
                'Profit': 0, 'Expiry Used': active_expiry.date()
            }]
            continue

        # ==========================================
        # 2. MANAGE POSITION (Locked Expiry Only)
        # ==========================================
        else:
            # Filter for the SPECIFIC locked expiry
            row_data = todays_contracts[todays_contracts['Expiry Date'] == active_expiry]
            
            if row_data.empty:
                # Force close if contract expired/missing and date passed
                if current_date > active_expiry:
                     grand_ledger.extend(cycle_ledger)
                     cycle_count += 1
                     cycle_summaries.append({
                        'Cycle': cycle_count,
                        'Start Date': cycle_ledger[0]['Date'].date(),
                        'End Date': current_date.date(),
                        'Outcome': "Data Gap/Expired",
                        'Profit': 0,
                        'Expiry': active_expiry.date(),
                        'Cumulative PnL': total_profit 
                     })
                     position_open = False
                     active_expiry = None
                continue
                
            row = row_data.iloc[0]
            high, low, close, open_p = row['High'], row['Low'], row['Close'], row['Open']
            cycle_res = None

            target = avg_price * 1.01
            
            # CHECK EXIT: Target
            if high >= target:
                pnl = (target - avg_price) * total_lots * MULTIPLIER
                cycle_ledger.append({
                    'Date': current_date, 'Action': 'SELL', 'Leg': 'Target', 'Qty': total_lots, 
                    'Price': target, 'AvgPrice': avg_price, 'Status': 'Profit Exit',
                    'Profit': pnl, 'Expiry Used': active_expiry.date()
                })
                cycle_res = {'reason': 'Target Hit', 'pnl': pnl, 'exit_price': target}

            # CHECK EXIT: Expiry
            elif current_date >= active_expiry:
                exit_p = avg_price if high >= avg_price else close
                status = "Expiry (NPNL)" if high >= avg_price else "Expiry (Loss)"
                pnl = (exit_p - avg_price) * total_lots * MULTIPLIER
                
                cycle_ledger.append({
                    'Date': current_date, 'Action': 'SELL', 'Leg': 'Expiry', 'Qty': total_lots, 
                    'Price': exit_p, 'AvgPrice': avg_price, 'Status': status,
                    'Profit': pnl, 'Expiry Used': active_expiry.date()
                })
                cycle_res = {'reason': status, 'pnl': pnl, 'exit_price': exit_p}

            # CHECK ENTRY: Next Leg
            elif current_leg < (max_legs - 1):
                next_leg = current_leg + 1
                gap_pct = gaps[next_leg]
                gap_avg = get_ceiled_gap(avg_price, gap_pct)
                gap_close = get_ceiled_gap(last_buy_day_close, gap_pct)
                trigger = min(avg_price - gap_avg, last_buy_day_close - gap_close)
                
                if low <= trigger:
                    buy_price = open_p if open_p < trigger else trigger
                    qty = lots[next_leg]
                    
                    total_cost = (avg_price * total_lots) + (qty * buy_price)
                    total_lots += qty
                    avg_price = total_cost / total_lots
                    
                    last_buy_day_close = close
                    current_leg += 1
                    
                    cycle_ledger.append({
                        'Date': current_date, 'Action': 'BUY', 'Leg': f'Leg {current_leg+1}', 'Qty': qty, 
                        'Price': buy_price, 'AvgPrice': avg_price, 'Status': "Limit/Gap",
                        'Profit': 0, 'Expiry Used': active_expiry.date()
                    })

            # FINALIZE CYCLE
            if cycle_res:
                grand_ledger.extend(cycle_ledger)
                cycle_count += 1
                total_profit += cycle_res['pnl']
                cycle_summaries.append({
                    'Cycle': cycle_count,
                    'Start Date': cycle_ledger[0]['Date'].date(),
                    'End Date': cycle_ledger[-1]['Date'].date(),
                    'Outcome': cycle_res['reason'],
                    'Profit': cycle_res['pnl'],
                    'Expiry': active_expiry.date()
                })
                
                if single_cycle_mode:
                    break
                
                position_open = False
                active_expiry = None
                next_entry_price = cycle_res['exit_price'] + 5
    
    progress_bar.empty()
    
    # --- CONVERT TO DATAFRAMES & CLEAN DATES ---
    ledger_df = pd.DataFrame(grand_ledger)
    summary_df = pd.DataFrame(cycle_summaries)
    
    if not ledger_df.empty:
        # Remove time component from Ledger Dates
        ledger_df['Date'] = pd.to_datetime(ledger_df['Date']).dt.date
    
    return ledger_df, summary_df, total_profit

# ==========================================
# 3. FRONTEND UI LAYOUT
# ==========================================

with st.sidebar:
    st.header("Configuration")
    
    num_legs = st.number_input("Number of Legs", min_value=1, max_value=20, value=5)
    
    lots = []
    gaps = []
    
    st.subheader("Leg Settings")
    for i in range(num_legs):
        c1, c2 = st.columns(2)
        with c1:
            def_lot = 6
            if i == 1: def_lot = 4
            l = st.number_input(f"Leg {i+1} Lots", value=def_lot, min_value=1, key=f"lot_{i}")
            lots.append(l)
        with c2:
            if i == 0:
                st.caption("Gap: 0% (Start)")
                gaps.append(0.0)
            else:
                def_gap = 1.0 + (0.5 * (i-1))
                g = st.number_input(f"Gap Leg {i+1} (%)", value=def_gap, step=0.1, min_value=0.0, key=f"gap_{i}")
                gaps.append(g)

st.title("Systematic Strategy Backtester")
st.write("Upload your Commodity Data (CSV, Excel) to begin.")

uploaded_files = st.file_uploader("Upload Data File(s)", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)

if uploaded_files:
    df = load_data(uploaded_files)
    
    if df is not None:
        st.success(f"Loaded {len(df)} rows. Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        st.divider()
        st.subheader("Simulation Settings")
        
        mode = st.radio("Select Mode", ["Single Cycle", "Continuous Backtest"])
        
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        
        end_date = max_date 
        if mode == "Continuous Backtest":
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        if st.button("Run Simulation", type="primary"):
            is_single = (mode == "Single Cycle")
            
            ledger_df, summary_df, total_pnl = run_simulation(df, start_date, end_date, lots, gaps, single_cycle_mode=is_single)
            
            if not summary_df.empty:
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
                
                # --- VISUALIZATION ---
                summary_df['Cumulative PnL'] = summary_df['Profit'].cumsum()
                
                if not is_single:
                    st.subheader("1. Equity Curve")
                    fig_eq = go.Figure()
                    
                    fig_eq.add_trace(go.Scatter(
                        x=summary_df['End Date'], 
                        y=summary_df['Cumulative PnL'],
                        mode='lines',
                        name='Equity',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    
                    marker_colors = ['#00CC96' if val >= 0 else '#EF553B' for val in summary_df['Cumulative PnL']]
                    fig_eq.add_trace(go.Scatter(
                        x=summary_df['End Date'],
                        y=summary_df['Cumulative PnL'],
                        mode='markers',
                        name='Status',
                        marker=dict(size=10, color=marker_colors)
                    ))
                    
                    fig_eq.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_eq.update_layout(showlegend=False, xaxis_title="Date", yaxis_title="Cumulative PnL")
                    st.plotly_chart(fig_eq, use_container_width=True)

                    st.subheader("2. Profit/Loss per Cycle")
                    fig_bar = go.Figure()
                    bar_colors = ['#00CC96' if val >= 0 else '#EF553B' for val in summary_df['Profit']]
                    
                    fig_bar.add_trace(go.Bar(
                        x=summary_df['Cycle'],
                        y=summary_df['Profit'],
                        marker_color=bar_colors,
                        name="Cycle PnL"
                    ))
                    fig_bar.update_layout(xaxis_title="Cycle #", yaxis_title="Profit/Loss")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # --- DATA TABLES ---
                tab1, tab2 = st.tabs(["Cycle Summary", "Detailed Ledger"])
                
                with tab1:
                    st.dataframe(summary_df.style.format({
                        "Profit": "{:,.2f}", 
                        "Cumulative PnL": "{:,.2f}"
                    }), use_container_width=True)
                
                with tab2:
                    st.dataframe(ledger_df.style.format({
                        "Price": "{:,.2f}", 
                        "AvgPrice": "{:,.2f}", 
                        "Profit": "{:,.2f}"
                    }), use_container_width=True)
                    
                    csv = ledger_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Full Ledger CSV", data=csv, file_name="strategy_results.csv", mime='text/csv')
            
            else:
                st.warning("No cycles completed. This often happens if required expiry contracts are missing from the data.")