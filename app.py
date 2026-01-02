import streamlit as st
import pandas as pd
import math
import plotly.graph_objects as go
import io

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
        return float(val.replace(',', '').strip())
    return float(val)

def get_ceiled_gap(price, percentage):
    gap_val = price * (percentage / 100) 
    return math.ceil(gap_val / 100) * 100

@st.cache_data
def load_data(uploaded_file):
    try:
        filename = uploaded_file.name.lower()
        
        # --- 1. FILE LOADING ---
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif filename.endswith('.xls'):
            try:
                df = pd.read_excel(uploaded_file, engine='xlrd')
            except Exception:
                uploaded_file.seek(0)
                tables = pd.read_html(uploaded_file)
                if tables:
                    df = tables[0]
                else:
                    st.error("Could not find data in the HTML/XLS file.")
                    return None
        
        # --- 2. CLEANING COLUMNS ---
        df.columns = df.columns.str.title().str.strip()
        
        cols_to_clean = ['Open', 'High', 'Low', 'Close']
        for col in cols_to_clean:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(clean_numeric)
        
        # --- 3. ROBUST DATE PARSING ---
        if 'Date' in df.columns:
            df['Date'] = df['Date'].astype(str).str.strip()
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d %b %Y', errors='raise')
            except ValueError:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            df = df.dropna(subset=['Date'])
            if df.empty:
                st.error("All dates failed to parse.")
                return None
        else:
            st.error("Column 'Date' not found.")
            return None

        # --- 4. EXPIRY DATE PARSING ---
        expiry_col = [c for c in df.columns if 'Expiry' in c]
        if expiry_col:
            col_name = expiry_col[0]
            try:
                df['Expiry Date'] = pd.to_datetime(df[col_name], format='%d%b%Y', errors='raise')
            except ValueError:
                 df['Expiry Date'] = pd.to_datetime(df[col_name], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Expiry Date'])
        else:
            st.error("No 'Expiry Date' column found!")
            return None

        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def run_simulation(df, start_date, end_date, lots, gaps, single_cycle_mode=False):
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
    max_legs = len(lots) # Dynamic leg count
    
    progress_bar = st.progress(0)
    
    while True:
        if current_idx >= len(df):
            break
        if not single_cycle_mode and df.iloc[current_idx]['Date'] > end_dt_obj:
            break
            
        subset = df.iloc[current_idx:].reset_index(drop=True)
        
        position_open = False
        current_leg = 0
        total_lots = 0
        avg_price = 0
        last_buy_day_close = 0
        cycle_ledger = []
        cycle_res = None
        
        for i, row in subset.iterrows():
            original_idx = current_idx + i
            date = row['Date']
            expiry = row['Expiry Date']
            high, low, close, open_p = row['High'], row['Low'], row['Close'], row['Open']

            # ENTRY
            if not position_open:
                if i == 0:
                    current_leg = 0
                    qty = lots[current_leg]
                    
                    if next_entry_price and not single_cycle_mode:
                        buy_price = next_entry_price
                        note = "Cycle Restart"
                    else:
                        buy_price = high
                        note = "Start High"
                    
                    total_lots = qty
                    avg_price = buy_price
                    last_buy_day_close = close
                    position_open = True
                    
                    cycle_ledger.append({
                        'Date': date, 'Action': 'BUY', 'Leg': 'Leg 1', 'Qty': qty, 
                        'Price': buy_price, 'AvgPrice': avg_price, 'Status': note,
                        'Profit': 0
                    })
                continue
            
            # TARGET EXIT
            target = avg_price * 1.01
            if high >= target:
                pnl = (target - avg_price) * total_lots
                cycle_ledger.append({
                    'Date': date, 'Action': 'SELL', 'Leg': 'Target', 'Qty': total_lots, 
                    'Price': target, 'AvgPrice': avg_price, 'Status': 'Profit Exit',
                    'Profit': pnl
                })
                cycle_res = {'end_idx': original_idx, 'pnl': pnl, 'reason': 'Target Hit', 'exit_price': target}
                break
                
            # EXPIRY EXIT
            if date >= expiry:
                exit_p = avg_price if high >= avg_price else close
                status = "Expiry (NPNL)" if high >= avg_price else "Expiry (Loss)"
                pnl = (exit_p - avg_price) * total_lots
                
                cycle_ledger.append({
                    'Date': date, 'Action': 'SELL', 'Leg': 'Expiry', 'Qty': total_lots, 
                    'Price': exit_p, 'AvgPrice': avg_price, 'Status': status,
                    'Profit': pnl
                })
                cycle_res = {'end_idx': original_idx, 'pnl': pnl, 'reason': status, 'exit_price': exit_p}
                break
                
            # NEXT LEGS (Dynamic Check)
            if current_leg < (max_legs - 1):
                next_leg = current_leg + 1
                gap_pct = gaps[next_leg]
                gap_avg = get_ceiled_gap(avg_price, gap_pct)
                gap_close = get_ceiled_gap(last_buy_day_close, gap_pct)
                trigger = min(avg_price - gap_avg, last_buy_day_close - gap_close)
                
                if low <= trigger:
                    buy_price = open_p if open_p < trigger else trigger
                    qty = lots[next_leg]
                    
                    # Update Avg Price
                    total_cost = (avg_price * total_lots) + (qty * buy_price)
                    total_lots += qty
                    avg_price = total_cost / total_lots
                    
                    last_buy_day_close = close
                    current_leg += 1
                    
                    cycle_ledger.append({
                        'Date': date, 'Action': 'BUY', 'Leg': f'Leg {current_leg+1}', 'Qty': qty, 
                        'Price': buy_price, 'AvgPrice': avg_price, 'Status': "Limit/Gap",
                        'Profit': 0
                    })

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
            
            if single_cycle_mode:
                break

            current_idx = cycle_res['end_idx']
            next_entry_price = cycle_res['exit_price'] + 5
            
            if df.iloc[current_idx]['Date'] >= df.iloc[current_idx]['Expiry Date']:
                current_idx += 1
        else:
            break
            
    progress_bar.empty()
    return pd.DataFrame(grand_ledger), pd.DataFrame(cycle_summaries), total_profit

# ==========================================
# 3. FRONTEND UI LAYOUT
# ==========================================

with st.sidebar:
    st.header("Configuration")
    
    # --- 1. Dynamic Legs Selection ---
    num_legs = st.number_input("Number of Legs", min_value=1, max_value=20, value=5)
    
    lots = []
    gaps = []
    
    st.subheader("Leg Settings")
    
    # Generate Inputs Loop
    for i in range(num_legs):
        c1, c2 = st.columns(2)
        with c1:
            # Default value logic: 6 for leg 1, then alternating 4/6 roughly based on user's old pattern
            def_lot = 6
            if i == 1: def_lot = 4
            
            l = st.number_input(f"Leg {i+1} Lots", value=def_lot, min_value=1, key=f"lot_{i}")
            lots.append(l)
            
        with c2:
            if i == 0:
                st.caption("Gap: 0% (Start)")
                gaps.append(0.0)
            else:
                # Default gap logic: 0.5% increments approx
                def_gap = 1.0 + (0.5 * (i-1))
                g = st.number_input(f"Gap Leg {i+1} (%)", value=def_gap, step=0.1, min_value=0.0, key=f"gap_{i}")
                gaps.append(g)

st.title("Jolly Gold 2 Strategy")
st.write("Upload your Commodity Data (CSV, Excel) to begin.")

uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
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
                
                # --- METRICS ---
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Profit", f"{total_pnl:,.2f}")
                m2.metric("Total Cycles", len(summary_df))
                
                wins = summary_df[summary_df['Profit'] > 0]
                win_rate = (len(wins) / len(summary_df)) * 100
                m3.metric("Win Rate", f"{win_rate:.1f}%")
                
                avg_trade = total_pnl / len(summary_df)
                m4.metric("Avg Profit/Cycle", f"{avg_trade:,.2f}")
                
                # --- VISUALIZATION ---
                # Calculate Cumulative PnL for table and chart
                summary_df['Cumulative PnL'] = summary_df['Profit'].cumsum()
                
                if not is_single:
                    # CHART 1: EQUITY CURVE
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

                    # CHART 2: PROFIT PER CYCLE
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
                    # Format Cumulative PnL to 2 decimals
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
                    st.download_button("Download Full Ledger CSV", data=csv, file_name="jolly_gold_results.csv", mime='text/csv')
            
            else:
                st.warning("No cycles completed in the selected period.")