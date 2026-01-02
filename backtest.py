import pandas as pd
import math

# ==========================================
# 1. VISUAL SETTINGS & FORMATTING
# ==========================================

class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

def format_currency(val):
    """Formats float to string like 50,000.00"""
    return f"{float(val):,.2f}"

def print_header(title):
    print(f"\n{Color.BOLD}{'='*60}")
    print(f" {title}")
    print(f"{'='*60}{Color.END}")

# ==========================================
# 2. SETUP & PARAMETERS
# ==========================================

file_path = 'MCX_BhavCopyCommodityWise - MCX_BhavCopyCommodityWise.xls.csv'
LOT_SIZES = [6, 4, 6, 6, 6]  
GAPS = [0, 0.01, 0.015, 0.02, 0.025] 

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def clean_numeric(val):
    if isinstance(val, str):
        return float(val.replace(',', ''))
    return float(val)

def get_ceiled_gap(price, percentage):
    gap_val = price * percentage
    return math.ceil(gap_val / 100) * 100

def load_data(path):
    try:
        df = pd.read_csv(path)
        cols_to_clean = ['Open', 'High', 'Low', 'Close']
        for col in cols_to_clean:
            df[col] = df[col].apply(clean_numeric)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Expiry Date'] = pd.to_datetime(df['Expiry Date'])
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"{Color.RED}Error loading data: {e}{Color.END}")
        return None

# ==========================================
# 4. STRATEGY ENGINE
# ==========================================

def simulate_cycle(df, start_index, forced_entry_price=None):
    
    position_open = False
    current_leg = 0
    total_lots = 0
    total_cost = 0
    avg_price = 0
    last_buy_day_close = 0
    ledger = []
    
    subset = df.iloc[start_index:].reset_index(drop=True)
    if subset.empty: return None

    for i, row in subset.iterrows():
        original_idx = start_index + i
        date = row['Date']
        expiry_date = row['Expiry Date']
        high = row['High']
        low = row['Low']
        close = row['Close']
        open_p = row['Open']

        # --- START LEG 1 ---
        if not position_open:
            if i == 0: 
                current_leg = 0
                qty = LOT_SIZES[current_leg]
                
                if forced_entry_price is not None:
                    buy_price = forced_entry_price
                    note = "Cycle Restart"
                else:
                    buy_price = high
                    note = "Start High"
                
                total_lots = qty
                total_cost = qty * buy_price
                avg_price = buy_price
                last_buy_day_close = close
                position_open = True
                
                ledger.append({
                    'Date': date.date(), 'Action': 'BUY', 'Leg': f'Leg 1', 
                    'Qty': qty, 'Price': buy_price, 'AvgPrice': avg_price, 
                    'Status': note
                })
            continue

        # --- MANAGE POSITION ---
        target_exit = avg_price * 1.01
        
        # Check Target
        if high >= target_exit:
            exit_price = target_exit
            profit = (exit_price - avg_price) * total_lots
            ledger.append({
                'Date': date.date(), 'Action': 'SELL', 'Leg': 'Target', 
                'Qty': total_lots, 'Price': exit_price, 'AvgPrice': avg_price, 
                'Status': 'Profit Exit'
            })
            return {'status': 'Closed', 'reason': 'Target Hit', 'end_index': original_idx,
                    'end_date': date, 'exit_price': exit_price, 'profit': profit, 'ledger': ledger}

        # Check Expiry
        if date >= expiry_date:
            if high >= avg_price:
                exit_price = avg_price
                reason = 'Expiry (NPNL)'
            else:
                exit_price = close
                reason = 'Expiry (Loss)'
            
            profit = (exit_price - avg_price) * total_lots
            ledger.append({
                'Date': date.date(), 'Action': 'SELL', 'Leg': 'Expiry', 
                'Qty': total_lots, 'Price': exit_price, 'AvgPrice': avg_price, 
                'Status': reason
            })
            return {'status': 'Closed', 'reason': reason, 'end_index': original_idx,
                    'end_date': date, 'exit_price': exit_price, 'profit': profit, 'ledger': ledger}

        # Check Next Legs
        if current_leg < 4:
            next_leg = current_leg + 1
            gap_pct = GAPS[next_leg]
            
            gap_avg = get_ceiled_gap(avg_price, gap_pct)
            gap_close = get_ceiled_gap(last_buy_day_close, gap_pct)
            trigger = min(avg_price - gap_avg, last_buy_day_close - gap_close)
            
            if low <= trigger:
                if open_p < trigger:
                    buy_price = open_p
                    note = "Gap Down"
                else:
                    buy_price = trigger
                    note = "Limit Hit"
                    
                qty = LOT_SIZES[next_leg]
                total_cost += qty * buy_price
                total_lots += qty
                avg_price = total_cost / total_lots
                last_buy_day_close = close
                current_leg += 1
                
                ledger.append({
                    'Date': date.date(), 'Action': 'BUY', 'Leg': f'Leg {current_leg+1}', 
                    'Qty': qty, 'Price': buy_price, 'AvgPrice': avg_price, 
                    'Status': note
                })
    
    # End of Data
    last_val = subset.iloc[-1]
    unrealized = (last_val['Close'] - avg_price) * total_lots
    return {'status': 'Open', 'reason': 'Data End', 'end_index': start_index + len(subset) - 1,
            'end_date': last_val['Date'], 'exit_price': last_val['Close'], 'profit': unrealized, 'ledger': ledger}

# ==========================================
# 5. MAIN LOOP
# ==========================================

df = load_data(file_path)
if df is None: exit()

print(f"\n{Color.YELLOW}Data Range: {df['Date'].min().date()} to {df['Date'].max().date()}{Color.END}")
print("1. Single Cycle")
print("2. Continuous Test")
mode = input("Enter Option: ").strip()

if mode == '2':
    s_date = input("Enter Start Date (YYYY-MM-DD): ")
    e_date = input("Enter End Date   (YYYY-MM-DD): ")
    
    try:
        start_dt = pd.to_datetime(s_date)
        end_dt = pd.to_datetime(e_date)
        current_idx = df[df['Date'] >= start_dt].index[0]
        
        grand_total = 0
        cycle_count = 0
        next_entry_price = None 
        cycle_summaries = [] # Store short summary for end table
        
        print_header("CONTINUOUS SIMULATION STARTED")
        
        while True:
            if df.iloc[current_idx]['Date'] > end_dt:
                break
                
            result = simulate_cycle(df, current_idx, forced_entry_price=next_entry_price)
            if result is None: break
            
            cycle_count += 1
            grand_total += result['profit']
            
            # FORMATTING OUTPUT
            p_color = Color.GREEN if result['profit'] >= 0 else Color.RED
            reason_str = f"{p_color}{result['reason']}{Color.END}"
            
            print(f"\n{Color.BOLD}CYCLE {cycle_count}{Color.END} | {result['ledger'][0]['Date']} -> {result['end_date'].date()}")
            print(f"Outcome: {reason_str} | P/L: {p_color}{format_currency(result['profit'])}{Color.END}")
            
            # Create formatted ledger dataframe
            ledger_df = pd.DataFrame(result['ledger'])
            ledger_df['Price'] = ledger_df['Price'].apply(format_currency)
            ledger_df['AvgPrice'] = ledger_df['AvgPrice'].apply(format_currency)
            
            print("-" * 75)
            # Use format string for aligned columns
            print(f"{'Date':<12} {'Action':<6} {'Leg':<8} {'Qty':<4} {'Price':>12} {'AvgPrice':>12} {'Status':<15}")
            for _, row in ledger_df.iterrows():
                act_color = Color.GREEN if row['Action'] == 'SELL' else ''
                print(f"{str(row['Date']):<12} {act_color}{row['Action']:<6}{Color.END} {row['Leg']:<8} {row['Qty']:<4} {row['Price']:>12} {row['AvgPrice']:>12} {row['Status']:<15}")
            print("-" * 75)

            # Store for summary table
            cycle_summaries.append([
                cycle_count, 
                result['ledger'][0]['Date'], 
                result['end_date'].date(), 
                result['reason'], 
                result['profit']
            ])
            
            if result['status'] == 'Open':
                print(f"{Color.YELLOW}Data ran out with position open.{Color.END}")
                break
            
            # Setup Next Cycle
            current_idx = result['end_index'] 
            next_entry_price = result['exit_price'] + 5
            
            # Expiry Safety Skip
            if df.iloc[current_idx]['Date'] >= df.iloc[current_idx]['Expiry Date']:
                 if current_idx + 1 < len(df): current_idx += 1
                 else: break
        
        # --- GRAND SUMMARY TABLE ---
        print_header("FINAL PERFORMANCE SUMMARY")
        print(f"{'#':<3} {'Start':<12} {'End':<12} {'Outcome':<15} {'Profit/Loss':>15}")
        print("-" * 60)
        
        for row in cycle_summaries:
            c_num, c_start, c_end, c_reason, c_pl = row
            pl_color = Color.GREEN if c_pl >= 0 else Color.RED
            print(f"{c_num:<3} {str(c_start):<12} {str(c_end):<12} {c_reason:<15} {pl_color}{format_currency(c_pl):>15}{Color.END}")
            
        print("-" * 60)
        total_color = Color.GREEN if grand_total >= 0 else Color.RED
        print(f"{Color.BOLD}TOTAL CYCLES:{Color.END} {cycle_count}")
        print(f"{Color.BOLD}TOTAL P/L:   {Color.END} {total_color}{format_currency(grand_total)}{Color.END}")
        print("=" * 60)

    except IndexError:
        print("Start date not found.")
    except Exception as e:
        print(f"Error: {e}")

elif mode == '1':
    # Single Cycle Logic (Simplified for brevity, uses same structure)
    pass # You can use the loop logic for single cycle too