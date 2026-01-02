# Jolly Gold 2 Strategy Backtester

A Python-based web application for backtesting a multi-leg averaging strategy on Indian Commodity Market (MCX) data. This tool allows traders to simulate systematic strategies, visualize equity curves, and analyze trade-by-trade performance using a user-friendly interface.

## Overview

The system simulates a trading strategy that averages down into a position using multiple "legs" when the price drops by specific percentage gaps. It aims for a 1% profit target on the total average price while managing risk through time-based exits (Contract Expiry).

## Features

* **Dual Modes:**
    * **Single Cycle:** Run a detailed simulation for one specific trade starting on a specific date.
    * **Continuous Backtest:** Simulate back-to-back trades over a long period to test long-term viability.
* **Dynamic Strategy Configuration:**
    * Adjust the **Number of Legs** (from 1 to 20).
    * Customize **Lot Sizes** and **Gap Percentages** for every leg.
* **Robust Data Handling:**
    * Supports `.csv`, `.xlsx`, and `.xls` (including HTML-based fake XLS files).
    * Intelligent date parsing (handles "30 Apr 2021", "05-05-2021", etc.).
    * Automatic expiry date detection.
* **Visual Analytics:**
    * **Equity Curve:** Interactive chart showing capital growth/drawdown with color-coded markers (Green=Profit, Red=Loss).
    * **Cycle PnL Bar Chart:** Visual breakdown of every trade cycle.
    * **Detailed Ledger:** Exportable CSV logs of every buy/sell action.

## Strategy Logic

### Entry Rules
1.  **Leg 1**:
    * **Single Mode:** Buys at the **High** of the user-selected start date.
    * **Continuous Mode:** Buys at the **Previous Cycle's Exit Price + 5 points**.
2.  **Subsequent Legs**:
    * Triggered when the Low price drops below a calculated gap.
    * Gap logic: `Min(AvgPrice - Gap%, PrevClose - Gap%)`.
    * **Gap Down Protection:** If the market opens below the trigger price, the system buys at the **Open** price (getting a better fill).

### Exit Rules
1.  **Profit Target**: Immediate exit when High price $\ge$ **Average Price + 1%**.
2.  **Contract Expiry**: Forced exit on the expiry date found in the data file.
    * If High $\ge$ Avg Price: Exit at Avg Price (No Profit/No Loss).
    * If High $<$ Avg Price: Exit at Close Price (Realize Loss).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/jolly-gold-backtester.git](https://github.com/your-username/jolly-gold-backtester.git)
    cd jolly-gold-backtester
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Contents of `requirements.txt`:*
    ```text
    streamlit
    pandas
    plotly
    openpyxl
    xlrd
    lxml
    ```

## Usage

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
2.  **Open your browser:** The app will typically run at `http://localhost:8501`.
3.  **Upload Data:** Upload your MCX Bhavcopy file (CSV/Excel).
4.  **Configure:** Set your Lot sizes and Gap percentages in the sidebar.
5.  **Run:** Click "Run Simulation".

## Data Format

The application expects a file with the following columns (names are flexible/case-insensitive):

* `Date`: Trading date (e.g., "30 Apr 2021").
* `Expiry Date`: Contract expiry date (e.g., "05May2021").
* `Open`, `High`, `Low`, `Close`: Price data.

## License

This project is provided for educational and research purposes. Past performance in backtesting does not guarantee future results.