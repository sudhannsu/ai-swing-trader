# utils/nifty_symbols.py
from nsetools import Nse

def fetch_top_nifty_symbols(limit=50):
    """
    Returns a list of top Nifty stock symbols (e.g. RELIANCE.NS, TCS.NS).
    Defaults to first 50 stocks from NSE stock list.
    """
    nse = Nse()
    codes = nse.get_stock_codes()  # dict: {SYMBOL: Company Name}
    if isinstance(codes, dict):
    stock_list = list(codes.keys())[1:]
    else:
    raise ValueError("Expected codes to be a dict, but got:", type(codes))
    # Remove the first key which is 'SYMBOL'
   # stock_list = list(codes.keys())[1:]

    # Append '.NS' to match yfinance format
    stock_list = [symbol + '.NS' for symbol in stock_list[:limit]]

    return stock_list
