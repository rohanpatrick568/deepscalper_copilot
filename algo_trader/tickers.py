"""
tickers.py — S&P 100 ticker list (trading universe for the DeepScalper system).

The S&P 100 Index consists of 100 major blue-chip companies drawn from the S&P 500.
This module is imported by config.py and every module that needs to iterate over
the trading universe.

Notes:
    - BRK.B is the Alpaca-compatible symbol for Berkshire Hathaway Class B.
    - GOOG and GOOGL are both included (Alphabet Class C and Class A respectively).
    - This list should be periodically reviewed for index rebalancing events.
"""

SP100_TICKERS: list[str] = [
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "AMZN",   # Amazon.com Inc.
    "NVDA",   # NVIDIA Corporation
    "GOOGL",  # Alphabet Inc. Class A
    "GOOG",   # Alphabet Inc. Class C
    "META",   # Meta Platforms Inc.
    "TSLA",   # Tesla Inc.
    "BRK.B",  # Berkshire Hathaway Inc. Class B
    "UNH",    # UnitedHealth Group Inc.
    "LLY",    # Eli Lilly and Company
    "JPM",    # JPMorgan Chase & Co.
    "V",      # Visa Inc.
    "AVGO",   # Broadcom Inc.
    "XOM",    # Exxon Mobil Corporation
    "MA",     # Mastercard Incorporated
    "COST",   # Costco Wholesale Corporation
    "PG",     # The Procter & Gamble Company
    "JNJ",    # Johnson & Johnson
    "HD",     # The Home Depot Inc.
    "ABBV",   # AbbVie Inc.
    "ORCL",   # Oracle Corporation
    "BAC",    # Bank of America Corporation
    "WMT",    # Walmart Inc.
    "NFLX",   # Netflix Inc.
    "KO",     # The Coca-Cola Company
    "CRM",    # Salesforce Inc.
    "CVX",    # Chevron Corporation
    "MRK",    # Merck & Co. Inc.
    "AMD",    # Advanced Micro Devices Inc.
    "CSCO",   # Cisco Systems Inc.
    "PEP",    # PepsiCo Inc.
    "ACN",    # Accenture plc
    "LIN",    # Linde plc
    "TMO",    # Thermo Fisher Scientific Inc.
    "MCD",    # McDonald's Corporation
    "ABT",    # Abbott Laboratories
    "IBM",    # International Business Machines Corporation
    "GE",     # GE Aerospace
    "TXN",    # Texas Instruments Incorporated
    "PM",     # Philip Morris International Inc.
    "GS",     # The Goldman Sachs Group Inc.
    "ISRG",   # Intuitive Surgical Inc.
    "CAT",    # Caterpillar Inc.
    "AXP",    # American Express Company
    "SPGI",   # S&P Global Inc.
    "AMGN",   # Amgen Inc.
    "RTX",    # RTX Corporation
    "PFE",    # Pfizer Inc.
    "BKNG",   # Booking Holdings Inc.
    "DHR",    # Danaher Corporation
    "MS",     # Morgan Stanley
    "INTU",   # Intuit Inc.
    "BLK",    # BlackRock Inc.
    "T",      # AT&T Inc.
    "VRTX",   # Vertex Pharmaceuticals Incorporated
    "HON",    # Honeywell International Inc.
    "NEE",    # NextEra Energy Inc.
    "UNP",    # Union Pacific Corporation
    "SYK",    # Stryker Corporation
    "C",      # Citigroup Inc.
    "LOW",    # Lowe's Companies Inc.
    "TJX",    # The TJX Companies Inc.
    "ADP",    # Automatic Data Processing Inc.
    "GILD",   # Gilead Sciences Inc.
    "DE",     # Deere & Company
    "PANW",   # Palo Alto Networks Inc.
    "BMY",    # Bristol-Myers Squibb Company
    "AMAT",   # Applied Materials Inc.
    "MDT",    # Medtronic plc
    "PLD",    # Prologis Inc.
    "SBUX",   # Starbucks Corporation
    "ADI",    # Analog Devices Inc.
    "TMUS",   # T-Mobile US Inc.
    "ETN",    # Eaton Corporation plc
    "SCHW",   # The Charles Schwab Corporation
    "CB",     # Chubb Limited
    "MMC",    # Marsh & McLennan Companies Inc.
    "BA",     # The Boeing Company
    "SO",     # The Southern Company
    "MO",     # Altria Group Inc.
    "WFC",    # Wells Fargo & Company
    "UPS",    # United Parcel Service Inc.
    "CI",     # The Cigna Group
    "MDLZ",   # Mondelez International Inc.
    "DUK",    # Duke Energy Corporation
    "CL",     # Colgate-Palmolive Company
    "INTC",   # Intel Corporation
    "REGN",   # Regeneron Pharmaceuticals Inc.
    "PH",     # Parker-Hannifin Corporation
    "EOG",    # EOG Resources Inc.
    "SLB",    # Schlumberger Limited (SLB)
    "ELV",    # Elevance Health Inc.
    "APD",    # Air Products and Chemicals Inc.
    "MCK",    # McKesson Corporation
    "COF",    # Capital One Financial Corporation
    "ZTS",    # Zoetis Inc.
    "BSX",    # Boston Scientific Corporation
    "GEV",    # GE Vernova
    "CME",    # CME Group Inc.
]

# Sanity check: ensure exactly 100 tickers
assert len(SP100_TICKERS) == 100, (
    f"SP100_TICKERS must contain exactly 100 symbols, got {len(SP100_TICKERS)}"
)
