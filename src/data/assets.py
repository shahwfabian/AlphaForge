"""
AlphaForge — Comprehensive Asset Directory
===========================================
Curated list of ~300 tradeable assets available on major US brokerages
(Robinhood, Fidelity, Schwab, etc.) organised by category.

Each entry: (ticker, full_name, category)

Tickers are yfinance-compatible. Used by the Streamlit sidebar to power
the searchable Ticker Symbol and Benchmark selectboxes.
"""

from __future__ import annotations

# ── Master asset list ─────────────────────────────────────────────────────────
# Ordered: broad market first so SPY / QQQ appear at top of every dropdown.

ASSET_LIST: list[tuple[str, str, str]] = [

    # ── Broad Market ETFs ─────────────────────────────────────────────────────
    ("SPY",   "SPDR S&P 500 ETF Trust",                          "Broad Market ETF"),
    ("IVV",   "iShares Core S&P 500 ETF",                        "Broad Market ETF"),
    ("VOO",   "Vanguard S&P 500 ETF",                            "Broad Market ETF"),
    ("QQQ",   "Invesco QQQ Trust (NASDAQ-100)",                   "Broad Market ETF"),
    ("VTI",   "Vanguard Total Stock Market ETF",                  "Broad Market ETF"),
    ("ITOT",  "iShares Core S&P Total U.S. Stock Market ETF",    "Broad Market ETF"),
    ("SCHB",  "Schwab U.S. Broad Market ETF",                    "Broad Market ETF"),
    ("IWM",   "iShares Russell 2000 ETF (Small-Cap)",            "Broad Market ETF"),
    ("MDY",   "SPDR S&P MidCap 400 ETF",                         "Broad Market ETF"),
    ("IJR",   "iShares Core S&P Small-Cap ETF",                  "Broad Market ETF"),
    ("DIA",   "SPDR Dow Jones Industrial Average ETF",           "Broad Market ETF"),
    ("RSP",   "Invesco S&P 500 Equal Weight ETF",                "Broad Market ETF"),
    ("OEF",   "iShares S&P 100 ETF",                             "Broad Market ETF"),
    ("VEA",   "Vanguard FTSE Developed Markets ETF",             "Broad Market ETF"),
    ("VWO",   "Vanguard FTSE Emerging Markets ETF",              "Broad Market ETF"),
    ("EFA",   "iShares MSCI EAFE ETF (Developed ex-US)",         "Broad Market ETF"),
    ("EEM",   "iShares MSCI Emerging Markets ETF",               "Broad Market ETF"),
    ("ACWI",  "iShares MSCI ACWI ETF (All Country World)",       "Broad Market ETF"),
    ("VT",    "Vanguard Total World Stock ETF",                  "Broad Market ETF"),

    # ── Sector ETFs (SPDR Select) ─────────────────────────────────────────────
    ("XLK",   "Technology Select Sector SPDR Fund",              "Sector ETF"),
    ("XLF",   "Financial Select Sector SPDR Fund",               "Sector ETF"),
    ("XLV",   "Health Care Select Sector SPDR Fund",             "Sector ETF"),
    ("XLY",   "Consumer Discretionary Select Sector SPDR Fund",  "Sector ETF"),
    ("XLP",   "Consumer Staples Select Sector SPDR Fund",        "Sector ETF"),
    ("XLE",   "Energy Select Sector SPDR Fund",                  "Sector ETF"),
    ("XLI",   "Industrial Select Sector SPDR Fund",              "Sector ETF"),
    ("XLU",   "Utilities Select Sector SPDR Fund",               "Sector ETF"),
    ("XLB",   "Materials Select Sector SPDR Fund",               "Sector ETF"),
    ("XLC",   "Communication Services Select Sector SPDR Fund",  "Sector ETF"),
    ("XLRE",  "Real Estate Select Sector SPDR Fund",             "Sector ETF"),
    ("XBI",   "SPDR S&P Biotech ETF",                            "Sector ETF"),
    ("IBB",   "iShares Biotechnology ETF",                       "Sector ETF"),
    ("SMH",   "VanEck Semiconductor ETF",                        "Sector ETF"),
    ("SOXX",  "iShares Semiconductor ETF",                       "Sector ETF"),
    ("IGV",   "iShares Expanded Tech-Software Sector ETF",       "Sector ETF"),
    ("HACK",  "ETFMG Prime Cyber Security ETF",                  "Sector ETF"),
    ("FINX",  "Global X FinTech ETF",                            "Sector ETF"),
    ("KBWB",  "Invesco KBW Bank ETF",                            "Sector ETF"),
    ("JETS",  "U.S. Global JETS ETF (Airlines)",                 "Sector ETF"),
    ("VNQ",   "Vanguard Real Estate ETF",                        "Sector ETF"),
    ("REET",  "iShares Global REIT ETF",                         "Sector ETF"),
    ("KRE",   "SPDR S&P Regional Banking ETF",                   "Sector ETF"),
    ("KBE",   "SPDR S&P Bank ETF",                               "Sector ETF"),
    ("IAT",   "iShares U.S. Regional Banks ETF",                 "Sector ETF"),

    # ── Bond ETFs ─────────────────────────────────────────────────────────────
    ("AGG",   "iShares Core U.S. Aggregate Bond ETF",            "Bond ETF"),
    ("BND",   "Vanguard Total Bond Market ETF",                  "Bond ETF"),
    ("TLT",   "iShares 20+ Year Treasury Bond ETF",              "Bond ETF"),
    ("IEF",   "iShares 7-10 Year Treasury Bond ETF",             "Bond ETF"),
    ("SHY",   "iShares 1-3 Year Treasury Bond ETF",              "Bond ETF"),
    ("GOVT",  "iShares U.S. Treasury Bond ETF",                  "Bond ETF"),
    ("VGSH",  "Vanguard Short-Term Treasury ETF",                "Bond ETF"),
    ("VGIT",  "Vanguard Intermediate-Term Treasury ETF",         "Bond ETF"),
    ("VGLT",  "Vanguard Long-Term Treasury ETF",                 "Bond ETF"),
    ("BSV",   "Vanguard Short-Term Bond ETF",                    "Bond ETF"),
    ("BIV",   "Vanguard Intermediate-Term Bond ETF",             "Bond ETF"),
    ("BLV",   "Vanguard Long-Term Bond ETF",                     "Bond ETF"),
    ("LQD",   "iShares iBoxx $ Investment Grade Corporate Bond ETF", "Bond ETF"),
    ("VCSH",  "Vanguard Short-Term Corporate Bond ETF",          "Bond ETF"),
    ("IGSB",  "iShares Short-Term Corporate Bond ETF",           "Bond ETF"),
    ("IGIB",  "iShares Intermediate-Term Corporate Bond ETF",    "Bond ETF"),
    ("HYG",   "iShares iBoxx $ High Yield Corporate Bond ETF",   "Bond ETF"),
    ("JNK",   "SPDR Bloomberg High Yield Bond ETF",              "Bond ETF"),
    ("USHY",  "iShares Broad USD High Yield Corporate Bond ETF", "Bond ETF"),
    ("ANGL",  "VanEck Fallen Angel High Yield Bond ETF",         "Bond ETF"),
    ("FALN",  "iShares Fallen Angels USD Bond ETF",              "Bond ETF"),
    ("MUB",   "iShares National Muni Bond ETF",                  "Bond ETF"),
    ("BNDX",  "Vanguard Total International Bond ETF",           "Bond ETF"),
    ("EMB",   "iShares J.P. Morgan USD Emerging Markets Bond ETF", "Bond ETF"),
    ("TIPS",  "iShares TIPS Bond ETF",                           "Bond ETF"),
    ("SCHP",  "Schwab U.S. TIPS ETF",                            "Bond ETF"),
    ("VTIP",  "Vanguard Short-Term Inflation-Protected Securities ETF", "Bond ETF"),
    ("PFF",   "iShares Preferred & Income Securities ETF",       "Bond ETF"),
    ("SJNK",  "SPDR Bloomberg Short Term High Yield Bond ETF",   "Bond ETF"),

    # ── Commodity ETFs ────────────────────────────────────────────────────────
    ("GLD",   "SPDR Gold Shares",                                "Commodity ETF"),
    ("IAU",   "iShares Gold Trust",                              "Commodity ETF"),
    ("GLDM",  "SPDR Gold MiniShares Trust",                      "Commodity ETF"),
    ("GDX",   "VanEck Gold Miners ETF",                          "Commodity ETF"),
    ("GDXJ",  "VanEck Junior Gold Miners ETF",                   "Commodity ETF"),
    ("SLV",   "iShares Silver Trust",                            "Commodity ETF"),
    ("PPLT",  "Aberdeen Standard Physical Platinum Shares ETF",  "Commodity ETF"),
    ("USO",   "United States Oil Fund LP",                       "Commodity ETF"),
    ("BNO",   "United States Brent Oil Fund LP",                 "Commodity ETF"),
    ("UNG",   "United States Natural Gas Fund LP",               "Commodity ETF"),
    ("PDBC",  "Invesco Optimum Yield Diversified Commodity Strategy ETF", "Commodity ETF"),
    ("DBC",   "Invesco DB Commodity Index Tracking Fund",        "Commodity ETF"),
    ("CPER",  "United States Copper Index Fund",                 "Commodity ETF"),
    ("CORN",  "Teucrium Corn Fund",                              "Commodity ETF"),
    ("WEAT",  "Teucrium Wheat Fund",                             "Commodity ETF"),
    ("SOYB",  "Teucrium Soybean Fund",                           "Commodity ETF"),

    # ── Thematic / Factor ETFs ────────────────────────────────────────────────
    ("ARKK",  "ARK Innovation ETF",                              "Thematic ETF"),
    ("ARKG",  "ARK Genomic Revolution ETF",                      "Thematic ETF"),
    ("ARKW",  "ARK Next Generation Internet ETF",                "Thematic ETF"),
    ("ARKF",  "ARK Fintech Innovation ETF",                      "Thematic ETF"),
    ("ARKQ",  "ARK Autonomous Technology & Robotics ETF",        "Thematic ETF"),
    ("BOTZ",  "Global X Robotics & Artificial Intelligence ETF", "Thematic ETF"),
    ("ROBO",  "ROBO Global Robotics and Automation Index ETF",   "Thematic ETF"),
    ("ICLN",  "iShares Global Clean Energy ETF",                 "Thematic ETF"),
    ("TAN",   "Invesco Solar ETF",                               "Thematic ETF"),
    ("PBW",   "Invesco WilderHill Clean Energy ETF",             "Thematic ETF"),
    ("FAN",   "First Trust Global Wind Energy ETF",              "Thematic ETF"),
    ("ESGV",  "Vanguard ESG U.S. Stock ETF",                     "Thematic ETF"),
    ("ESGU",  "iShares MSCI USA ESG Select ETF",                 "Thematic ETF"),
    ("MTUM",  "iShares MSCI USA Momentum Factor ETF",            "Thematic ETF"),
    ("VLUE",  "iShares MSCI USA Value Factor ETF",               "Thematic ETF"),
    ("QUAL",  "iShares MSCI USA Quality Factor ETF",             "Thematic ETF"),
    ("USMV",  "iShares MSCI USA Min Vol Factor ETF",             "Thematic ETF"),
    ("SIZE",  "iShares MSCI USA Size Factor ETF",                "Thematic ETF"),

    # ── Leveraged / Inverse ETFs ──────────────────────────────────────────────
    ("TQQQ",  "ProShares UltraPro QQQ (3x Long NASDAQ-100)",     "Leveraged ETF"),
    ("UPRO",  "ProShares UltraPro S&P500 (3x Long S&P 500)",     "Leveraged ETF"),
    ("SSO",   "ProShares Ultra S&P500 (2x Long S&P 500)",        "Leveraged ETF"),
    ("SOXL",  "Direxion Daily Semiconductor Bull 3x Shares",     "Leveraged ETF"),
    ("TECL",  "Direxion Daily Technology Bull 3X Shares",        "Leveraged ETF"),
    ("SQQQ",  "ProShares UltraPro Short QQQ (3x Short NASDAQ)",  "Leveraged ETF"),
    ("SPXS",  "Direxion Daily S&P 500 Bear 3X Shares",           "Leveraged ETF"),
    ("TECS",  "Direxion Daily Technology Bear 3X Shares",        "Leveraged ETF"),
    ("VXX",   "iPath Series B S&P 500 VIX Short-Term Futures ETN","Leveraged ETF"),
    ("UVXY",  "ProShares Ultra VIX Short-Term Futures ETF",      "Leveraged ETF"),
    ("SVXY",  "ProShares Short VIX Short-Term Futures ETF",      "Leveraged ETF"),

    # ── Crypto ETFs ───────────────────────────────────────────────────────────
    ("IBIT",  "iShares Bitcoin Trust ETF",                       "Crypto ETF"),
    ("FBTC",  "Fidelity Wise Origin Bitcoin Fund",               "Crypto ETF"),
    ("GBTC",  "Grayscale Bitcoin Trust",                         "Crypto ETF"),
    ("ETHE",  "Grayscale Ethereum Trust",                        "Crypto ETF"),
    ("BITO",  "ProShares Bitcoin Strategy ETF",                  "Crypto ETF"),

    # ── Technology ────────────────────────────────────────────────────────────
    ("AAPL",  "Apple Inc.",                                      "Technology"),
    ("MSFT",  "Microsoft Corporation",                           "Technology"),
    ("NVDA",  "NVIDIA Corporation",                              "Technology"),
    ("AVGO",  "Broadcom Inc.",                                   "Technology"),
    ("ORCL",  "Oracle Corporation",                              "Technology"),
    ("CSCO",  "Cisco Systems Inc.",                              "Technology"),
    ("ADBE",  "Adobe Inc.",                                      "Technology"),
    ("CRM",   "Salesforce Inc.",                                 "Technology"),
    ("IBM",   "International Business Machines Corporation",     "Technology"),
    ("INTC",  "Intel Corporation",                               "Technology"),
    ("AMD",   "Advanced Micro Devices Inc.",                     "Technology"),
    ("TXN",   "Texas Instruments Inc.",                          "Technology"),
    ("QCOM",  "QUALCOMM Inc.",                                   "Technology"),
    ("ADI",   "Analog Devices Inc.",                             "Technology"),
    ("MU",    "Micron Technology Inc.",                          "Technology"),
    ("AMAT",  "Applied Materials Inc.",                          "Technology"),
    ("LRCX",  "Lam Research Corporation",                        "Technology"),
    ("KLAC",  "KLA Corporation",                                 "Technology"),
    ("MRVL",  "Marvell Technology Inc.",                         "Technology"),
    ("NXPI",  "NXP Semiconductors N.V.",                         "Technology"),
    ("PANW",  "Palo Alto Networks Inc.",                         "Technology"),
    ("CRWD",  "CrowdStrike Holdings Inc.",                       "Technology"),
    ("FTNT",  "Fortinet Inc.",                                   "Technology"),
    ("NOW",   "ServiceNow Inc.",                                 "Technology"),
    ("INTU",  "Intuit Inc.",                                     "Technology"),
    ("SNPS",  "Synopsys Inc.",                                   "Technology"),
    ("CDNS",  "Cadence Design Systems Inc.",                     "Technology"),
    ("ANSS",  "ANSYS Inc.",                                      "Technology"),
    ("WDAY",  "Workday Inc.",                                    "Technology"),
    ("DDOG",  "Datadog Inc.",                                    "Technology"),
    ("SNOW",  "Snowflake Inc.",                                  "Technology"),
    ("MDB",   "MongoDB Inc.",                                    "Technology"),
    ("NET",   "Cloudflare Inc.",                                 "Technology"),
    ("ZS",    "Zscaler Inc.",                                    "Technology"),
    ("OKTA",  "Okta Inc.",                                       "Technology"),
    ("PLTR",  "Palantir Technologies Inc.",                      "Technology"),
    ("PATH",  "UiPath Inc.",                                     "Technology"),
    ("AI",    "C3.ai Inc.",                                      "Technology"),
    ("GTLB",  "GitLab Inc.",                                     "Technology"),
    ("HUBS",  "HubSpot Inc.",                                    "Technology"),
    ("DOCU",  "DocuSign Inc.",                                   "Technology"),
    ("ZM",    "Zoom Video Communications Inc.",                  "Technology"),
    ("TWLO",  "Twilio Inc.",                                     "Technology"),
    ("RBLX",  "Roblox Corporation",                              "Technology"),
    ("U",     "Unity Software Inc.",                             "Technology"),
    ("HOOD",  "Robinhood Markets Inc.",                          "Technology"),

    # ── Communication Services / Media ────────────────────────────────────────
    ("GOOGL", "Alphabet Inc. (Class A)",                         "Communication Services"),
    ("GOOG",  "Alphabet Inc. (Class C)",                         "Communication Services"),
    ("META",  "Meta Platforms Inc.",                             "Communication Services"),
    ("NFLX",  "Netflix Inc.",                                    "Communication Services"),
    ("DIS",   "The Walt Disney Company",                         "Communication Services"),
    ("CMCSA", "Comcast Corporation",                             "Communication Services"),
    ("T",     "AT&T Inc.",                                       "Communication Services"),
    ("VZ",    "Verizon Communications Inc.",                     "Communication Services"),
    ("TMUS",  "T-Mobile US Inc.",                                "Communication Services"),
    ("SNAP",  "Snap Inc.",                                       "Communication Services"),
    ("PINS",  "Pinterest Inc.",                                  "Communication Services"),
    ("SPOT",  "Spotify Technology S.A.",                         "Communication Services"),
    ("MTCH",  "Match Group Inc.",                                "Communication Services"),
    ("WBD",   "Warner Bros. Discovery Inc.",                     "Communication Services"),
    ("PARA",  "Paramount Global",                                "Communication Services"),
    ("EA",    "Electronic Arts Inc.",                            "Communication Services"),
    ("TTWO",  "Take-Two Interactive Software Inc.",              "Communication Services"),
    ("ATVI",  "Activision Blizzard Inc.",                        "Communication Services"),

    # ── Consumer Discretionary ────────────────────────────────────────────────
    ("AMZN",  "Amazon.com Inc.",                                 "Consumer Discretionary"),
    ("TSLA",  "Tesla Inc.",                                      "Consumer Discretionary"),
    ("HD",    "The Home Depot Inc.",                             "Consumer Discretionary"),
    ("MCD",   "McDonald's Corporation",                          "Consumer Discretionary"),
    ("NKE",   "NIKE Inc.",                                       "Consumer Discretionary"),
    ("SBUX",  "Starbucks Corporation",                           "Consumer Discretionary"),
    ("TJX",   "TJX Companies Inc.",                              "Consumer Discretionary"),
    ("BKNG",  "Booking Holdings Inc.",                           "Consumer Discretionary"),
    ("LOW",   "Lowe's Companies Inc.",                           "Consumer Discretionary"),
    ("TGT",   "Target Corporation",                              "Consumer Discretionary"),
    ("ABNB",  "Airbnb Inc.",                                     "Consumer Discretionary"),
    ("UBER",  "Uber Technologies Inc.",                          "Consumer Discretionary"),
    ("LYFT",  "Lyft Inc.",                                       "Consumer Discretionary"),
    ("DKNG",  "DraftKings Inc.",                                 "Consumer Discretionary"),
    ("PTON",  "Peloton Interactive Inc.",                        "Consumer Discretionary"),
    ("CHWY",  "Chewy Inc.",                                      "Consumer Discretionary"),
    ("ETSY",  "Etsy Inc.",                                       "Consumer Discretionary"),
    ("W",     "Wayfair Inc.",                                    "Consumer Discretionary"),
    ("RIVN",  "Rivian Automotive Inc.",                          "Consumer Discretionary"),
    ("LCID",  "Lucid Group Inc.",                                "Consumer Discretionary"),
    ("GM",    "General Motors Company",                          "Consumer Discretionary"),
    ("F",     "Ford Motor Company",                              "Consumer Discretionary"),
    ("RACE",  "Ferrari N.V.",                                    "Consumer Discretionary"),

    # ── Consumer Staples ──────────────────────────────────────────────────────
    ("PG",    "Procter & Gamble Co.",                            "Consumer Staples"),
    ("KO",    "The Coca-Cola Company",                           "Consumer Staples"),
    ("PEP",   "PepsiCo Inc.",                                    "Consumer Staples"),
    ("COST",  "Costco Wholesale Corporation",                    "Consumer Staples"),
    ("WMT",   "Walmart Inc.",                                    "Consumer Staples"),
    ("PM",    "Philip Morris International Inc.",                "Consumer Staples"),
    ("MO",    "Altria Group Inc.",                               "Consumer Staples"),
    ("MDLZ",  "Mondelez International Inc.",                     "Consumer Staples"),
    ("CL",    "Colgate-Palmolive Company",                       "Consumer Staples"),
    ("KHC",   "The Kraft Heinz Company",                         "Consumer Staples"),
    ("KMB",   "Kimberly-Clark Corporation",                      "Consumer Staples"),
    ("GIS",   "General Mills Inc.",                              "Consumer Staples"),
    ("K",     "Kellanova (formerly Kellogg's)",                  "Consumer Staples"),
    ("HSY",   "The Hershey Company",                             "Consumer Staples"),
    ("STZ",   "Constellation Brands Inc.",                       "Consumer Staples"),
    ("TAP",   "Molson Coors Beverage Company",                   "Consumer Staples"),

    # ── Health Care ───────────────────────────────────────────────────────────
    ("UNH",   "UnitedHealth Group Inc.",                         "Health Care"),
    ("LLY",   "Eli Lilly and Company",                           "Health Care"),
    ("ABBV",  "AbbVie Inc.",                                     "Health Care"),
    ("MRK",   "Merck & Co. Inc.",                                "Health Care"),
    ("TMO",   "Thermo Fisher Scientific Inc.",                   "Health Care"),
    ("ABT",   "Abbott Laboratories",                             "Health Care"),
    ("DHR",   "Danaher Corporation",                             "Health Care"),
    ("AMGN",  "Amgen Inc.",                                      "Health Care"),
    ("GILD",  "Gilead Sciences Inc.",                            "Health Care"),
    ("ISRG",  "Intuitive Surgical Inc.",                         "Health Care"),
    ("SYK",   "Stryker Corporation",                             "Health Care"),
    ("BSX",   "Boston Scientific Corporation",                   "Health Care"),
    ("BDX",   "Becton Dickinson and Company",                    "Health Care"),
    ("ZTS",   "Zoetis Inc.",                                     "Health Care"),
    ("REGN",  "Regeneron Pharmaceuticals Inc.",                  "Health Care"),
    ("VRTX",  "Vertex Pharmaceuticals Inc.",                     "Health Care"),
    ("BIIB",  "Biogen Inc.",                                     "Health Care"),
    ("HUM",   "Humana Inc.",                                     "Health Care"),
    ("CI",    "The Cigna Group",                                 "Health Care"),
    ("ELV",   "Elevance Health Inc.",                            "Health Care"),
    ("CVS",   "CVS Health Corporation",                          "Health Care"),
    ("MOH",   "Molina Healthcare Inc.",                          "Health Care"),
    ("MRNA",  "Moderna Inc.",                                    "Health Care"),
    ("BNTX",  "BioNTech SE",                                     "Health Care"),
    ("PFE",   "Pfizer Inc.",                                     "Health Care"),
    ("JNJ",   "Johnson & Johnson",                               "Health Care"),
    ("BMY",   "Bristol-Myers Squibb Company",                    "Health Care"),
    ("NVO",   "Novo Nordisk A/S (ADR)",                          "Health Care"),
    ("LLY",   "Eli Lilly and Company",                           "Health Care"),

    # ── Financials ────────────────────────────────────────────────────────────
    ("JPM",   "JPMorgan Chase & Co.",                            "Financials"),
    ("BAC",   "Bank of America Corporation",                     "Financials"),
    ("WFC",   "Wells Fargo & Company",                           "Financials"),
    ("GS",    "The Goldman Sachs Group Inc.",                    "Financials"),
    ("MS",    "Morgan Stanley",                                  "Financials"),
    ("C",     "Citigroup Inc.",                                  "Financials"),
    ("USB",   "U.S. Bancorp",                                    "Financials"),
    ("PNC",   "PNC Financial Services Group Inc.",               "Financials"),
    ("TFC",   "Truist Financial Corporation",                    "Financials"),
    ("SCHW",  "Charles Schwab Corporation",                      "Financials"),
    ("BLK",   "BlackRock Inc.",                                  "Financials"),
    ("SPGI",  "S&P Global Inc.",                                 "Financials"),
    ("MCO",   "Moody's Corporation",                             "Financials"),
    ("ICE",   "Intercontinental Exchange Inc.",                  "Financials"),
    ("CME",   "CME Group Inc.",                                  "Financials"),
    ("CB",    "Chubb Limited",                                   "Financials"),
    ("PGR",   "Progressive Corporation",                         "Financials"),
    ("MMC",   "Marsh & McLennan Companies Inc.",                 "Financials"),
    ("AXP",   "American Express Company",                        "Financials"),
    ("V",     "Visa Inc.",                                       "Financials"),
    ("MA",    "Mastercard Inc.",                                 "Financials"),
    ("PYPL",  "PayPal Holdings Inc.",                            "Financials"),
    ("SQ",    "Block Inc.",                                      "Financials"),
    ("COIN",  "Coinbase Global Inc.",                            "Financials"),
    ("SOFI",  "SoFi Technologies Inc.",                          "Financials"),
    ("AFRM",  "Affirm Holdings Inc.",                            "Financials"),
    ("UPST",  "Upstart Holdings Inc.",                           "Financials"),

    # ── Industrials ───────────────────────────────────────────────────────────
    ("GE",    "GE Aerospace",                                    "Industrials"),
    ("CAT",   "Caterpillar Inc.",                                "Industrials"),
    ("BA",    "The Boeing Company",                              "Industrials"),
    ("HON",   "Honeywell International Inc.",                    "Industrials"),
    ("RTX",   "RTX Corporation",                                 "Industrials"),
    ("LMT",   "Lockheed Martin Corporation",                     "Industrials"),
    ("NOC",   "Northrop Grumman Corporation",                    "Industrials"),
    ("GD",    "General Dynamics Corporation",                    "Industrials"),
    ("DE",    "Deere & Company",                                 "Industrials"),
    ("ETN",   "Eaton Corporation plc",                           "Industrials"),
    ("EMR",   "Emerson Electric Co.",                            "Industrials"),
    ("ITW",   "Illinois Tool Works Inc.",                        "Industrials"),
    ("PH",    "Parker Hannifin Corporation",                     "Industrials"),
    ("ROK",   "Rockwell Automation Inc.",                        "Industrials"),
    ("FDX",   "FedEx Corporation",                               "Industrials"),
    ("UPS",   "United Parcel Service Inc.",                      "Industrials"),
    ("CSX",   "CSX Corporation",                                 "Industrials"),
    ("UNP",   "Union Pacific Corporation",                       "Industrials"),
    ("NSC",   "Norfolk Southern Corporation",                    "Industrials"),
    ("WM",    "Waste Management Inc.",                           "Industrials"),
    ("RSG",   "Republic Services Inc.",                          "Industrials"),

    # ── Energy ────────────────────────────────────────────────────────────────
    ("XOM",   "Exxon Mobil Corporation",                         "Energy"),
    ("CVX",   "Chevron Corporation",                             "Energy"),
    ("SLB",   "SLB (formerly Schlumberger)",                     "Energy"),
    ("EOG",   "EOG Resources Inc.",                              "Energy"),
    ("COP",   "ConocoPhillips",                                  "Energy"),
    ("OXY",   "Occidental Petroleum Corporation",                "Energy"),
    ("PXD",   "Pioneer Natural Resources Company",               "Energy"),
    ("MPC",   "Marathon Petroleum Corporation",                  "Energy"),
    ("PSX",   "Phillips 66",                                     "Energy"),
    ("VLO",   "Valero Energy Corporation",                       "Energy"),
    ("DVN",   "Devon Energy Corporation",                        "Energy"),
    ("HAL",   "Halliburton Company",                             "Energy"),
    ("BKR",   "Baker Hughes Company",                            "Energy"),
    ("KMI",   "Kinder Morgan Inc.",                              "Energy"),
    ("WMB",   "Williams Companies Inc.",                         "Energy"),

    # ── Materials ─────────────────────────────────────────────────────────────
    ("LIN",   "Linde plc",                                       "Materials"),
    ("APD",   "Air Products and Chemicals Inc.",                 "Materials"),
    ("SHW",   "Sherwin-Williams Company",                        "Materials"),
    ("ECL",   "Ecolab Inc.",                                     "Materials"),
    ("NEM",   "Newmont Corporation",                             "Materials"),
    ("FCX",   "Freeport-McMoRan Inc.",                           "Materials"),
    ("NUE",   "Nucor Corporation",                               "Materials"),
    ("VMC",   "Vulcan Materials Company",                        "Materials"),
    ("MLM",   "Martin Marietta Materials Inc.",                  "Materials"),
    ("IP",    "International Paper Company",                     "Materials"),

    # ── Real Estate ───────────────────────────────────────────────────────────
    ("PLD",   "Prologis Inc.",                                   "Real Estate"),
    ("AMT",   "American Tower Corporation",                      "Real Estate"),
    ("EQIX",  "Equinix Inc.",                                    "Real Estate"),
    ("CCI",   "Crown Castle Inc.",                               "Real Estate"),
    ("SPG",   "Simon Property Group Inc.",                       "Real Estate"),
    ("PSA",   "Public Storage",                                  "Real Estate"),
    ("O",     "Realty Income Corporation",                       "Real Estate"),
    ("WELL",  "Welltower Inc.",                                   "Real Estate"),
    ("DLR",   "Digital Realty Trust Inc.",                       "Real Estate"),
    ("AVB",   "AvalonBay Communities Inc.",                      "Real Estate"),
    ("EQR",   "Equity Residential",                              "Real Estate"),

    # ── Utilities ─────────────────────────────────────────────────────────────
    ("NEE",   "NextEra Energy Inc.",                             "Utilities"),
    ("SO",    "Southern Company",                                "Utilities"),
    ("DUK",   "Duke Energy Corporation",                         "Utilities"),
    ("D",     "Dominion Energy Inc.",                            "Utilities"),
    ("SRE",   "Sempra",                                          "Utilities"),
    ("AEP",   "American Electric Power Company Inc.",            "Utilities"),
    ("EXC",   "Exelon Corporation",                              "Utilities"),
    ("XEL",   "Xcel Energy Inc.",                                "Utilities"),
    ("ED",    "Consolidated Edison Inc.",                        "Utilities"),
    ("AWK",   "American Water Works Company Inc.",               "Utilities"),

    # ── E-Commerce / Internet ─────────────────────────────────────────────────
    ("SHOP",  "Shopify Inc.",                                    "E-Commerce"),
    ("MELI",  "MercadoLibre Inc.",                               "E-Commerce"),
    ("SE",    "Sea Limited (ADR)",                               "E-Commerce"),
    ("JD",    "JD.com Inc. (ADR)",                               "E-Commerce"),
    ("BABA",  "Alibaba Group Holding Limited (ADR)",             "E-Commerce"),
    ("PDD",   "PDD Holdings Inc. (ADR / Temu)",                  "E-Commerce"),
    ("TEMU",  "PDD Holdings Inc.",                               "E-Commerce"),

    # ── International / ADRs ─────────────────────────────────────────────────
    ("TSM",   "Taiwan Semiconductor Manufacturing (ADR)",        "International"),
    ("ASML",  "ASML Holding N.V. (ADR)",                         "International"),
    ("NVO",   "Novo Nordisk A/S (ADR)",                          "International"),
    ("SAP",   "SAP SE (ADR)",                                    "International"),
    ("TM",    "Toyota Motor Corporation (ADR)",                  "International"),
    ("SONY",  "Sony Group Corporation (ADR)",                    "International"),
    ("SAN",   "Banco Santander S.A. (ADR)",                      "International"),
    ("RIO",   "Rio Tinto plc (ADR)",                             "International"),
    ("BP",    "BP p.l.c. (ADR)",                                 "International"),
    ("SHEL",  "Shell plc (ADR)",                                 "International"),
    ("ARM",   "Arm Holdings plc",                                "International"),
]

# ── Derived lookups ───────────────────────────────────────────────────────────

# Remove duplicate tickers (keep first occurrence)
_seen: set[str] = set()
_deduped: list[tuple[str, str, str]] = []
for _entry in ASSET_LIST:
    if _entry[0] not in _seen:
        _seen.add(_entry[0])
        _deduped.append(_entry)
ASSET_LIST = _deduped

# Flat ticker → name mapping
TICKER_NAMES: dict[str, str] = {t: n for t, n, _ in ASSET_LIST}

# Flat ticker → category mapping
TICKER_CATEGORIES: dict[str, str] = {t: c for t, _, c in ASSET_LIST}

# All tickers in display order (ETFs first, then stocks)
ALL_TICKERS: list[str] = [t for t, _, _ in ASSET_LIST]

# Grouped by category
ASSET_CATEGORIES: dict[str, list[str]] = {}
for _ticker, _name, _cat in ASSET_LIST:
    ASSET_CATEGORIES.setdefault(_cat, []).append(_ticker)


def format_ticker(ticker: str) -> str:
    """
    Format a ticker for display in a Streamlit selectbox.

    Examples
    --------
    >>> format_ticker("AAPL")
    'AAPL — Apple Inc.'
    >>> format_ticker("XYZ")
    'XYZ'
    """
    name = TICKER_NAMES.get(ticker, "")
    cat  = TICKER_CATEGORIES.get(ticker, "")
    if name and cat:
        return f"{ticker}  —  {name}"
    elif name:
        return f"{ticker}  —  {name}"
    return ticker


def search_assets(query: str, max_results: int = 20) -> list[str]:
    """
    Return tickers whose ticker or name contains ``query`` (case-insensitive).
    Used for dynamic filtering if needed.
    """
    q = query.lower().strip()
    if not q:
        return ALL_TICKERS[:max_results]
    results = [
        t for t in ALL_TICKERS
        if q in t.lower() or q in TICKER_NAMES.get(t, "").lower()
    ]
    return results[:max_results]
