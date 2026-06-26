"""
DAILY INDIA MARKET SCAN (NSE-listed) + EMAIL

Conditions (all must pass):
  1. Quarterly Revenue growth    > 30%  vs same quarter last year
  2. Quarterly Net Income growth > 50%  vs same quarter last year
  3. Price > ₹100
  4. Market Cap between ₹1,000 Crores and ₹20,000 Crores
  5. Average Volume (last 30 days) > 2,00,000
  6. All 3 SuperTrends (13,4) / (14,5) / (15,6) are GREEN today
  7. Price is above 200 DEMA (Double EMA) on the day all STs turn green

Two email lists:
  A) ALIGNED - all conditions met
  B) FRESH   - subset of A where >=1 SuperTrend flipped red->green TODAY

Runs every weekday at 11:30 AM UTC (5:00 PM IST), after NSE closes at 3:30 PM IST.
"""

import os
import io
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import yfinance as yf

# ---- EMAIL SETTINGS ----
GMAIL_USER        = "tomailsasidhar@gmail.com"
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")  # GitHub Secret
TO_EMAIL          = "tomailsasidhar@gmail.com"

# ---- FILTER THRESHOLDS ----
MIN_PRICE        = 100                  # ₹100 minimum price
MIN_MCAP_CR      = 1_000               # ₹1,000 Crores
MAX_MCAP_CR      = 20_000              # ₹20,000 Crores
MIN_AVG_VOL      = 200_000             # 2 lakh average volume
CRORE            = 1_00_00_000        # 1 Crore = 10 million

ST_PARAMS        = [(13, 4), (14, 5), (15, 6)]
DELAY            = 1.2    # increased from 0.4 — cloud IPs need more breathing room
MAX_RETRIES      = 2      # retry each stock up to 2 extra times on failure


# ------------------------------------------------------------------ #
#  SuperTrend                                                          #
# ------------------------------------------------------------------ #
def supertrend(df, period, multiplier):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    hl2 = (high + low) / 2
    upper, lower = hl2 + multiplier * atr, hl2 - multiplier * atr
    final_upper, final_lower = upper.copy(), lower.copy()
    trend = pd.Series(index=df.index, dtype=float)
    st    = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i - 1]):
            continue
        final_upper.iloc[i] = (upper.iloc[i]
            if upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]
            else final_upper.iloc[i-1])
        final_lower.iloc[i] = (lower.iloc[i]
            if lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]
            else final_lower.iloc[i-1])

    for i in range(period, len(df)):
        if close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1] if i > period else 1
        st.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]

    return st


def dema(series, period):
    """Double Exponential Moving Average = 2*EMA(n) - EMA(EMA(n))"""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


# ------------------------------------------------------------------ #
#  Per-stock check (with retry on failure)                             #
# ------------------------------------------------------------------ #
def check_stock_once(symbol):
    t = yf.Ticker(symbol)

    # ---- 1 & 2: Quarterly fundamental growth ----
    q = t.quarterly_income_stmt
    if q is None or q.empty or q.shape[1] < 2:
        return None

    q = q.sort_index(axis=1, ascending=False)
    q.columns = pd.to_datetime(q.columns)

    rev = get_row(q, ["Total Revenue", "TotalRevenue"])
    ni  = get_row(q, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
    if rev is None or ni is None:
        return None

    latest_date  = q.columns[0]
    one_year_ago = latest_date - pd.DateOffset(years=1)
    time_diffs   = abs(q.columns - one_year_ago)
    same_qtr_idx = time_diffs.argmin()

    if time_diffs[same_qtr_idx] > pd.Timedelta(days=45):
        return None

    rev_curr, rev_prev = rev.iloc[0], rev.iloc[same_qtr_idx]
    ni_curr,  ni_prev  = ni.iloc[0],  ni.iloc[same_qtr_idx]

    if rev_prev == 0 or ni_prev == 0:
        return None

    sales_growth  = (rev_curr - rev_prev) / abs(rev_prev) * 100
    profit_growth = (ni_curr  - ni_prev)  / abs(ni_prev)  * 100
    if sales_growth < 30 or profit_growth < 50:
        return None

    # ---- 3: Price > ₹100  (fetch 2y so 200 DEMA has enough data) ----
    hist = t.history(period="2y")
    hist = hist.dropna(subset=["Close", "High", "Low", "Volume"])
    if len(hist) < 250:
        return None

    price = hist["Close"].iloc[-1]
    if price < MIN_PRICE:
        return None

    # ---- 4: Market Cap ₹1,000 Cr – ₹20,000 Cr ----
    info    = t.fast_info
    mcap    = getattr(info, "market_cap", None)
    if mcap is None or mcap == 0:
        return None
    mcap_cr = mcap / CRORE
    if not (MIN_MCAP_CR <= mcap_cr <= MAX_MCAP_CR):
        return None

    # ---- 5: Avg Volume (last 30 trading days) > 2,00,000 ----
    avg_vol = hist["Volume"].tail(30).mean()
    if avg_vol < MIN_AVG_VOL:
        return None

    # ---- 6: All 3 SuperTrends green today, track flips ----
    price_yday = hist["Close"].iloc[-2]
    flipped    = []
    for period, mult in ST_PARAMS:
        st_series         = supertrend(hist, period, mult)
        st_today, st_yday = st_series.iloc[-1], st_series.iloc[-2]
        if pd.isna(st_today) or pd.isna(st_yday):
            return None
        if price <= st_today:
            return None
        if price_yday <= st_yday:
            flipped.append(f"{period},{mult}")

    # ---- 7: Price > 200 DEMA ----
    dema200 = dema(hist["Close"], 200).iloc[-1]
    if pd.isna(dema200) or price <= dema200:
        return None

    return (
        symbol.replace(".NS", ""),
        round(price, 2),
        round(mcap_cr, 0),
        round(sales_growth, 1),
        round(profit_growth, 1),
        round(avg_vol / 1000, 1),
        ",".join(flipped)
    )


def check_stock(symbol):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return check_stock_once(symbol)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)   # wait 1s, then 2s before retrying
            else:
                print(f"  SKIP {symbol}: {type(e).__name__}: {e}")
                return None


# ------------------------------------------------------------------ #
#  Email                                                               #
# ------------------------------------------------------------------ #
def format_section(rows, title):
    if not rows:
        return f"{title}: none today\n"
    header = f"{'Symbol':<12}{'Price(₹)':>10}{'MCap(Cr)':>10}{'SalesG%':>9}{'ProfitG%':>10}{'AvgVol(K)':>11}  Flipped"
    lines  = [f"{title}: {len(rows)} stock(s)", header]
    for r in rows:
        lines.append(
            f"{r[0]:<12}{r[1]:>10}{r[2]:>10,.0f}{r[3]:>9}{r[4]:>10}  {r[5]:>9}K  {r[6] or '-'}"
        )
    return "\n".join(lines) + "\n"


def send_email(aligned, fresh, scanned, elapsed_min):
    body  = "Daily NSE India Stock Scan Results\n"
    body += "=" * 60 + "\n\n"
    body += format_section(fresh,   "B) FRESH  — >=1 SuperTrend flipped to GREEN today") + "\n"
    body += format_section(aligned, "A) ALIGNED — all 3 SuperTrends GREEN today")        + "\n"
    body += f"\nScanned {scanned} NSE stocks in {elapsed_min:.0f} min.\n"
    body += "Conditions: SalesG>30% | ProfitG>50% | Price>₹100 | MCap ₹1K-20K Cr | AvgVol>2L | All 3 ST Green | Price>200 DEMA\n"

    msg            = MIMEMultipart()
    msg["From"]    = GMAIL_USER
    msg["To"]      = TO_EMAIL
    msg["Subject"] = f"NSE Scan: {len(fresh)} fresh signal(s), {len(aligned)} aligned — {pd.Timestamp.now().strftime('%d %b %Y')}"
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
start = time.time()
print("Loading NSE ticker list...")

try:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp    = requests.get(
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        headers=headers, timeout=30
    )
    nse_df  = pd.read_csv(io.StringIO(resp.text))
    symbols = nse_df["SYMBOL"].dropna().tolist()
except Exception as e:
    print(f"NSE direct fetch failed ({e}), trying fallback...")
    # Fallback: use nse-listed tickers via a GitHub-hosted backup list
    resp    = requests.get(
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        timeout=30
    )
    # If fallback also fails, abort cleanly
    print("Could not load ticker list. Exiting.")
    raise SystemExit(1)

# Add .NS suffix for yfinance, filter out bad symbols
tickers = [f"{s}.NS" for s in symbols
           if isinstance(s, str) and s.isalpha() or
           (isinstance(s, str) and all(c.isalnum() or c in "-&" for c in s))]
tickers = [t for t in tickers if 5 <= len(t) <= 15]

print(f"Scanning {len(tickers)} NSE stocks...\n")

aligned, fresh = [], []
for i, sym in enumerate(tickers, 1):
    result = check_stock(sym)
    if result:
        aligned.append(result)
        if result[6]:                           # has flipped SuperTrend(s)
            fresh.append(result)
        tag = "FRESH  " if result[6] else "ALIGNED"
        print(f"{tag}: {result[0]:<12} ₹{result[1]:<8} MCap={result[2]:,.0f}Cr "
              f"SalesG={result[3]}% ProfitG={result[4]}% Flipped={result[6] or '-'}")
    if i % 200 == 0:
        print(f"  ...{i}/{len(tickers)} scanned, {len(aligned)} matches so far")
    time.sleep(DELAY)

elapsed = (time.time() - start) / 60
print(f"\nDone — ALIGNED: {len(aligned)}  |  FRESH: {len(fresh)}  |  Time: {elapsed:.0f} min")

# Save CSVs
cols = ["Symbol", "Price(₹)", "MCap(Cr)", "SalesGrowth%", "ProfitGrowth%", "AvgVol(K)", "Flipped"]
if aligned:
    pd.DataFrame(aligned, columns=cols).to_csv("india_aligned.csv", index=False)
    print("Saved india_aligned.csv")
if fresh:
    pd.DataFrame(fresh, columns=cols).to_csv("india_fresh.csv", index=False)
    print("Saved india_fresh.csv")

# Send email
print("Sending email...")
try:
    send_email(aligned, fresh, len(tickers), elapsed)
    print("Email sent successfully to", TO_EMAIL)
except Exception as e:
    print(f"Email failed: {e}")
