"""
DAILY INDIA MARKET SCAN + EMAIL

Produces two lists (same fundamentals + price/market-cap filters for both):
  A) ALIGNED - all 3 SuperTrends (13,4)/(14,5)/(15,6) green today
  B) FRESH   - subset of A where >=1 SuperTrend flipped red->green TODAY

Filters (apply to both lists):
  - Quarterly Revenue growth    > 30%      vs same quarter last year
  - Quarterly Net Income growth > 50%      vs same quarter last year
  - Price                       > Rs 50
  - Market Cap                  < Rs 20,000 Cr

Universe: all NSE main-board ("EQ" series) stocks, pulled fresh each run
from NSE's public equity list (so new listings/delistings are picked up
automatically - no manually maintained ticker file).

Emails both lists when the run finishes. Schedule with Task Scheduler / cron
/ GitHub Actions to run after NSE market close (3:30pm IST) - see chat for
setup steps.

FILL IN BEFORE RUNNING (the 3 lines below):
  GMAIL_USER, GMAIL_APP_PASSWORD, TO_EMAIL

REQUIRES: pip install yfinance pandas httpx[http2]
(httpx with the http2 extra is required - NSE blocks plain HTTP/1.1
requests from cloud/server IPs such as GitHub Actions runners; see
get_nse_universe() below for details)
"""

import os
import io
import time
import smtplib
import httpx
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

# ---- EMAIL SETTINGS ----
GMAIL_USER = "tomailsasidhar@gmail.com"   # sends from this address
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")  # set as a GitHub Secret
TO_EMAIL = "tomailsasidhar@gmail.com"     # results go here

ST_PARAMS = [(13, 4), (14, 5), (15, 6)]
DELAY = 0.3

MIN_PRICE = 50          # Rs - same numeric floor as the US version, now in rupees
MAX_MCAP_CR = 20000     # Rs Cr - stocks with market cap BELOW this pass
CR = 1e7                # 1 crore = 10,000,000 rupees


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
    st = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i - 1]):
            continue
        final_upper.iloc[i] = upper.iloc[i] if (upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]) else final_upper.iloc[i - 1]
        final_lower.iloc[i] = lower.iloc[i] if (lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]) else final_lower.iloc[i - 1]

    for i in range(period, len(df)):
        if close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1] if i > period else 1
        st.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]

    return st


def get_row(df, names):
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def get_nse_universe():
    """Pulls the current NSE main-board ('EQ' series) equity list.

    NSE silently times out plain HTTP/1.1 requests (what the `requests`
    library speaks) when they come from datacenter/cloud IPs - AWS, Azure,
    GCP, and that includes GitHub Actions runners - but is fine with the
    exact same request over HTTP/2. httpx with http2=True speaks HTTP/2,
    which is what makes this work in CI. A plain `requests` session works
    fine from a home network, which is why this can look "correct" when
    tested locally but still time out once it's running in GitHub Actions.
    Retries a few times since NSE's archive server can also just be slow.
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"),
        "Accept": "text/csv,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }
    url = "https://nsearchives.nseindia.com/content/equity_lists/EQUITY_L.csv"

    last_err = None
    for attempt in range(3):
        try:
            with httpx.Client(http2=True, headers=headers, timeout=25,
                               follow_redirects=True) as client:
                try:
                    client.get("https://www.nseindia.com")  # warm up cookies, best-effort
                except Exception:
                    pass
                resp = client.get(url)
                resp.raise_for_status()
                text = resp.text

            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.strip() for c in df.columns]
            df = df[df["SERIES"].str.strip() == "EQ"]
            return [f"{s.strip()}.NS" for s in df["SYMBOL"].dropna().tolist()]
        except Exception as e:
            last_err = e
            time.sleep(3)

    raise RuntimeError(f"Could not fetch NSE ticker list after 3 attempts: {last_err}")


def check_stock(symbol):
    try:
        t = yf.Ticker(symbol)

        # --- Fundamental filters (cheapest checks first) ---
        q = t.quarterly_income_stmt
        if q is None or q.empty or q.shape[1] < 5:
            return None
        rev = get_row(q, ["Total Revenue", "TotalRevenue"])
        ni = get_row(q, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
        if rev is None or ni is None:
            return None

        sales_growth = (rev.iloc[0] - rev.iloc[4]) / abs(rev.iloc[4]) * 100
        profit_growth = (ni.iloc[0] - ni.iloc[4]) / abs(ni.iloc[4]) * 100
        if sales_growth < 30 or profit_growth < 50:
            return None

        # --- Market cap filter ---
        mcap = t.info.get("marketCap")
        if not mcap or mcap >= MAX_MCAP_CR * CR:
            return None

        # --- Price + SuperTrend filters ---
        hist = t.history(period="6mo")
        hist = hist.dropna(subset=["Close", "High", "Low"])
        if len(hist) < 30:
            return None

        price = hist["Close"].iloc[-1]
        if price <= MIN_PRICE:
            return None

        price_yday = hist["Close"].iloc[-2]
        flipped = []
        for period, mult in ST_PARAMS:
            st_series = supertrend(hist, period, mult)
            st_today, st_yday = st_series.iloc[-1], st_series.iloc[-2]
            if pd.isna(st_today) or pd.isna(st_yday):
                return None
            if price <= st_today:
                return None  # not all green today -> fails List A (and B)
            if price_yday <= st_yday:
                flipped.append(f"{period},{mult}")

        # Reaching here = List A (ALIGNED). Non-empty 'flipped' = also List B (FRESH).
        return (symbol.replace(".NS", ""), round(price, 2), round(mcap / CR),
                round(sales_growth, 1), round(profit_growth, 1), ",".join(flipped))

    except Exception:
        return None


def format_section(rows, title):
    if not rows:
        return f"{title}: none today\n"
    lines = [f"{title}: {len(rows)} stock(s)",
             f"{'Symbol':<10}{'Price':>9}{'MCap(Cr)':>10}{'SalesG%':>9}{'ProfitG%':>10}  Flipped"]
    for r in rows:
        lines.append(f"{r[0]:<10}{r[1]:>9}{r[2]:>10}{r[3]:>9}{r[4]:>10}  {r[5] or '-'}")
    return "\n".join(lines) + "\n"


def send_email(aligned, fresh, scanned, elapsed_min):
    body = "Daily India stock scan results\n\n"
    body += format_section(fresh, "B) FRESH (>=1 SuperTrend flipped to green today)") + "\n"
    body += format_section(aligned, "A) ALIGNED (all 3 SuperTrends green today)") + "\n"
    body += f"\nScanned {scanned} stocks in {elapsed_min:.0f} min."

    msg = MIMEMultipart()
    msg["From"] = GMAIL_USER
    msg["To"] = TO_EMAIL
    msg["Subject"] = f"India Stock Scan: {len(fresh)} fresh, {len(aligned)} aligned"
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)


# ---- Main ----
start = time.time()
print("Loading NSE ticker list...")
try:
    tickers = get_nse_universe()
except Exception as e:
    print(f"FAILED to load NSE ticker list: {e}")
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_USER
        msg["To"] = TO_EMAIL
        msg["Subject"] = "India Stock Scan: FAILED to fetch ticker list"
        msg.attach(MIMEText(f"Could not fetch the NSE ticker list, so today's scan "
                             f"did not run.\n\nError:\n{e}", "plain"))
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print("Failure notification emailed.")
    except Exception as mail_err:
        print(f"Also failed to send failure email: {mail_err}")
    raise

print(f"Scanning {len(tickers)} stocks...\n")

aligned, fresh = [], []
for i, sym in enumerate(tickers, 1):
    result = check_stock(sym)
    if result:
        aligned.append(result)
        if result[5]:
            fresh.append(result)
        print(f"ALIGNED: {result[0]:<10} Price={result[1]:<8} MCap(Cr)={result[2]:<7} "
              f"SalesG%={result[3]:<7} ProfitG%={result[4]:<7} Flipped={result[5] or '-'}")
    if i % 250 == 0:
        print(f"  ...{i}/{len(tickers)} scanned")
    time.sleep(DELAY)

elapsed = (time.time() - start) / 60
print(f"\nALIGNED: {len(aligned)}   FRESH: {len(fresh)}")

cols = ["Symbol", "Price", "MCap_Cr", "SalesGrowth%", "ProfitGrowth%", "Flipped"]
if aligned:
    pd.DataFrame(aligned, columns=cols).to_csv("aligned.csv", index=False)
if fresh:
    pd.DataFrame(fresh, columns=cols).to_csv("fresh.csv", index=False)

print("Sending email...")
try:
    send_email(aligned, fresh, len(tickers), elapsed)
    print("Email sent.")
except Exception as e:
    print(f"Email failed: {e}")
    print("Check GMAIL_USER / GMAIL_APP_PASSWORD / TO_EMAIL at the top of this file.")
