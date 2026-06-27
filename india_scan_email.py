"""
india_scan_email.py  –  NSE Stock Scanner
==========================================
Replaced yfinance (rate-limited on GitHub Actions cloud IPs) with
NSE's own public APIs, which work reliably from cloud IPs when
accessed with a properly initialised browser-like session.

Data sources (all free, no API key required):
  • NSE bhavcopy CSV   – bulk price/volume data for quick pre-filter
  • NSE quote-equity   – current price + issued-share count → market cap
  • NSE historical CM  – daily OHLCV for SuperTrend / 200-DEMA
  • NSE corporate-results – quarterly P&L for sales/profit growth

Architecture: 3-phase to minimise API calls
  Phase 1 (1 download)   : bhavcopy pre-filter → price > ₹100, rough volume
  Phase 2 (~500-800 hits): historical data → volume, DEMA, SuperTrend filters
  Phase 3 (~10-100 hits) : quote (mcap) + financials for final survivors only

Required env vars (same as before):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, TO_EMAIL (opt, defaults below)
"""

from __future__ import annotations

import io
import logging
import os
import smtplib
import time
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nse_scan")

# ── Constants ──────────────────────────────────────────────────────────────────
IST = ZoneInfo("Asia/Kolkata")

PRICE_MIN        = 100          # ₹
MCAP_MIN_CR      = 1_000        # ₹ Crore
MCAP_MAX_CR      = 20_000       # ₹ Crore
AVG_VOL_MIN      = 200_000      # 30-day avg traded quantity
SALES_GROWTH_MIN = 30.0         # % YoY same-quarter
PROFIT_GROWTH_MIN= 50.0         # % YoY same-quarter
LOOKBACK_DAYS    = 320          # calendar days fetched for history (≈220 trading days)

# SuperTrend parameter sets  (period, multiplier)
ST_PARAMS = [(13, 4), (14, 5), (15, 6)]

# Concurrency: keep low; NSE is tolerant but not unlimited
MAX_WORKERS      = 3
INTER_REQ_DELAY  = 0.35         # seconds between successive calls in a thread

TO_EMAIL_DEFAULT = "tomailsasidhar@gmail.com"

# ── NSE Session ────────────────────────────────────────────────────────────────

class NSESession:
    """
    Thin wrapper around requests.Session that handles NSE's cookie/header
    requirements.  Each thread should create its own instance (not shared).
    """

    BASE     = "https://www.nseindia.com"
    ARCHIVES = "https://nsearchives.nseindia.com"

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
        "Referer":         "https://www.nseindia.com/",
    }

    def __init__(self):
        self._s = requests.Session()
        self._s.headers.update(self._HEADERS)
        self._warm_up()

    def _warm_up(self) -> None:
        """Hit NSE home + market-data page to populate session cookies."""
        try:
            self._s.get(self.BASE + "/", timeout=12)
            time.sleep(0.8)
            self._s.get(self.BASE + "/market-data/live-equity-market", timeout=12)
            time.sleep(0.5)
        except Exception as exc:
            log.debug("NSE warm-up warning: %s", exc)

    def _get(self, url: str, params: dict | None = None, retries: int = 3) -> dict | bytes | None:
        for attempt in range(retries):
            try:
                r = self._s.get(url, params=params, timeout=20)
                if r.status_code == 200:
                    ct = r.headers.get("Content-Type", "")
                    if "json" in ct:
                        return r.json()
                    return r.content          # zip / csv bytes
                if r.status_code in (401, 403):
                    log.debug("Session expired (%s) – re-warming", r.status_code)
                    self._warm_up()
                    time.sleep(2)
                else:
                    log.debug("HTTP %s for %s", r.status_code, url)
            except requests.Timeout:
                log.debug("Timeout %s (attempt %d)", url, attempt + 1)
            except Exception as exc:
                log.debug("Error %s: %s", url, exc)
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
        return None

    # ── Public API wrappers ──────────────────────────────────────────────────

    def get_bhavcopy(self, trade_date: date) -> bytes | None:
        """Return raw bytes of bhavcopy for trade_date (zip or csv)."""
        ymd  = trade_date.strftime("%Y%m%d")
        dmy  = trade_date.strftime("%d%b%Y").upper()
        y    = trade_date.strftime("%Y")
        mon  = trade_date.strftime("%b").upper()

        candidates = [
            # New format (2024+)
            f"{self.ARCHIVES}/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip",
            # Alternate new format
            f"{self.ARCHIVES}/products/content/sec_bhavdata_full_{dmy}.csv",
            # Old format
            f"{self.ARCHIVES}/content/historical/EQUITIES/{y}/{mon}/cm{dmy}bhav.csv.zip",
        ]
        for url in candidates:
            raw = self._get(url)
            if raw:
                return raw
        return None

    def get_equity_list(self) -> bytes | None:
        """EQUITY_L.csv – all listed NSE equities with series."""
        return self._get(f"{self.ARCHIVES}/content/equities/EQUITY_L.csv")

    def get_quote(self, symbol: str) -> dict | None:
        """Current quote: price, issued-size, trade info."""
        return self._get(f"{self.BASE}/api/quote-equity", {"symbol": symbol})

    def get_history(self, symbol: str, from_dt: date, to_dt: date) -> dict | None:
        """Historical daily OHLCV."""
        return self._get(
            f"{self.BASE}/api/historical/cm/equity",
            {
                "symbol": symbol,
                "series": '["EQ"]',
                "from":   from_dt.strftime("%d-%m-%Y"),
                "to":     to_dt.strftime("%d-%m-%Y"),
            },
        )

    def get_financials(self, symbol: str) -> dict | None:
        """Quarterly corporate financial results."""
        return self._get(
            f"{self.BASE}/api/corporates-financial-results",
            {"index": "equities", "symbol": symbol},
        )


# ── Thread-local sessions (one per worker thread) ─────────────────────────────
_tls = threading.local()

def _session() -> NSESession:
    if not hasattr(_tls, "nse"):
        _tls.nse = NSESession()
    return _tls.nse


# ── Phase 1: Symbol list + bhavcopy quick-filter ──────────────────────────────

def _parse_bhavcopy(raw: bytes) -> pd.DataFrame | None:
    """Parse bhavcopy bytes (zip or plain CSV) into a normalised DataFrame."""
    try:
        # Try zip
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = z.namelist()[0]
            df = pd.read_csv(z.open(name))
    except zipfile.BadZipFile:
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            return None
    except Exception:
        return None

    df.columns = df.columns.str.strip().str.upper()

    # Normalise known column-name variants
    renames = {}
    for col in df.columns:
        c = col.strip().upper()
        # SYMBOL
        if c in ("SYMBOL", "TCKRSYMB", "SCRIP_CD"):
            renames[col] = "SYMBOL"
        elif "SYMBOL" in c and "SERIES" not in c and c not in renames.values():
            renames[col] = "SYMBOL"
        # SERIES
        elif c in ("SERIES", "SCTYSRS", "SCTYSER"):
            renames[col] = "SERIES"
        # CLOSE price (exclude PREVCLOSE / previous-close variants)
        elif c in ("CLOSE", "CLSPRIC", "CLOSE_PRICE", "LASTPRIC") or (
            "CLOSE" in c and "PREV" not in c and "PRV" not in c
        ):
            renames[col] = "CLOSE"
        # VOLUME
        elif c in ("TOTTRDQTY", "VOLUME", "TOTALTRADEDQTY", "TTTRADGVOL", "TTLTRADGVOL") or (
            "TRADG" in c and "VOL" in c
        ) or (
            "TOT" in c and "QTY" in c and "VAL" not in c
        ):
            renames[col] = "VOLUME"
    df.rename(columns=renames, inplace=True)

    # Drop duplicate column names (e.g. two cols both mapped to "CLOSE")
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    needed = {"SYMBOL", "CLOSE"}
    if not needed.issubset(df.columns):
        return None

    # Keep EQ series only (if column present)
    if "SERIES" in df.columns:
        df = df[df["SERIES"].str.strip() == "EQ"].copy()

    df["CLOSE"]  = pd.to_numeric(df["CLOSE"],  errors="coerce")
    if "VOLUME" in df.columns:
        df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")

    return df.dropna(subset=["SYMBOL", "CLOSE"])


def get_pre_filtered_symbols(today: date) -> list[str]:
    """
    Returns a symbol list pre-filtered by price > ₹100 (and rough volume).
    Falls back to the full EQUITY_L.csv list if bhavcopy is unavailable.
    """
    session = NSESession()  # dedicated session for Phase 1

    # Try bhavcopy for today (may be unavailable before ~18:30 IST)
    # Walk back up to 3 business days
    for delta in range(4):
        d = today - timedelta(days=delta)
        if d.weekday() >= 5:   # skip weekends
            continue
        raw = session.get_bhavcopy(d)
        if raw:
            df = _parse_bhavcopy(raw)
            if df is not None and len(df) > 100:
                log.info("Bhavcopy loaded for %s  (%d rows)", d, len(df))
                # Price filter
                df = df[df["CLOSE"] >= PRICE_MIN]
                # Rough volume pre-filter (today's vol ≥ 30% of 30d threshold)
                if "VOLUME" in df.columns:
                    df = df[df["VOLUME"] >= AVG_VOL_MIN * 0.30]
                syms = df["SYMBOL"].str.strip().tolist()
                log.info("Pre-filter via bhavcopy: %d symbols remain", len(syms))
                return syms

    # Fallback: full equity list from EQUITY_L.csv
    log.warning("Bhavcopy unavailable – falling back to EQUITY_L.csv (full scan)")
    raw = session.get_equity_list()
    if raw:
        try:
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = df.columns.str.strip()
            sym_col = next((c for c in df.columns if "SYMBOL" in c.upper()), df.columns[0])
            ser_col = next((c for c in df.columns if "SERIES" in c.upper()), None)
            if ser_col:
                df = df[df[ser_col].str.strip() == "EQ"]
            return df[sym_col].str.strip().tolist()
        except Exception as exc:
            log.error("Could not parse EQUITY_L.csv: %s", exc)
    return []


# ── Indicators ────────────────────────────────────────────────────────────────

def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed ATR (matches TradingView / most charting platforms)."""
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.roll(close, 1)),
        np.abs(low  - np.roll(close, 1)),
    ])
    tr[0] = high[0] - low[0]

    atr = np.full(len(tr), np.nan)
    if len(tr) < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_supertrend(df: pd.DataFrame, period: int, multiplier: float) -> np.ndarray:
    """
    SuperTrend.  Returns direction array: +1 = bullish (green), -1 = bearish (red).
    Implements the standard algorithm used in TradingView's built-in SuperTrend.
    """
    n = len(df)
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    atr = _wilder_atr(high, low, close, period)
    hl2 = (high + low) / 2.0

    bu = hl2 + multiplier * atr   # basic upper band
    bl = hl2 - multiplier * atr   # basic lower band

    fu = bu.copy()   # final upper band
    fl = bl.copy()   # final lower band
    direction = np.ones(n, dtype=int)

    for i in range(1, n):
        # Ratchet upper band down / lower band up
        fu[i] = bu[i] if (bu[i] < fu[i-1] or close[i-1] > fu[i-1]) else fu[i-1]
        fl[i] = bl[i] if (bl[i] > fl[i-1] or close[i-1] < fl[i-1]) else fl[i-1]

        if direction[i-1] == -1:
            direction[i] = 1  if close[i] > fu[i] else -1
        else:
            direction[i] = -1 if close[i] < fl[i] else  1

    return direction


def calc_dema(series: pd.Series, period: int) -> pd.Series:
    """Double Exponential Moving Average."""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    return 2 * ema1 - ema2


# ── History parser ────────────────────────────────────────────────────────────

_HIST_FIELD_MAP = {
    "date":   ["CH_TIMESTAMP",        "mTIMESTAMP",  "TIMESTAMP"],
    "open":   ["CH_OPENING_PRICE",    "mOPEN",       "OPEN"],
    "high":   ["CH_TRADE_HIGH_PRICE", "mHIGH",       "HIGH"],
    "low":    ["CH_TRADE_LOW_PRICE",  "mLOW",        "LOW"],
    "close":  ["CH_CLOSING_PRICE",    "mCLOSE",      "CLOSE"],
    "volume": ["CH_TOT_TRADED_QTY",   "mQTY",        "TOTTRDQTY"],
}

def _pick(d: dict, candidates: list[str]):
    for k in candidates:
        if k in d:
            return d[k]
    return None


def parse_history(raw: dict | None) -> pd.DataFrame | None:
    if not raw or "data" not in raw or not raw["data"]:
        return None

    rows = []
    for d in raw["data"]:
        try:
            rows.append({
                "date":   pd.to_datetime(_pick(d, _HIST_FIELD_MAP["date"])),
                "open":   float(_pick(d, _HIST_FIELD_MAP["open"])  or 0),
                "high":   float(_pick(d, _HIST_FIELD_MAP["high"])  or 0),
                "low":    float(_pick(d, _HIST_FIELD_MAP["low"])   or 0),
                "close":  float(_pick(d, _HIST_FIELD_MAP["close"]) or 0),
                "volume": int(  _pick(d, _HIST_FIELD_MAP["volume"])or 0),
            })
        except (TypeError, ValueError):
            continue

    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df[df["close"] > 0]


# ── Financials parser ─────────────────────────────────────────────────────────

def _to_period_dt(period_str: str) -> pd.Timestamp | None:
    """Parse NSE period strings like 'Dec 2024', 'Q3 FY25', etc."""
    for fmt in ("%b %Y", "%B %Y"):
        try:
            return pd.to_datetime(period_str.strip(), format=fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(period_str.strip())
    except Exception:
        return None


def _extract_num(record: dict, *keys: str) -> float | None:
    for k in keys:
        v = record.get(k)
        if v is not None:
            try:
                f = float(v)
                return f if not np.isnan(f) else None
            except (ValueError, TypeError):
                pass
    return None


def parse_financials(raw: dict | None) -> dict | None:
    """
    Parse NSE corporate-results response.
    Returns { sales_growth, profit_growth, quarter } or None.
    """
    if raw is None:
        return None

    records = raw if isinstance(raw, list) else raw.get("data", [])
    if not records:
        return None

    # Prefer Consolidated; fall back to Standalone
    for result_type in ("Consolidated", "Standalone", None):
        if result_type:
            subset = [r for r in records
                      if result_type.lower() in str(r.get("resultType", "")).lower()
                      or result_type.lower() in str(r.get("consolidated", "")).lower()]
        else:
            subset = records

        # Keep quarterly only (filter out annual / half-yearly)
        quarterly = [
            r for r in subset
            if "quarterly" in str(r.get("reInd", "quarterly")).lower()
            or "quarter"   in str(r.get("period", "")).lower()
            # Catch "Dec 2024" style (month + year = quarterly)
            or (_to_period_dt(str(r.get("period", ""))) is not None
                and len(str(r.get("period", "")).split()) == 2)
        ]

        if len(quarterly) < 5:
            quarterly = subset       # Accept whatever we have if filtering was too strict

        # Sort descending by period date
        dated = []
        for r in quarterly:
            dt = _to_period_dt(str(r.get("period", "")))
            if dt:
                dated.append((dt, r))

        if not dated:
            continue

        dated.sort(key=lambda x: x[0], reverse=True)

        curr_dt, curr = dated[0]

        # Match same quarter one year prior (within ±45 days)
        target = curr_dt.replace(year=curr_dt.year - 1)
        prev = None
        for dt, r in dated[1:]:
            if abs((dt - target).days) <= 45:
                prev = r
                break

        if prev is None:
            continue

        curr_sales  = _extract_num(curr, "income", "netSales", "totalIncome", "revenue", "sales")
        prev_sales  = _extract_num(prev, "income", "netSales", "totalIncome", "revenue", "sales")
        curr_profit = _extract_num(curr, "netProfit", "profit", "pat", "profitAfterTax")
        prev_profit = _extract_num(prev, "netProfit", "profit", "pat", "profitAfterTax")

        if None in (curr_sales, prev_sales, curr_profit, prev_profit):
            continue
        if prev_sales <= 0 or prev_profit <= 0:
            continue

        return {
            "sales_growth":  round((curr_sales  - prev_sales)  / abs(prev_sales)  * 100, 1),
            "profit_growth": round((curr_profit - prev_profit) / abs(prev_profit) * 100, 1),
            "quarter":       str(curr.get("period", "")),
        }

    return None


# ── Market cap ────────────────────────────────────────────────────────────────

def market_cap_cr(quote: dict | None) -> float | None:
    """Derive market cap (₹ Crore) from NSE quote-equity response."""
    if not quote:
        return None
    try:
        price = float(quote.get("priceInfo", {}).get("lastPrice", 0))
        issued = float(quote.get("securityInfo", {}).get("issuedSize", 0))
        if price > 0 and issued > 0:
            return (price * issued) / 1e7   # → Crore
    except (TypeError, ValueError):
        pass
    return None


# ── Phase 2: Technical check (per stock) ─────────────────────────────────────

def check_technicals(symbol: str, today: date, lookback_from: date) -> dict | None:
    """
    Fetch historical data and evaluate:
      • 30-day avg volume > AVG_VOL_MIN
      • Close > 200 DEMA
      • All 3 SuperTrends green today

    Returns partial result dict (no fundamentals yet) or None if filtered out.
    """
    nse = _session()
    time.sleep(INTER_REQ_DELAY)

    raw_hist = nse.get_history(symbol, lookback_from, today)
    hist = parse_history(raw_hist)

    if hist is None or len(hist) < 60:
        return None

    close  = hist["close"]
    volume = hist["volume"]

    # 30-day avg volume
    avg_vol = volume.tail(30).mean()
    if avg_vol < AVG_VOL_MIN:
        return None

    # Need 200+ bars for DEMA
    if len(hist) < 200:
        return None

    last_close = close.iloc[-1]

    # 200 DEMA
    dema200 = calc_dema(close, 200).iloc[-1]
    if last_close <= dema200:
        return None

    # SuperTrend – all 3 must be green today; track prev-day for FRESH
    today_dirs = []
    prev_dirs  = []
    for period, mult in ST_PARAMS:
        d = calc_supertrend(hist, period, mult)
        today_dirs.append(d[-1])
        prev_dirs.append( d[-2])

    if not all(d == 1 for d in today_dirs):
        return None

    is_fresh = any(c == 1 and p == -1 for c, p in zip(today_dirs, prev_dirs))

    return {
        "symbol":    symbol,
        "close":     round(last_close, 2),
        "avg_vol":   int(avg_vol),
        "dema200":   round(dema200, 2),
        "is_fresh":  is_fresh,
    }


# ── Phase 3: Fundamental + MCap check (per surviving stock) ──────────────────

def check_fundamentals(tech: dict) -> dict | None:
    """
    For a stock that passed technicals, fetch:
      • Market cap (from quote-equity)
      • Quarterly financials (sales growth / profit growth)

    Returns full result dict or None if filtered out.
    """
    sym = tech["symbol"]
    nse = _session()

    # Market cap
    time.sleep(INTER_REQ_DELAY)
    quote = nse.get_quote(sym)
    mcap = market_cap_cr(quote)

    if mcap is None or not (MCAP_MIN_CR <= mcap <= MCAP_MAX_CR):
        return None

    # Price sanity check from quote (in case bhavcopy day lagged)
    try:
        live_price = float(quote.get("priceInfo", {}).get("lastPrice", tech["close"]))
        if live_price < PRICE_MIN:
            return None
    except (TypeError, ValueError):
        pass

    # Financials
    time.sleep(INTER_REQ_DELAY)
    fin_raw = nse.get_financials(sym)
    fin = parse_financials(fin_raw)

    if fin is None:
        return None
    if fin["sales_growth"]  < SALES_GROWTH_MIN:
        return None
    if fin["profit_growth"] < PROFIT_GROWTH_MIN:
        return None

    return {
        **tech,
        "mcap_cr":       round(mcap, 0),
        "sales_growth":  fin["sales_growth"],
        "profit_growth": fin["profit_growth"],
        "quarter":       fin["quarter"],
    }


# ── Email ──────────────────────────────────────────────────────────────────────

_TABLE_HEADER = (
    "<tr style='background:#1a1a2e;color:#e0e0e0;'>"
    "<th>Symbol</th><th>Price ₹</th><th>MCap Cr</th>"
    "<th>Avg Vol 30d</th><th>Sales Growth</th>"
    "<th>Profit Growth</th><th>Quarter</th>"
    "</tr>"
)

def _row(r: dict, fresh: bool = False) -> str:
    bg = "#fff8e1" if fresh else "#ffffff"
    badge = " 🆕" if fresh else ""
    return (
        f"<tr style='background:{bg};'>"
        f"<td><b>{r['symbol']}{badge}</b></td>"
        f"<td>₹{r['close']:.2f}</td>"
        f"<td>₹{r['mcap_cr']:.0f}</td>"
        f"<td>{r['avg_vol']:,}</td>"
        f"<td style='color:{'green' if r['sales_growth']>0 else 'red'};'>"
        f"{r['sales_growth']:+.1f}%</td>"
        f"<td style='color:{'green' if r['profit_growth']>0 else 'red'};'>"
        f"{r['profit_growth']:+.1f}%</td>"
        f"<td>{r['quarter']}</td>"
        f"</tr>"
    )


def send_email(
    aligned: list[dict],
    fresh:   list[dict],
    total_scanned: int,
    to_addr: str,
) -> None:
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = "tomailsasidhar@gmail.com"     # <-- Your Gmail address
    smtp_pass = os.environ["GMAIL_APP_PASSWORD"]
    from_addr = smtp_user

    now_ist    = datetime.now(IST)
    today_str  = now_ist.strftime("%d %b %Y")
    fresh_syms = {r["symbol"] for r in fresh}

    subject = (
        f"NSE Scan {today_str} | "
        f"ALIGNED: {len(aligned)} | FRESH: {len(fresh)}"
    )

    def make_table(rows: list[dict], mark_fresh: bool = False) -> str:
        if not rows:
            return "<p><i>None</i></p>"
        html_rows = "".join(
            _row(r, fresh=mark_fresh and r["symbol"] in fresh_syms)
            for r in sorted(rows, key=lambda x: x["symbol"])
        )
        return (
            "<table border='1' cellpadding='6' cellspacing='0' "
            "style='border-collapse:collapse;font-size:13px;'>"
            f"{_TABLE_HEADER}{html_rows}</table>"
        )

    body = f"""
<html><body style="font-family:Arial,sans-serif;color:#222;">
<h2 style="color:#1a1a2e;">📊 NSE Stock Scanner – {today_str}</h2>

<p style="font-size:12px;color:#555;">
  Filters: Price &gt; ₹{PRICE_MIN} &nbsp;|&nbsp;
  MCap ₹{MCAP_MIN_CR:,}–{MCAP_MAX_CR:,} Cr &nbsp;|&nbsp;
  Avg Vol(30d) &gt; {AVG_VOL_MIN:,} &nbsp;|&nbsp;
  Sales Growth &gt; {SALES_GROWTH_MIN}% YoY &nbsp;|&nbsp;
  Profit Growth &gt; {PROFIT_GROWTH_MIN}% YoY &nbsp;|&nbsp;
  All 3 SuperTrends (13,4)/(14,5)/(15,6) Green &nbsp;|&nbsp;
  Close &gt; 200 DEMA
</p>

<h3>✅ ALIGNED — {len(aligned)} stock{'s' if len(aligned)!=1 else ''}</h3>
<p style="font-size:12px;"><i>All conditions met. 🆕 = also FRESH today.</i></p>
{make_table(aligned, mark_fresh=True)}

<h3>🔄 FRESH — {len(fresh)} stock{'s' if len(fresh)!=1 else ''}</h3>
<p style="font-size:12px;"><i>All conditions met + ≥1 SuperTrend flipped red→green today.</i></p>
{make_table(fresh)}

<hr style="margin-top:30px;">
<p style="font-size:11px;color:#888;">
  Scanned {total_scanned} NSE symbols &nbsp;·&nbsp;
  Data: NSE Official API (no yfinance) &nbsp;·&nbsp;
  Generated {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}
</p>
</body></html>
"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as srv:
        srv.ehlo()
        srv.starttls()
        srv.login(smtp_user, smtp_pass)
        srv.sendmail(from_addr, [to_addr], msg.as_string())

    log.info("Email sent → %s  |  Subject: %s", to_addr, subject)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    to_email = os.environ.get("TO_EMAIL", TO_EMAIL_DEFAULT)
    now_ist  = datetime.now(IST)
    today    = now_ist.date()

    # Safety: should be handled by cron, but guard anyway
    if today.weekday() >= 5:
        log.info("Weekend (%s) – skipping scan", today.strftime("%A"))
        return

    log.info("=" * 60)
    log.info("NSE Scan  %s  (IST %s)", today, now_ist.strftime("%H:%M"))
    log.info("=" * 60)

    # ── Phase 1: symbol pre-filter ────────────────────────────────────────────
    log.info("Phase 1: downloading bhavcopy for price/volume pre-filter …")
    symbols = get_pre_filtered_symbols(today)

    if not symbols:
        log.error("No symbols found – aborting.")
        return

    total_scanned = len(symbols)
    lookback_from = today - timedelta(days=LOOKBACK_DAYS)
    log.info("Phase 1 done: %d symbols to check technically", total_scanned)

    # ── Phase 2: technical analysis (concurrent) ──────────────────────────────
    log.info("Phase 2: technical analysis with %d workers …", MAX_WORKERS)
    tech_passed: list[dict] = []
    tech_errors = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(check_technicals, sym, today, lookback_from): sym
            for sym in symbols
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            sym = futs[fut]
            try:
                result = fut.result()
                if result:
                    tech_passed.append(result)
            except Exception as exc:
                tech_errors += 1
                log.debug("Tech error %s: %s", sym, exc)
            if done % 100 == 0 or done == total_scanned:
                log.info(
                    "  %d/%d  |  tech-passed: %d  |  errors: %d",
                    done, total_scanned, len(tech_passed), tech_errors,
                )

    log.info(
        "Phase 2 done: %d/%d pass technicals (%d errors)",
        len(tech_passed), total_scanned, tech_errors,
    )

    # ── Phase 3: fundamentals + mcap (sequential, far fewer stocks) ───────────
    log.info("Phase 3: fundamentals & market-cap for %d candidates …", len(tech_passed))
    aligned: list[dict] = []
    fresh:   list[dict] = []

    # Use a single fresh session for Phase 3 (sequential = no thread contention)
    _tls.nse = NSESession()

    for i, tech in enumerate(tech_passed, 1):
        sym = tech["symbol"]
        try:
            result = check_fundamentals(tech)
            if result:
                aligned.append(result)
                if result["is_fresh"]:
                    fresh.append(result)
                log.info(
                    "  ✅ %-12s  Sales %+.1f%%  Profit %+.1f%%  MCap ₹%.0fCr%s",
                    sym,
                    result["sales_growth"],
                    result["profit_growth"],
                    result["mcap_cr"],
                    "  🆕FRESH" if result["is_fresh"] else "",
                )
        except Exception as exc:
            log.debug("Fundamental error %s: %s", sym, exc)

        if i % 10 == 0:
            log.info("  %d/%d candidates processed", i, len(tech_passed))

    log.info(
        "Phase 3 done: ALIGNED=%d  FRESH=%d  (scanned %d total symbols)",
        len(aligned), len(fresh), total_scanned,
    )

    # ── Send email ────────────────────────────────────────────────────────────
    send_email(aligned, fresh, total_scanned, to_email)
    log.info("Done.")


if __name__ == "__main__":
    main()
