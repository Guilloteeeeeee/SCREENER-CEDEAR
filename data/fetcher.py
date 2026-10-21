"""
Módulo de descarga de datos desde Yahoo Finance (yfinance).
Obtiene métricas actuales e históricas para calcular medianas.

Estrategia de obtención de medianas (en orden de prioridad):
  1. Calcular directamente desde estados financieros históricos (tk.financials, etc.)
  2. Si falla, usar el valor actual como mediana (ratio = 1.0, score neutro = 50)

Este fallback garantiza que empresas con datos actuales válidos NUNCA queden
en DATOS INSUFICIENTES por falta de histórico.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
import logging
import time

logger = logging.getLogger(__name__)

# Años de histórico para calcular medianas
HISTORY_YEARS = 5

# Pausa entre requests para no saturar la API
REQUEST_DELAY = 0.3  # segundos


def _safe_float(val, min_val=None, max_val=None) -> Optional[float]:
    """Convierte a float de forma segura, con filtros de rango opcionales."""
    try:
        if val is None:
            return None
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        if min_val is not None and f < min_val:
            return None
        if max_val is not None and f > max_val:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _find_row(df: pd.DataFrame, *keywords, exclude=None) -> Optional[str]:
    """
    Busca la primera fila del DataFrame cuyo índice contenga TODOS los keywords.
    Si exclude es una lista, descarta filas que contengan alguna de esas palabras.
    """
    if df is None or df.empty:
        return None
    exclude = exclude or []
    for idx in df.index:
        idx_str = str(idx).lower()
        if all(kw.lower() in idx_str for kw in keywords):
            if not any(ex.lower() in idx_str for ex in exclude):
                return idx
    return None


def _get_col_value(df: pd.DataFrame, row_key: str, col) -> Optional[float]:
    """Obtiene un valor del DataFrame de forma segura."""
    try:
        if row_key is None or df is None or df.empty:
            return None
        if row_key not in df.index:
            return None
        if col not in df.columns:
            return None
        return _safe_float(df.loc[row_key, col])
    except Exception:
        return None


def _get_annual_price(hist: pd.DataFrame, year: int) -> Optional[float]:
    """Precio de cierre al final de un año dado."""
    try:
        year_data = hist[hist.index.year == year]
        if not year_data.empty:
            return _safe_float(year_data["Close"].iloc[-1])
    except Exception:
        pass
    return None


def fetch_ticker_data(ticker: str) -> dict:
    """
    Descarga todos los datos necesarios para un ticker.
    Retorna un dict con métricas actuales e históricas.
    """
    result = {
        "ticker": ticker,
        "nombre": None,
        "pais": None,
        "sector_yf": None,
        "market_cap": None,
        "precio_actual": None,
        "precio_max_52s": None,
        "precio_min_52s": None,
        "posicion_52s": None,
        # Valores actuales
        "ev_ebitda_actual": None,
        "per_actual": None,
        "p_fcf_actual": None,
        "p_b_actual": None,
        "dividend_yield_actual": None,
        "peg_actual": None,
        "p_ffo_actual": None,
        "p_affo_actual": None,
        # Medianas históricas
        "ev_ebitda_mediana": None,
        "per_mediana": None,
        "p_fcf_mediana": None,
        "p_b_mediana": None,
        "dividend_yield_mediana": None,
        "peg_mediana": None,
        "p_ffo_mediana": None,
        "p_affo_mediana": None,
        # Metadatos
        "error": None,
        "metricas_disponibles": 0,
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        precio = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if precio is None:
            result["error"] = "Sin datos en Yahoo Finance"
            return result

        # --- Datos básicos ---
        result["nombre"] = info.get("shortName") or info.get("longName") or ticker
        result["pais"] = info.get("country")
        result["sector_yf"] = info.get("sector")
        result["market_cap"] = _safe_float(info.get("marketCap"))
        result["precio_actual"] = precio
        result["precio_max_52s"] = _safe_float(info.get("fiftyTwoWeekHigh"))
        result["precio_min_52s"] = _safe_float(info.get("fiftyTwoWeekLow"))

        # --- Posición en rango 52 semanas ---
        p = result["precio_actual"]
        hi = result["precio_max_52s"]
        lo = result["precio_min_52s"]
        if p and hi and lo and hi != lo:
            result["posicion_52s"] = round((p - lo) / (hi - lo), 4)

        # --- Métricas actuales desde info (múltiples campos alternativos) ---

        # EV/EBITDA
        result["ev_ebitda_actual"] = _safe_float(info.get("enterpriseToEbitda"), min_val=0, max_val=500)

        # PER: trailing primero, luego forward
        result["per_actual"] = (
            _safe_float(info.get("trailingPE"), min_val=0, max_val=2000) or
            _safe_float(info.get("forwardPE"), min_val=0, max_val=2000)
        )

        # P/B
        result["p_b_actual"] = _safe_float(info.get("priceToBook"), min_val=0, max_val=200)

        # Dividend Yield
        result["dividend_yield_actual"] = _safe_float(info.get("dividendYield"), min_val=0, max_val=0.5)

        # PEG
        result["peg_actual"] = _safe_float(info.get("pegRatio"), min_val=0, max_val=100)

        # P/FCF desde info
        fcf = _safe_float(info.get("freeCashflow"))
        mktcap = result["market_cap"]
        if fcf and fcf > 0 and mktcap and mktcap > 0:
            result["p_fcf_actual"] = round(mktcap / fcf, 2)

        # Si P/FCF no está en info, intentar calcularlo de otra forma
        if result["p_fcf_actual"] is None:
            pfcf_info = _safe_float(info.get("priceToFreeCashflows"), min_val=0, max_val=2000)
            result["p_fcf_actual"] = pfcf_info

        # P/FFO y P/AFFO para REITs (proxy via P/E)
        result["p_ffo_actual"] = result["per_actual"]
        result["p_affo_actual"] = result["per_actual"]

        # --- Histórico anual para medianas ---
        time.sleep(REQUEST_DELAY)
        _enrich_with_history(tk, result, info)

    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Error fetching {ticker}: {e}")

    # Contar métricas disponibles (con valor actual Y mediana)
    metric_keys = [
        "ev_ebitda_actual", "per_actual", "p_fcf_actual", "p_b_actual",
        "dividend_yield_actual", "peg_actual", "p_ffo_actual", "p_affo_actual"
    ]
    result["metricas_disponibles"] = sum(
        1 for k in metric_keys if result[k] is not None
    )

    return result


def _enrich_with_history(tk: yf.Ticker, result: dict, info: dict) -> None:
    """
    Agrega medianas históricas al resultado.

    Estrategia:
    1. Intenta calcular medianas desde estados financieros históricos.
    2. Si falla (pocos datos, yfinance no devuelve histórico), usa el valor actual
       como mediana — así el scoring arroja ~50 (bien valuada) en lugar de DATOS INSUFICIENTES.
    """
    try:
        income = tk.financials      # income statement anual
        balance = tk.balance_sheet  # balance sheet anual
        cashflow = tk.cashflow      # cash flow anual
        hist = tk.history(period=f"{HISTORY_YEARS}y", interval="1mo")

        shares = _safe_float(info.get("sharesOutstanding")) or _safe_float(info.get("impliedSharesOutstanding"))

        # --- EV/EBITDA histórico ---
        ev_ebitda_list = []
        if not (income is None or income.empty) and shares and not hist.empty:
            ev_ebitda_list = _calc_ev_ebitda_history(income, balance, hist, shares)
        if ev_ebitda_list:
            result["ev_ebitda_mediana"] = round(float(np.median(ev_ebitda_list)), 2)
        elif result["ev_ebitda_actual"] is not None:
            result["ev_ebitda_mediana"] = result["ev_ebitda_actual"]  # fallback

        # --- PER histórico ---
        per_list = []
        if not (income is None or income.empty) and not hist.empty:
            per_list = _calc_per_history(income, balance, hist, shares, info)
        if per_list:
            result["per_mediana"] = round(float(np.median(per_list)), 2)
        elif result["per_actual"] is not None:
            result["per_mediana"] = result["per_actual"]  # fallback

        # --- P/FCF histórico ---
        pfcf_list = []
        if not (cashflow is None or cashflow.empty) and not (balance is None or balance.empty) and not hist.empty:
            pfcf_list = _calc_pfcf_history(cashflow, balance, hist, shares, info)
        if pfcf_list:
            result["p_fcf_mediana"] = round(float(np.median(pfcf_list)), 2)
        elif result["p_fcf_actual"] is not None:
            result["p_fcf_mediana"] = result["p_fcf_actual"]  # fallback

        # --- P/B histórico ---
        pb_list = []
        if not (balance is None or balance.empty) and not hist.empty:
            pb_list = _calc_pb_history(balance, hist, shares, info)
        if pb_list:
            result["p_b_mediana"] = round(float(np.median(pb_list)), 2)
        elif result["p_b_actual"] is not None:
            result["p_b_mediana"] = result["p_b_actual"]  # fallback

        # --- Dividend Yield histórico ---
        dy_list = []
        if not hist.empty:
            dy_list = _calc_divy_history(tk, hist)
        if dy_list:
            result["dividend_yield_mediana"] = round(float(np.median(dy_list)), 4)
        elif result["dividend_yield_actual"] is not None:
            result["dividend_yield_mediana"] = result["dividend_yield_actual"]  # fallback

        # PEG: usar actual como mediana
        result["peg_mediana"] = result["peg_actual"]

        # FFO/AFFO: proxy via PER
        result["p_ffo_mediana"] = result["per_mediana"]
        result["p_affo_mediana"] = result["per_mediana"]

    except Exception as e:
        logger.debug(f"Error en histórico para {result['ticker']}: {e}")
        # Aunque falle todo, aplicar fallback: mediana = actual
        _apply_fallback_medianas(result)


def _apply_fallback_medianas(result: dict) -> None:
    """Si la mediana es None pero el actual existe, usa el actual como mediana."""
    metric_pairs = [
        ("ev_ebitda_actual", "ev_ebitda_mediana"),
        ("per_actual", "per_mediana"),
        ("p_fcf_actual", "p_fcf_mediana"),
        ("p_b_actual", "p_b_mediana"),
        ("dividend_yield_actual", "dividend_yield_mediana"),
        ("peg_actual", "peg_mediana"),
        ("p_ffo_actual", "p_ffo_mediana"),
        ("p_affo_actual", "p_affo_mediana"),
    ]
    for actual_key, mediana_key in metric_pairs:
        if result.get(mediana_key) is None and result.get(actual_key) is not None:
            result[mediana_key] = result[actual_key]


def _calc_ev_ebitda_history(income, balance, hist, shares: float) -> list:
    """Calcula EV/EBITDA para cada año disponible."""
    values = []
    try:
        # Buscar EBITDA directo o calcular desde EBIT + D&A
        ebitda_row = _find_row(income, "ebitda")
        ebit_row = _find_row(income, "ebit", exclude=["ebitda"])
        da_row = _find_row(income, "depreciation") or _find_row(income, "amortization")

        # Buscar deuda y caja en balance
        debt_row = (
            _find_row(balance, "total debt") or
            _find_row(balance, "long term debt") or
            _find_row(balance, "totaldebt")
        )
        cash_row = (
            _find_row(balance, "cash and cash equivalents") or
            _find_row(balance, "cash equivalents") or
            _find_row(balance, "cash", exclude=["restricted"])
        )

        for col in income.columns[:HISTORY_YEARS]:
            year = col.year if hasattr(col, 'year') else pd.Timestamp(col).year
            price = _get_annual_price(hist, year)
            if not price:
                continue

            ebitda = _get_col_value(income, ebitda_row, col)
            if ebitda is None and ebit_row and da_row:
                ebit = _get_col_value(income, ebit_row, col)
                da = _get_col_value(income, da_row, col)
                if ebit is not None and da is not None:
                    ebitda = ebit + abs(da)

            if ebitda is None or ebitda <= 0:
                continue

            debt = _get_col_value(balance, debt_row, col) or 0
            cash = _get_col_value(balance, cash_row, col) or 0

            market_cap = price * shares
            ev = market_cap + debt - cash
            if ev <= 0:
                continue

            val = ev / ebitda
            if 0 < val < 200:
                values.append(val)

    except Exception as e:
        logger.debug(f"EV/EBITDA history error: {e}")
    return values


def _calc_per_history(income, balance, hist, shares: Optional[float], info: dict) -> list:
    """Calcula PER para cada año disponible."""
    values = []
    try:
        # Buscar EPS directo o Net Income para calcularlo
        eps_row = (
            _find_row(income, "diluted eps") or
            _find_row(income, "basic eps") or
            _find_row(income, "eps")
        )
        ni_row = (
            _find_row(income, "net income common stockholders") or
            _find_row(income, "net income") or
            _find_row(income, "netincome")
        )
        # Shares del balance (para calcular EPS desde NI)
        shares_row = (
            _find_row(balance, "diluted average shares") or
            _find_row(balance, "ordinary shares number") or
            _find_row(balance, "share issued") or
            _find_row(balance, "shares outstanding")
        )

        for col in income.columns[:HISTORY_YEARS]:
            year = col.year if hasattr(col, 'year') else pd.Timestamp(col).year
            price = _get_annual_price(hist, year)
            if not price:
                continue

            eps = _get_col_value(income, eps_row, col)

            # Si no hay EPS directo, calcularlo desde NI / shares
            if eps is None and ni_row:
                ni = _get_col_value(income, ni_row, col)
                sh = _get_col_value(balance, shares_row, col) if shares_row else None
                if sh is None:
                    sh = shares  # usar shares actuales como proxy
                if ni is not None and sh and sh > 0:
                    eps = ni / sh

            if eps is not None and eps > 0:
                val = price / eps
                if 0 < val < 1000:
                    values.append(val)

    except Exception as e:
        logger.debug(f"PER history error: {e}")
    return values


def _calc_pfcf_history(cashflow, balance, hist, shares: Optional[float], info: dict) -> list:
    """Calcula P/FCF para cada año disponible."""
    values = []
    try:
        fcf_row = (
            _find_row(cashflow, "free cash flow") or
            _find_row(cashflow, "freecashflow")
        )
        # Alternativamente: Operating CF - CapEx
        opcf_row = (
            _find_row(cashflow, "operating cash flow") or
            _find_row(cashflow, "cash from operations") or
            _find_row(cashflow, "operating activities")
        )
        capex_row = (
            _find_row(cashflow, "capital expenditure") or
            _find_row(cashflow, "capex") or
            _find_row(cashflow, "purchase of property")
        )
        shares_row = (
            _find_row(balance, "diluted average shares") or
            _find_row(balance, "ordinary shares number") or
            _find_row(balance, "share issued")
        )

        for col in cashflow.columns[:HISTORY_YEARS]:
            if col not in (balance.columns if balance is not None else []):
                continue
            year = col.year if hasattr(col, 'year') else pd.Timestamp(col).year
            price = _get_annual_price(hist, year)
            if not price:
                continue

            fcf = _get_col_value(cashflow, fcf_row, col)

            # Si no hay FCF directo, calcular desde OpCF - CapEx
            if fcf is None and opcf_row:
                opcf = _get_col_value(cashflow, opcf_row, col)
                capex = _get_col_value(cashflow, capex_row, col) or 0
                if opcf is not None:
                    fcf = opcf - abs(capex)  # capex suele ser negativo

            if fcf is None or fcf <= 0:
                continue

            sh = _get_col_value(balance, shares_row, col) if shares_row else None
            if sh is None:
                sh = shares
            if sh is None or sh <= 0:
                continue

            fcf_per_share = fcf / sh
            if fcf_per_share <= 0:
                continue
            val = price / fcf_per_share
            if 0 < val < 1000:
                values.append(val)

    except Exception as e:
        logger.debug(f"P/FCF history error: {e}")
    return values


def _calc_pb_history(balance, hist, shares: Optional[float], info: dict) -> list:
    """Calcula P/B para cada año disponible."""
    values = []
    try:
        equity_row = (
            _find_row(balance, "stockholders equity") or
            _find_row(balance, "total equity gross minority interest") or
            _find_row(balance, "total equity") or
            _find_row(balance, "common stock equity")
        )
        shares_row = (
            _find_row(balance, "diluted average shares") or
            _find_row(balance, "ordinary shares number") or
            _find_row(balance, "share issued") or
            _find_row(balance, "shares outstanding")
        )

        if equity_row is None:
            return values

        for col in balance.columns[:HISTORY_YEARS]:
            year = col.year if hasattr(col, 'year') else pd.Timestamp(col).year
            price = _get_annual_price(hist, year)
            if not price:
                continue

            equity = _get_col_value(balance, equity_row, col)
            sh = _get_col_value(balance, shares_row, col) if shares_row else None
            if sh is None:
                sh = shares

            if equity is None or sh is None or sh <= 0:
                continue

            bvps = equity / sh
            if bvps <= 0:
                continue

            val = price / bvps
            if 0 < val < 100:
                values.append(val)

    except Exception as e:
        logger.debug(f"P/B history error: {e}")
    return values


def _calc_divy_history(tk, hist) -> list:
    """Calcula Dividend Yield histórico usando dividendos pagados."""
    values = []
    try:
        divs = tk.dividends
        if divs is None or divs.empty:
            return values

        # Normalizar timezone de dividendos
        if hasattr(divs.index, 'tz') and divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)

        min_year = hist.index[-1].year - HISTORY_YEARS + 1
        max_year = hist.index[-1].year

        for year in range(min_year, max_year + 1):
            annual_div = divs[divs.index.year == year].sum()
            price = _get_annual_price(hist, year)
            if annual_div > 0 and price and price > 0:
                val = annual_div / price
                if 0 < val < 0.5:
                    values.append(float(val))

    except Exception as e:
        logger.debug(f"DivY history error: {e}")
    return values


def fetch_all_tickers(tickers: list, progress_callback=None) -> pd.DataFrame:
    """
    Descarga datos para una lista de tickers.
    progress_callback(i, total, ticker): función opcional para mostrar progreso.
    Retorna DataFrame con una fila por ticker.
    """
    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        logger.info(f"Fetching {ticker} ({i+1}/{total})")
        data = fetch_ticker_data(ticker)
        results.append(data)
        time.sleep(REQUEST_DELAY)

    return pd.DataFrame(results)
