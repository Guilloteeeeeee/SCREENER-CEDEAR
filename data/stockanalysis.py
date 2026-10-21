"""
Scraper de Stock Analysis (stockanalysis.com) para obtener ratios históricos.

Se usa como fuente secundaria cuando yfinance no devuelve estados financieros
completos para calcular las medianas históricas.

Endpoints usados:
  - /stocks/{ticker}/financials/ratios/?p=annual  → PE Ratio, PB Ratio, Dividend Yield
  - /stocks/{ticker}/financials/?p=annual         → EPS, FCF, Dividend Per Share
  - /stocks/{ticker}/financials/balance-sheet/?p=annual → Book Value Per Share, Shares

Para ADRs (tickers sin sufijo como BABA, TSM, IBN):
  - Primero intenta /stocks/{ticker}/
  - Si falla, intenta /etf/{ticker}/ (algunos ETFs)
"""

import re
import time
import logging
import numpy as np
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Headers para no ser bloqueado
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Pausa entre requests
REQUEST_DELAY = 1.0

# Años a buscar
MAX_YEARS = 5


def _fetch_html(url: str, session=None) -> Optional[str]:
    """Descarga HTML de una URL."""
    try:
        import requests
        s = session or requests.Session()
        resp = s.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        logger.debug(f"HTTP {resp.status_code} para {url}")
    except Exception as e:
        logger.debug(f"Error fetching {url}: {e}")
    return None


def _parse_table(html: str, section_keywords: list) -> dict:
    """
    Parsea una tabla de Stock Analysis y retorna dict {row_label: {year: value}}.
    Busca la tabla que contenga alguna de las keywords en sus filas.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("BeautifulSoup no instalado — pip install beautifulsoup4")
        return {}

    result = {}
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Stock Analysis usa tablas con clase específica o simplemente <table>
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            # Leer encabezados (años)
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(["th", "td"]):
                text = th.get_text(strip=True)
                headers.append(text)

            if len(headers) < 2:
                continue

            # Leer filas de datos
            table_data = {}
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                label = cells[0].get_text(strip=True)
                if not label:
                    continue
                row_data = {}
                for i, cell in enumerate(cells[1:], 1):
                    if i < len(headers):
                        val_text = cell.get_text(strip=True)
                        val = _parse_number(val_text)
                        if val is not None:
                            row_data[headers[i]] = val
                if row_data:
                    table_data[label] = row_data

            # ¿Esta tabla tiene alguna de las keywords?
            found = any(
                any(kw.lower() in label.lower() for kw in section_keywords)
                for label in table_data.keys()
            )
            if found:
                result.update(table_data)

    except Exception as e:
        logger.debug(f"Error parseando tabla: {e}")

    return result


def _parse_number(text: str) -> Optional[float]:
    """Convierte texto de Stock Analysis a float."""
    if not text or text in ("-", "—", "N/A", "", "∞"):
        return None
    try:
        # Eliminar signos de porcentaje, $, comas, espacios
        clean = text.replace("%", "").replace("$", "").replace(",", "").strip()
        # Manejar sufijos B/M/K (billones, millones, miles)
        multiplier = 1.0
        if clean.endswith("B"):
            multiplier = 1e9
            clean = clean[:-1]
        elif clean.endswith("M"):
            multiplier = 1e6
            clean = clean[:-1]
        elif clean.endswith("K"):
            multiplier = 1e3
            clean = clean[:-1]
        val = float(clean) * multiplier
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None


def _find_label(data: dict, *keywords, exclude: list = None) -> Optional[str]:
    """Busca la primera clave del dict que contenga todos los keywords."""
    exclude = exclude or []
    for label in data.keys():
        label_lower = label.lower()
        if all(kw.lower() in label_lower for kw in keywords):
            if not any(ex.lower() in label_lower for ex in exclude):
                return label
    return None


def _extract_annual_series(row_data: dict, max_years: int = MAX_YEARS) -> list:
    """
    Extrae valores anuales de un dict {header: value}.
    Filtra solo columnas que parecen años (FY20xx, 20xx, TTM).
    Retorna lista de floats, máximo max_years valores.
    """
    values = []
    # Ordenar por año descendente: FY2025 > FY2024 > ...
    year_cols = []
    for col in row_data.keys():
        # Buscar años: FY2021, 2021, FY 2021, etc.
        match = re.search(r'(20\d{2})', str(col))
        if match:
            year_cols.append((int(match.group(1)), col))

    year_cols.sort(reverse=True)

    for _, col in year_cols[:max_years]:
        val = row_data.get(col)
        if val is not None and val > 0:
            values.append(val)

    return values


def fetch_ratios_stockanalysis(ticker: str, session=None) -> dict:
    """
    Obtiene ratios históricos de Stock Analysis para un ticker.

    Retorna dict con listas de valores históricos:
      - per_historia: [float, ...]
      - p_b_historia: [float, ...]
      - ev_ebitda_historia: [float, ...]
      - p_fcf_historia: [float, ...]
      - dividend_yield_historia: [float, ...]
    """
    result = {
        "per_historia": [],
        "p_b_historia": [],
        "ev_ebitda_historia": [],
        "p_fcf_historia": [],
        "dividend_yield_historia": [],
    }

    base = f"https://stockanalysis.com/stocks/{ticker.lower()}"

    try:
        import requests
        session = session or requests.Session()

        # --- 1. Ratios page: PE, PB, Dividend Yield ---
        time.sleep(REQUEST_DELAY)
        ratios_html = _fetch_html(f"{base}/financials/ratios/?p=annual", session)

        if ratios_html:
            ratios_data = _parse_table(ratios_html, ["PE Ratio", "PB Ratio", "Dividend Yield"])

            pe_label = _find_label(ratios_data, "pe ratio")
            if pe_label:
                result["per_historia"] = _extract_annual_series(ratios_data[pe_label])

            pb_label = _find_label(ratios_data, "pb ratio")
            if pb_label:
                result["p_b_historia"] = _extract_annual_series(ratios_data[pb_label])

            dy_label = _find_label(ratios_data, "dividend yield")
            if dy_label:
                vals = _extract_annual_series(ratios_data[dy_label])
                # Dividend Yield en SA viene como porcentaje (ej: 2.5 = 2.5%)
                result["dividend_yield_historia"] = [v / 100.0 for v in vals if 0 < v < 50]

        # --- 2. Income statement: EPS, FCF ---
        time.sleep(REQUEST_DELAY)
        income_html = _fetch_html(f"{base}/financials/?p=annual", session)

        if income_html:
            income_data = _parse_table(income_html, ["EPS", "Free Cash Flow", "Revenue"])

            # EPS (para calcular PER si no lo tenemos de ratios)
            eps_label = _find_label(income_data, "eps", exclude=["growth", "diluted"])
            if not eps_label:
                eps_label = _find_label(income_data, "earnings per share")

            # FCF para P/FCF — necesitamos precio del año también
            # Por ahora guardamos FCF para combinar con precio histórico
            fcf_label = _find_label(income_data, "free cash flow")
            if fcf_label:
                # Guardamos FCF crudo para combinar con shares y precio
                result["_fcf_historia_raw"] = income_data[fcf_label]

        # --- 3. Balance sheet: Book Value Per Share ---
        time.sleep(REQUEST_DELAY)
        balance_html = _fetch_html(f"{base}/financials/balance-sheet/?p=annual", session)

        if balance_html:
            balance_data = _parse_table(balance_html, ["Book Value", "Shares Outstanding"])

            # Book Value Per Share (más directo para P/B)
            bvps_label = _find_label(balance_data, "book value per share",
                                     exclude=["tangible"])
            if bvps_label and not result["p_b_historia"]:
                # Si no tenemos P/B de ratios, lo calculamos
                # Necesitaríamos precio histórico también — lo dejamos para la integración
                result["_bvps_historia_raw"] = balance_data[bvps_label]

            # Shares Outstanding para calcular P/FCF
            shares_label = (
                _find_label(balance_data, "shares outstanding", exclude=["filing"]) or
                _find_label(balance_data, "common shares outstanding")
            )
            if shares_label:
                result["_shares_historia_raw"] = balance_data[shares_label]

    except Exception as e:
        logger.warning(f"Error en Stock Analysis para {ticker}: {e}")

    return result


def enrich_with_stockanalysis(ticker: str, result: dict, precio_hist: dict) -> None:
    """
    Enriquece el result dict con medianas de Stock Analysis cuando yfinance no tiene histórico.

    precio_hist: dict {year: precio_cierre} para calcular ratios de precio.
    Modifica result in-place, solo llena campos que siguen siendo None.
    """
    # Solo actuar si hay campos que aún necesitan mediana
    needs = [
        ("per_mediana", "per_actual"),
        ("p_b_mediana", "p_b_actual"),
        ("ev_ebitda_mediana", "ev_ebitda_actual"),
        ("p_fcf_mediana", "p_fcf_actual"),
        ("dividend_yield_mediana", "dividend_yield_actual"),
    ]
    if not any(result.get(med) is None and result.get(act) is not None
               for med, act in needs):
        return  # Ya tiene todas las medianas, no hace falta

    logger.info(f"Usando Stock Analysis como fallback para {ticker}")

    try:
        sa = fetch_ratios_stockanalysis(ticker)

        # PER mediana
        if result.get("per_mediana") is None and sa["per_historia"]:
            result["per_mediana"] = round(float(np.median(sa["per_historia"])), 2)

        # P/B mediana
        if result.get("p_b_mediana") is None and sa["p_b_historia"]:
            result["p_b_mediana"] = round(float(np.median(sa["p_b_historia"])), 2)

        # Dividend Yield mediana
        if result.get("dividend_yield_mediana") is None and sa["dividend_yield_historia"]:
            result["dividend_yield_mediana"] = round(
                float(np.median(sa["dividend_yield_historia"])), 4
            )

        # P/B desde BVPS + precio histórico (si P/B de ratios no disponible)
        if result.get("p_b_mediana") is None and "_bvps_historia_raw" in sa and precio_hist:
            pb_vals = []
            for col, bvps in sa["_bvps_historia_raw"].items():
                match = re.search(r'(20\d{2})', str(col))
                if match and bvps and bvps > 0:
                    year = int(match.group(1))
                    precio = precio_hist.get(year)
                    if precio and precio > 0:
                        val = precio / bvps
                        if 0 < val < 100:
                            pb_vals.append(val)
            if pb_vals:
                result["p_b_mediana"] = round(float(np.median(pb_vals)), 2)

        # P/FCF desde FCF + shares + precio histórico
        if result.get("p_fcf_mediana") is None and "_fcf_historia_raw" in sa and precio_hist:
            pfcf_vals = []
            shares_raw = sa.get("_shares_historia_raw", {})
            for col, fcf in sa["_fcf_historia_raw"].items():
                match = re.search(r'(20\d{2})', str(col))
                if not match or fcf is None or fcf <= 0:
                    continue
                year = int(match.group(1))
                precio = precio_hist.get(year)
                if not precio or precio <= 0:
                    continue
                # Buscar shares del mismo año
                shares = None
                for scol, sv in shares_raw.items():
                    smatch = re.search(r'(20\d{2})', str(scol))
                    if smatch and int(smatch.group(1)) == year and sv and sv > 0:
                        shares = sv
                        break
                # Si no hay shares histórico, usar shares actuales
                if shares is None:
                    shares_act = result.get("market_cap") and result.get("precio_actual")
                    if shares_act:
                        shares = result["market_cap"] / result["precio_actual"]
                if shares and shares > 0:
                    # FCF en SA puede estar en millones
                    fcf_per_share = fcf / shares
                    if fcf_per_share > 0:
                        val = precio / fcf_per_share
                        if 0 < val < 1000:
                            pfcf_vals.append(val)
            if pfcf_vals:
                result["p_fcf_mediana"] = round(float(np.median(pfcf_vals)), 2)

    except Exception as e:
        logger.warning(f"Error enriqueciendo con Stock Analysis para {ticker}: {e}")
