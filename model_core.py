#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 10:17:25 2025

@author: donazlobko
"""

# model_core.py
# — Jedro meta-modela: učenje 2D koeficientov, omejevanje izhodov (sredinsko/squash/clip),
#   in javne funkcije U_generalized(PE,G,T) ter g_generalized(U,G,T).
#   Ob uvozu NI izpisov ali interaktivnosti.

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

# =========================
# Nastavitve
# =========================
U_MIN, U_MAX = 0.2, 2.4
G_MIN, G_MAX = 0.2, 0.7

# Način, ko surova vrednost uide iz mej:
#  - "clip": trd odrez [lo,hi]
#  - "squash": gladka logistična preslikava v [lo,hi]
#  - "mid_squash": gladka preslikava v ožji pas [lo+δ, hi-δ] (sredinska opcija)
OUT_OF_RANGE_MODE = "mid_squash"

# Mehkejši squash (večji floors => manj stiskanja)
U_SCALE_FLOOR = 0.10
g_SCALE_FLOOR = 0.06

# Center “usidraj” v tipične pogoje
PE_REF = 130.0
USE_ANCHORED_CENTER = True

# Kontrola nabora družin za 2D koeficiente
EXCLUDE_QUADRATICS = True  # True = samo osnovne; False = doda GT, G^2, T^2

# =========================
# Učni podatki (tri lokacije)
# =========================
GLOB = {"Tromso": 732.0, "Ljubljana": 1250.0, "Malaga": 1716.0}      # kWh/m2·a
TEMP = {"Tromso": 2.9, "Ljubljana": 12.6, "Malaga": 17.6}            # °C

# U(PE) = a0 + a1·PE + a2·PE^2 + a3·PE^3
U_COEFS = {
    "Tromso":    dict(a0=-1.508,  a1=0.068,  a2=0.000,  a3=0.0),
    "Ljubljana": dict(a0=-2.963,  a1=0.250,  a2=-0.005, a3=0.0),
    "Malaga":    dict(a0=-13.988, a1=1.376,  a2=-0.029, a3=0.0),
}

# g(U) = b0 + b1·U + b2·U^2 + b3·U^3
G_COEFS = {
    "Tromso":    dict(b0=0.223, b1=0.140, b2=-0.018, b3=0.0),
    "Ljubljana": dict(b0=0.202, b1=0.087, b2=-0.014, b3=0.0),
    "Malaga":    dict(b0=0.192, b1=0.035, b2=0.000,  b3=0.0),
}

# =========================
# 2D fit (izbira družin + ridge)
# =========================
@dataclass
class Fit2DResult:
    name: str
    params: np.ndarray
    pcount: int
    rmse_loocv: float
    mae_in: float

RIDGE_ALPHA = 1e-8

def _ridge_solve(X: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    XT = X.T
    A = XT @ X + alpha * np.eye(X.shape[1])
    b = XT @ y
    return np.linalg.solve(A, b)

def _build_X(G: np.ndarray, T: np.ndarray, family: str):
    import math as _m
    if family == "lin_G_T":
        X = np.column_stack([np.ones_like(G), G, T])
        predict = lambda params, g, t: float(params[0] + params[1]*g + params[2]*t)
        return X, predict
    if family == "lin_lnG_T":
        X = np.column_stack([np.ones_like(G), np.log(G), T])
        predict = lambda params, g, t: float(params[0] + params[1]*_m.log(g) + params[2]*t)
        return X, predict
    if family == "lin_G_lnT":
        X = np.column_stack([np.ones_like(G), G, np.log(T)])
        predict = lambda params, g, t: float(params[0] + params[1]*g + params[2]*_m.log(t))
        return X, predict
    if family == "lin_lnG_lnT":
        X = np.column_stack([np.ones_like(G), np.log(G), np.log(T)])
        predict = lambda params, g, t: float(params[0] + params[1]*_m.log(g) + params[2]*_m.log(t))
        return X, predict
    if family == "lin_G_T_GT":
        X = np.column_stack([np.ones_like(G), G, T, G*T])
        predict = lambda params, g, t: float(params[0] + params[1]*g + params[2]*t + params[3]*g*t)
        return X, predict
    if family == "lin_G_T_G2":
        X = np.column_stack([np.ones_like(G), G, T, G**2])
        predict = lambda params, g, t: float(params[0] + params[1]*g + params[2]*t + params[3]*g*g)
        return X, predict
    if family == "lin_G_T_T2":
        X = np.column_stack([np.ones_like(G), G, T, T**2])
        predict = lambda params, g, t: float(params[0] + params[1]*g + params[2]*t + params[3]*t*t)
        return X, predict
    if family == "exp_GT":
        X = np.column_stack([np.ones_like(G), G, T])
        predict = lambda params, g, t: float(_m.exp(params[0] + params[1]*g + params[2]*t))
        return X, predict
    if family == "power_GT":
        X = np.column_stack([np.ones_like(G), np.log(G), np.log(T)])
        predict = lambda params, g, t: float(_m.exp(params[0]) * (g**params[1]) * (t**params[2]))
        return X, predict
    raise ValueError("Neznana družina modelov.")

def _candidate_families(y: np.ndarray):
    base = [
        ("lin_G_T", False),
        ("lin_lnG_T", False),
        ("lin_G_lnT", False),
        ("lin_lnG_lnT", False),
    ]
    rich = [
        ("lin_G_T_GT", False),
        ("lin_G_T_G2", False),
        ("lin_G_T_T2", False),
    ]
    cands = base if EXCLUDE_QUADRATICS else (base + rich)
    if np.all(y > 0):
        cands += [("exp_GT", True), ("power_GT", True)]
    return cands

def _loocv_rmse_2d(G: np.ndarray, T: np.ndarray, y: np.ndarray, family: str, use_logy: bool) -> float:
    n = len(y); se = []
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i] = False
        Gtr, Ttr, ytr = G[m], T[m], y[m]
        yfit = np.log(ytr) if use_logy else ytr
        Xtr, predict = _build_X(Gtr, Ttr, family)
        params = _ridge_solve(Xtr, yfit, RIDGE_ALPHA)
        yi = predict(params, G[i], T[i])
        if use_logy: yi = float(np.exp(yi))
        se.append((y[i] - yi)**2)
    return float(np.sqrt(np.mean(se)))

def _fit_all_2d(G: np.ndarray, T: np.ndarray, y: np.ndarray, family: str, use_logy: bool):
    yfit = np.log(y) if use_logy else y
    X, predict = _build_X(G, T, family)
    params = _ridge_solve(X, yfit, RIDGE_ALPHA)
    pred = np.array([predict(params, g, t) for g, t in zip(G, T)], float)
    if use_logy: pred = np.exp(pred)
    mae = float(np.mean(np.abs(y - pred)))
    return params, mae

@dataclass
class _Choice:
    name: str
    params: np.ndarray
    pcount: int
    rmse_loocv: float
    mae_in: float

def choose_model_2d(G: np.ndarray, T: np.ndarray, y: np.ndarray) -> _Choice:
    best = None
    for family, use_logy in _candidate_families(y):
        try:
            rmse = _loocv_rmse_2d(G, T, y, family, use_logy)
            params, mae = _fit_all_2d(G, T, y, family, use_logy)
            cand = _Choice(("logy_"+family if use_logy else family),
                           np.atleast_1d(params), len(params), rmse, mae)
            if (best is None) or (cand.rmse_loocv < best.rmse_loocv - 1e-12) or \
               (abs(cand.rmse_loocv - best.rmse_loocv) <= 1e-12 and cand.pcount < best.pcount):
                best = cand
        except Exception:
            continue
    return best

def build_predictor_2d(fr: _Choice):
    name = fr.name.replace("logy_", "")
    _, predict = _build_X(np.array([1.0]), np.array([1.0]), name)
    if fr.name.startswith("logy_"):
        return lambda G, T: float(np.exp(predict(fr.params, G, T)))
    return lambda G, T: float(predict(fr.params, G, T))

def coeff_keys(prefix: str, store: Dict[str, Dict[str, float]]) -> List[str]:
    keys = set()
    for loc in store:
        for k in store[loc]:
            if k.startswith(prefix):
                keys.add(k)
    return sorted([k for k in keys if int(k[1:]) <= 3], key=lambda s: int(s[1:]))

def stack_y(keys: List[str], store: Dict[str, Dict[str, float]], locs: List[str]):
    return {k: np.array([store[loc].get(k, 0.0) for loc in locs], float) for k in keys}

# =========================
# Učenje 2D meta-koeficientov (pri importu)
# =========================
_locs = list(GLOB.keys())
_G = np.array([GLOB[L] for L in _locs], float)
_T = np.array([TEMP[L] for L in _locs], float)

a_keys = coeff_keys('a', U_COEFS)
b_keys = coeff_keys('b', G_COEFS)

a_fit  = {k: choose_model_2d(_G, _T, stack_y(a_keys, U_COEFS, _locs)[k]) for k in a_keys}
b_fit  = {k: choose_model_2d(_G, _T, stack_y(b_keys, G_COEFS, _locs)[k]) for k in b_keys}

a_pred = {k: build_predictor_2d(a_fit[k]) for k in a_keys}
b_pred = {k: build_predictor_2d(b_fit[k]) for k in b_keys}

# =========================
# Surovi izračun U in g
# =========================
def __U_raw(PE: float, G: float, T: float) -> float:
    val = 0.0
    for k in a_keys:
        p = int(k[1:])
        val += a_pred[k](G, T) * (PE ** p)
    return float(val)

def __g_raw(U: float, G: float, T: float) -> float:
    val = 0.0
    for k in b_keys:
        p = int(k[1:])
        val += b_pred[k](G, T) * (U ** p)
    return float(val)

# =========================
# Omejevanje izhodov (uporabimo samo, ko gremo IZVEN mej)
# =========================
def __squash_sigmoid(x: float, lo: float, hi: float, center: float, scale: float) -> float:
    z = (x - center) / (scale if scale > 1e-12 else 1.0)
    s = 1.0 / (1.0 + math.exp(-z))
    return lo + (hi - lo) * s

def __clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))

def __mid_squash(x: float, lo: float, hi: float, center: float, scale: float) -> float:
    """
    Sredinska opcija: gladek odrez v ožji pas [lo+δ, hi-δ].
    δ je del razpona in rahlo narašča z oddaljenostjo od centra.
    """
    span = (hi - lo)
    base_margin = 0.125 * span
    dist_factor = min(1.0, abs(x - center) / (2.0 * span))  # 0..1
    delta = base_margin * (0.8 + 0.2 * dist_factor)         # 80–100% base_margin
    lo2, hi2 = lo + delta, hi - delta
    if lo2 >= hi2:  # varovalo
        lo2, hi2 = (lo + 0.45*span, lo + 0.55*span)
    z = (x - center) / (scale if scale > 1e-12 else 1.0)
    s = 1.0 / (1.0 + math.exp(-z))
    return lo2 + (hi2 - lo2) * s

def __limit_if_needed(x: float, lo: float, hi: float, center: float, scale: float) -> float:
    """Če je x v [lo,hi], vrni x; sicer uporabi izbrani način omejitve."""
    if lo <= x <= hi:
        return float(x)
    if OUT_OF_RANGE_MODE == "clip":
        return __clip(x, lo, hi)
    if OUT_OF_RANGE_MODE == "squash":
        return __squash_sigmoid(x, lo, hi, center, scale)
    return __mid_squash(x, lo, hi, center, scale)  # "mid_squash"

# =========================
# Kalibracija centra in skale za squash
# =========================
def __calibrate_centers_and_scales():
    G_ref = float(np.mean(list(GLOB.values())))
    T_ref = float(np.mean(list(TEMP.values())))

    PE_grid = np.linspace(10.0, 250.0, 200)
    U_grid  = np.linspace(U_MIN, U_MAX, 200)

    U_vals, g_vals = [], []
    for L in _locs:
        Gv, Tv = GLOB[L], TEMP[L]
        U_vals.append(np.array([__U_raw(pe, Gv, Tv) for pe in PE_grid], float))
        g_vals.append(np.array([__g_raw(u,  Gv, Tv) for u  in U_grid], float))

    U_all = np.concatenate(U_vals)
    g_all = np.concatenate(g_vals)

    if USE_ANCHORED_CENTER:
        U_center = float(__U_raw(PE_REF, G_ref, T_ref))
        g_center = float(__g_raw(U_center, G_ref, T_ref))
    else:
        U_center = float(np.median(U_all))
        g_center = float(np.median(g_all))

    def iqr(x):
        q1, q3 = np.percentile(x, [25, 75])
        return max(q3 - q1, 1e-6)

    U_scale = float(iqr(U_all) / 1.349)
    g_scale = float(iqr(g_all) / 1.349)

    if U_scale < U_SCALE_FLOOR: U_scale = U_SCALE_FLOOR
    if g_scale < g_SCALE_FLOOR: g_scale = g_SCALE_FLOOR

    g_scale *= 1.5  # malo “odkleni” g
    return U_center, U_scale, g_center, g_scale

_U_center, _U_scale, _g_center, _g_scale = __calibrate_centers_and_scales()

# =========================
# Javne funkcije – “surovo, če v mejah; sicer omeji”
# =========================
def U_generalized(PE: float, G: float, T: float) -> float:
    u = __U_raw(PE, G, T)
    return __limit_if_needed(u, U_MIN, U_MAX, _U_center, _U_scale)

def g_generalized(U: float, G: float, T: float) -> float:
    gg = __g_raw(U, G, T)
    return __limit_if_needed(gg, G_MIN, G_MAX, _g_center, _g_scale)

# (konec model_core.py)

    gg = __g_raw(U, G, T)
    return __squash_sigmoid(gg, G_MIN, G_MAX, _g_center, _g_scale)

