#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 10:17:25 2025

@author: donazlobko
"""

# model_core.py
# — jedro meta-modela: definicije in funkciji U_generalized(PE,G,T) in g_generalized(U,G,T)
#   NI nobenih printov, string-tekstov za prikaz ali interaktivnosti.

import numpy as np
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# -------------------------
# VHODNI PODATKI (tri “učne” lokacije)
# -------------------------
GLOB = {"Tromso": 732.0, "Ljubljana": 1250.0, "Malaga": 1716.0}      # kWh/m2·a
TEMP = {"Tromso": 2.9, "Ljubljana": 12.6, "Malaga": 17.6}            # °C

# U(PE) = a0 + a1*PE + a2*PE^2 + a3*PE^3
U_COEFS = {
    "Tromso":    dict(a0=-1.508,  a1=0.068,  a2=0.000,  a3=0.0),
    "Ljubljana": dict(a0=-2.963,  a1=0.250,  a2=-0.005, a3=0.0),
    "Malaga":    dict(a0=-13.988, a1=1.376,  a2=-0.029, a3=0.0),
}

# g(U) = b0 + b1*U + b2*U^2 + b3*U^3
G_COEFS = {
    "Tromso":    dict(b0=0.223, b1=0.140, b2=-0.018, b3=0.0),
    "Ljubljana": dict(b0=0.202, b1=0.087, b2=-0.014, b3=0.0),
    "Malaga":    dict(b0=0.192, b1=0.035, b2=0.000,  b3=0.0),
}

# -------------------------
# 2D FIT KANDIDATI in ORODJA (isti kot prej, le brez printov)
# -------------------------
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
    if family == "lin_G_T":
        X = np.column_stack([np.ones_like(G), G, T])
        predict = lambda params, g, t: float(params[0] + params[1]*g + params[2]*t)
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
    if family == "lin_lnG_T":
        X = np.column_stack([np.ones_like(G), np.log(G), T])
        predict = lambda params, g, t: float(params[0] + params[1]*math.log(g) + params[2]*t)
        return X, predict
    if family == "lin_G_lnT":
        X = np.column_stack([np.ones_like(G), G, np.log(T)])
        predict = lambda params, g, t: float(params[0] + params[1]*g + params[2]*math.log(t))
        return X, predict
    if family == "lin_lnG_lnT":
        X = np.column_stack([np.ones_like(G), np.log(G), np.log(T)])
        predict = lambda params, g, t: float(params[0] + params[1]*math.log(g) + params[2]*math.log(t))
        return X, predict
    if family == "exp_GT":
        X = np.column_stack([np.ones_like(G), G, T])
        predict = lambda params, g, t: float(math.exp(params[0] + params[1]*g + params[2]*t))
        return X, predict
    if family == "power_GT":
        X = np.column_stack([np.ones_like(G), np.log(G), np.log(T)])
        predict = lambda params, g, t: float(math.exp(params[0]) * (g**params[1]) * (t**params[2]))
        return X, predict
    raise ValueError("Neznana družina modelov.")

def _candidate_families(y: np.ndarray):
    cands = [
        ("lin_G_T", False),
        ("lin_G_T_GT", False),
        ("lin_G_T_G2", False),
        ("lin_G_T_T2", False),
        ("lin_lnG_T", False),
        ("lin_G_lnT", False),
        ("lin_lnG_lnT", False),
    ]
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

def choose_model_2d(G: np.ndarray, T: np.ndarray, y: np.ndarray) -> Fit2DResult:
    best = None
    for family, use_logy in _candidate_families(y):
        try:
            rmse = _loocv_rmse_2d(G, T, y, family, use_logy)
            params, mae = _fit_all_2d(G, T, y, family, use_logy)
            fr = Fit2DResult(("logy_"+family if use_logy else family), np.atleast_1d(params), len(params), rmse, mae)
            if (best is None) or (fr.rmse_loocv < best.rmse_loocv - 1e-12) or \
               (abs(fr.rmse_loocv - best.rmse_loocv) <= 1e-12 and fr.pcount < best.pcount):
                best = fr
        except Exception:
            continue
    return best

def build_predictor_2d(fr: Fit2DResult):
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

# — nauči 2D meta-koeficiente pri importu (brez izpisov)
_locs = list(GLOB.keys())
_G = np.array([GLOB[L] for L in _locs], float)
_T = np.array([TEMP[L] for L in _locs], float)

a_keys = coeff_keys('a', U_COEFS)
b_keys = coeff_keys('b', G_COEFS)

a_fit  = {k: choose_model_2d(_G, _T, stack_y(a_keys, U_COEFS, _locs)[k]) for k in a_keys}
b_fit  = {k: choose_model_2d(_G, _T, stack_y(b_keys, G_COEFS, _locs)[k]) for k in b_keys}

a_pred = {k: build_predictor_2d(a_fit[k]) for k in a_keys}
b_pred = {k: build_predictor_2d(b_fit[k]) for k in b_keys}

# -------------------------
# JAVNE FUNKCIJE
# -------------------------
def U_generalized(PE: float, G: float, T: float) -> float:
    return sum(a_pred[k](G, T) * (PE ** int(k[1:])) for k in a_keys)

def g_generalized(U: float, G: float, T: float) -> float:
    return sum(b_pred[k](G, T) * (U  ** int(k[1:])) for k in b_keys)
