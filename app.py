#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 10:21:54 2025

@author: donazlobko
"""

# app.py — Streamlit UI z ENOTNIM opozorilom za PE (15–90)
import numpy as np
import streamlit as st
from model_core import U_generalized, g_generalized  # brez izpisov ob importu

st.set_page_config(page_title="U & g – izračun", layout="centered")
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
      .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Izračun U(PE, G, T) in g(U, G, T)")

# ===== FIKSNE MEJE PE ZA VSE PRIMERE =====
PE_MIN, PE_MAX = 15.0, 90.0   # kWh/m²a — enako za vse kraje

# --- VNOSI ---
kraj = st.text_input("Kraj (poljubno besedilo)", placeholder="npr. Ljubljana")

col1, col2 = st.columns(2)
with col1:
    G = st.number_input("Letno globalno obsevanje G [kWh/m²a]", min_value=1.0, value=1200.0, step=10.0)
with col2:
    T = st.number_input("Povprečna letna temperatura T [°C]", value=12.0, step=0.1, format="%.1f")

st.caption(f"Sprejemljiv razpon primarne energije (PE): {PE_MIN:.0f}–{PE_MAX:.0f} kWh/m²a.")

PE = st.number_input(
    "Želen PE [kWh/m²a]",
    min_value=0.0, max_value=1000.0,
    value=max(PE_MIN, 0.0),
    step=0.5
)

st.divider()

# --- GUMB IN IZPIS ---
if st.button("Izračunaj"):
    # eno opozorilo glede na fiksne meje
    if PE < PE_MIN - 1e-9:
        st.warning(f"Izbrana PE ({PE:.1f}) je **premajhna**. Sprejemljiv razpon je {PE_MIN:.0f}–{PE_MAX:.0f} kWh/m²a.")
    elif PE > PE_MAX + 1e-9:
        st.warning(f"Izbrana PE ({PE:.1f}) je **prevelika**. Sprejemljiv razpon je {PE_MIN:.0f}–{PE_MAX:.0f} kWh/m²a.")

    try:
        Uopt = U_generalized(PE=PE, G=G, T=T)
        gopt = g_generalized(U=Uopt, G=G, T=T)
        st.metric("U*(PE, G, T) [W/m²K]", f"{Uopt:.4f}")
        st.metric("g*(U, G, T) [-]", f"{gopt:.4f}")
    except Exception:
        st.error("Napaka pri izračunu.")
