#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 10:21:54 2025

@author: donazlobko
"""
# app.py — Streamlit UI
import streamlit as st
from model_core import U_generalized, g_generalized

st.set_page_config(page_title="U & g – izračun", layout="centered")
st.markdown("""
<style>
  #MainMenu {visibility: hidden;}
  header {visibility: hidden;}
  footer {visibility: hidden;}
  .block-container {padding-top: 1rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("Izračun U(PE, G, T) in g(U, G, T)")

# ---- VNOSI ----
kraj = st.text_input("Kraj (poljubno besedilo)", placeholder="npr. Ljubljana")

col1, col2 = st.columns(2)
with col1:
    G = st.number_input("Letno globalno obsevanje G [kWh/m²a]", min_value=1.0, value=1250.0, step=10.0)
with col2:
    T = st.number_input("Povprečna letna temperatura T [°C]", value=12.6, step=0.1, format="%.1f")

PE = st.number_input("Želen PE [kWh/m²a]", min_value=0.0, max_value=10000.0, value=120.0, step=0.5)
PE_MIN, PE_MAX = 15.0, 90.0
st.caption(f"Sprejemljiv razpon za PE je {PE_MIN:.0f}–{PE_MAX:.0f} kWh/m²a.")

st.divider()

# ---- GUMB IN IZPIS ----
if st.button("Izračunaj"):
    # ✅ VALIDACIJA: ne računaj, če je PE izven razpona
    if PE < PE_MIN:
        st.warning(f"Izbrana PE ({PE:.1f}) je **premajhna**. Sprejemljiv razpon je {PE_MIN:.0f}–{PE_MAX:.0f} kWh/m²a.")
        st.stop()
    if PE > PE_MAX:
        st.warning(f"Izbrana PE ({PE:.1f}) je **prevelika**. Sprejemljiv razpon je {PE_MIN:.0f}–{PE_MAX:.0f} kWh/m²a.")
        st.stop()

    # če pridemo sem, je PE veljavna → izračunamo
    try:
        Uopt = U_generalized(PE=PE, G=G, T=T)
        gopt = g_generalized(U=Uopt, G=G, T=T)
        st.metric("U*(PE, G, T) [W/m²K]", f"{Uopt:.4f}")
        st.metric("g*(U, G, T) [-]", f"{gopt:.4f}")
    except Exception as e:
        st.error(f"Napaka pri izračunu: {e}")
