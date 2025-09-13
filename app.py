#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 10:21:54 2025

@author: donazlobko
"""


# app.py — minimalni Streamlit UI za izračun U(PE,G,T) in g(U,G,T)

import streamlit as st
from model_core import U_generalized, g_generalized  # ničesar ne izpiše ob importu

# minimalen videz (brez menija, headerja, footerja)
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

#naslov
st.title("Izračun U(PE, G, T) in g(U, G, T)")


# --- VNOSI (brez drop-downa za kraj) ---
kraj = st.text_input("Kraj (poljubno besedilo)", placeholder="npr. Ljubljana")
PE = st.number_input("Želen PE [kWh/m²a]", min_value=0.0, value=30.0, step=0.5)
G  = st.number_input("Letno globalno obsevanje G [kWh/m²a]", min_value=1.0, value=1200.0, step=10.0)
T  = st.number_input("Povprečna letna temperatura T [°C]", value=12.0, step=0.1, format="%.1f")

st.divider()

# --- GUMB IN IZPIS (samo rezultata) ---
if st.button("Izračunaj"):
    try:
        Uopt = U_generalized(PE=PE, G=G, T=T)
        gopt = g_generalized(U=Uopt, G=G, T=T)
        st.metric("U*(PE, G, T) [W/m²K]", f"{Uopt:.4f}")
        st.metric("g*(U, G, T) [-]", f"{gopt:.4f}")
    except Exception as e:
        # edino kar pokažemo ob napaki je kratko sporočilo
        st.error("Napaka pri izračunu.")
