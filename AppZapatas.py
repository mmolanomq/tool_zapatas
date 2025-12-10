# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:23:33 2025

@author: Usuario
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Diseño Óptimo de Zapatas", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; }
    .stMetric { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. INTERFAZ DE USUARIO (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.title("Optimización de Zapatas")
    st.info("Enfoque: Minimización de Volúmenes de Material")
    
    st.subheader("1. Estratigrafía del Suelo")
    PROF_LIMIT = st.number_input("Profundidad Cambio de Estrato (m)", 0.5, 10.0, 1.0, 0.1)
    
    with st.expander("Suelo 1 (Superior)", expanded=True):
        s1_name = st.text_input("Nombre S1", "Arcilla Blanda")
        s1_gamma = st.number_input("Gamma (kN/m³)", 10.0, 25.0, 17.0, key="s1g")
        s1_phi = st.number_input("Fricción Phi (°)", 0.0, 45.0, 20.0, key="s1p")
        s1_c = st.number_input("Cohesión c (kPa)", 0.0, 200.0, 15.0, key="s1c")
        s1_E = st.number_input("Módulo E (kPa)", 1000.0, 100000.0, 1800.0, key="s1e")
        s1_nu = st.number_input("Poisson nu", 0.1, 0.5, 0.35, key="s1n")
        
    with st.expander("Suelo 2 (Inferior)"):
        s2_name = st.text_input("Nombre S2", "Arena Densa")
        s2_gamma = st.number_input("Gamma (kN/m³)", 10.0, 25.0, 19.0, key="s2g")
        s2_phi = st.number_input("Fricción Phi (°)", 0.0, 45.0, 32.0, key="s2p")
        s2_c = st.number_input("Cohesión c (kPa)", 0.0, 200.0, 0.0, key="s2c")
        s2_E = st.number_input("Módulo E (kPa)", 1000.0, 100000.0, 8000.0, key="s2e")
        s2_nu = st.number_input("Poisson nu", 0.1, 0.5, 0.30, key="s2n")

    st.subheader("2. Cargas y Restricciones")
    CARGA_P = st.number_input("Carga Vertical P (kN)", 100.0, 5000.0, 800.0)
    FS_OBJETIVO = st.slider("FS Mínimo", 1.5, 4.0, 3.0, 0.1)
    ASENT_MAX = st.slider("Asentamiento Máx (cm)", 1.0, 10.0, 3.0, 0.5) / 100
    
    # NOTA: Se eliminaron los inputs de costos monetarios
    
    btn_calc = st.button("🚀 OPTIMIZAR VOLÚMENES")

# Diccionarios de Suelo Globales
SUELO_1 = {"gamma": s1_gamma, "phi": s1_phi, "c": s1_c, "E": s1_E, "nu": s1_nu, "nombre": s1_name}
SUELO_2 = {"gamma": s2_gamma, "phi": s2_phi, "c": s2_c, "E": s2_E, "nu": s2_nu, "nombre": s2_name}

# ==============================================================================
# 2. MOTOR DE CÁLCULO (GEOTECNIA)
# ==============================================================================
def factores_vesic(phi):
    rad = math.radians(phi)
    if phi > 0:
        Nq = (math.tan(math.radians(45 + phi/2)))**2 * math.exp(math.pi * math.tan(rad))
        Nc = (Nq - 1) / math.tan(rad)
        Ny = 2 * (Nq + 1) * math.tan(rad)
    else: # Caso phi=0
        Nq = 1.0; Nc = 5.14; Ny = 0.0
    return Nc, Nq, Ny

def calc_geotecnia(B, L, Df):
    # Determinar estrato de apoyo
    if Df < PROF_LIMIT: suelo = SUELO_1; suelo_bajo = SUELO_2
    else: suelo = SUELO_2; suelo_bajo = SUELO_2
        
    Nc, Nq, Ny = factores_vesic(suelo["phi"])
    
    # Factores de forma (Vesic)
    sc = 1 + (Nq/Nc)*(B/L) if suelo["c"] > 0 else 1
    sy = 1 - 0.4*(B/L)
    sq = 1 + (B/L)*math.tan(math.radians(suelo["phi"]))
    
    # Sobrecarga q (gamma * Df)
    if Df <= PROF_LIMIT:
        q_sobre = Df * SUELO_1["gamma"]
    else:
        q_sobre = (PROF_LIMIT * SUELO_1["gamma"]) + ((Df - PROF_LIMIT) * SUELO_2["gamma"])
    
    # Capacidad Última (q_ult)
    q_ult = (suelo["c"]*Nc*sc) + (q_sobre*Nq*sq) + (0.5*suelo["gamma"]*B*Ny*sy)
    q_act = CARGA_P / (B*L)
    
    # --- ASENTAMIENTOS ---
    # 1. Asentamiento Elástico (Inmediato)
    z_infl = 2*B
    E_calc = suelo["E"]
    
    # Ponderación de E si el bulbo toca el segundo estrato
    if Df < PROF_LIMIT and (PROF_LIMIT - Df) < z_infl:
        h1 = PROF_LIMIT - Df
        h2 = z_infl - h1
        if h2 > 0: E_calc = ((suelo["E"]*h1) + (suelo_bajo["E"]*h2)) / z_infl
            
    sett_elastico = q_act * B * ((1 - suelo["nu"]**2) / E_calc) * 0.88 # Factor forma rígido
    
    # 2. Asentamiento por Consolidación (Simplificado si es arcilla)
    sett_consolidacion = 0.0
    # Si el suelo de apoyo es arcilla (Phi < 5°) asumimos comportamiento cohesivo
    if suelo["phi"] < 5:
        # Cc estimado ~ 0.009(LL-10), asumimos un Cc/1+e0 genérico si no hay input
        # O usamos la formula m_v * delta_sigma * H
        mv = 1.0 / E_calc # Aproximación inversa del módulo E
        H_comprimible = min(z_infl, 5.0) # Espesor activo
        sett_consolidacion = mv * q_act * H_comprimible 
        
    sett_total = sett_elastico + sett_consolidacion
    
    return q_ult, q_act, sett_total

def funcion_fitness(ind):
    """Función Objetivo: Minimizar Volumen de Material"""
    B, L, Df = ind
    # Restricciones geométricas estrictas
    if B > L or B < 0.8 or L > 5.0 or Df < 0.5 or Df > 5.0: return 1e9
    
    q_ult, q_act, sett = calc_geotecnia(B, L, Df)
    
    if q_act <= 0: return 1e9
    
    # Penalizaciones (Castigo severo si no cumple ingeniería)
    if (q_ult/q_act) < FS_OBJETIVO: return 1e7
    if sett > ASENT_MAX: return 1e6
    
    # Objetivo: Minimizar Volumen Concreto (Principal) + Excavación (Secundario)
    vol_conc = B * L * 0.5 # Asumimos zapata h=0.5m
    vol_exc = B * L * Df
    
    # El fitness es el volumen ponderado
    # Damos más peso al concreto (material caro)
    return vol_conc + (0.1 * vol_exc)

# ==============================================================================
# 3. APP PRINCIPAL
# ==============================================================================
st.title("🏗️ Diseño y Optimización de Zapatas Aisladas")
st.markdown("**Objetivo:** Encontrar las dimensiones que minimizan la cantidad de material ($m^3$ Concreto) cumpliendo factores de seguridad y asentamientos.")

tab1, tab2, tab3 = st.tabs(["📊 Optimización & Cantidades", "🌍 Análisis de Sensibilidad", "📝 Ecuaciones"])

if btn_calc:
    with st.spinner('Iterando geometría para minimizar volúmenes...'):
        # --- ALGORITMO GENÉTICO ---
        pob = [[random.uniform(1,4.5), random.uniform(1,4.5), random.uniform(0.5,5.0)] for _ in range(300)]
        
        for gen in range(40):
            scores = [(ind, funcion_fitness(ind)) for ind in pob]
            scores.sort(key=lambda x: x[1])
            padres = [x[0] for x in scores[:50]]
            nueva_pob = padres[:]
            while len(nueva_pob) < 300:
                p1, p2 = random.choice(padres), random.choice(padres)
                hijo = p1[:random.randint(0,2)] + p2[random.randint(0,2):]
                if random.random() < 0.2:
                    idx = random.randint(0,2)
                    hijo[idx] += random.uniform(-0.3, 0.3)
                if hijo[0] > hijo[1]: hijo[0], hijo[1] = hijo[1], hijo[0] 
                nueva_pob.append(hijo)
            pob = nueva_pob

        # --- SELECCIÓN FINAL ---
        raw_scores = sorted([(ind, funcion_fitness(ind)) for ind in pob], key=lambda x: x[1])
        top_5 = []
        unique = set()
        
        for item in raw_scores:
            if item[1] > 1e5: continue
            h = f"{item[0][0]:.1f}-{item[0][2]:.1f}"
            if h not in unique:
                q_u, q_a, s = calc_geotecnia(*item[0])
                vol_c = item[0][0] * item[0][1] * 0.5
                vol_e = item[0][0] * item[0][1] * item[0][2]
                top_5.append({
                    "B": item[0][0], "L": item[0][1], "Df": item[0][2],
                    "Vol_Conc": vol_c, "Vol_Exc": vol_e, 
                    "FS": q_u/q_a, "Sett": s, "Qact": q_a
                })
                unique.add(h)
            if len(top_5) >= 5: break
            
        if not top_5:
            st.error("No se encontraron soluciones factibles.")
            st.stop()

        best = top_5[0]

    # ==========================================================================
    # PESTAÑA 1: RESULTADOS
    # ==========================================================================
    with tab1:
        # Métricas de Cantidades
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vol. Concreto (m³)", f"{best['Vol_Conc']:.2f}")
        c2.metric("Vol. Excavación (m³)", f"{best['Vol_Exc']:.2f}")
        c3.metric("Geometría (BxL)", f"{best['B']:.2f} x {best['L']:.2f} m")
        c4.metric("Desplante (Df)", f"{best['Df']:.2f} m")

        # Gráfico Perfil
        st.subheader("Esquema de la Solución Óptima")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Suelo
        r1 = patches.Rectangle((-2, 0), 14, PROF_LIMIT, fc='#D7BDE2', alpha=0.5, label=f'{SUELO_1["nombre"]}')
        r2 = patches.Rectangle((-2, PROF_LIMIT), 14, 6, fc='#F9E79F', alpha=0.5, label=f'{SUELO_2["nombre"]}')
        ax.add_patch(r1); ax.add_patch(r2)
        ax.axhline(PROF_LIMIT, color='brown', ls='--', lw=1)
        
        # Zapata
        xc = 5
        ax.add_patch(patches.Rectangle((xc-0.2, 0), 0.4, best['Df'], fc='gray', alpha=0.6)) # Columna
        ax.add_patch(patches.Rectangle((xc-best['B']/2, best['Df']), best['B'], 0.5, fc='#2C3E50', ec='black')) # Zapata
        
        ax.text(xc, best['Df']+0.25, f"B={best['B']:.2f}m", color='white', ha='center', va='center')
        ax.text(-1.5, PROF_LIMIT, f"Cambio Estrato ({PROF_LIMIT}m)", va='bottom', fontsize=8, color='brown')
        
        ax.set_ylim(max(PROF_LIMIT+2, best['Df']+2), 0)
        ax.set_xlim(0, 10)
        ax.set_ylabel("Profundidad (m)")
        ax.legend(loc='lower right')
        st.pyplot(fig)

        # Tabla
        st.subheader("Mejores Alternativas (Cantidades)")
        df_top = pd.DataFrame(top_5)
        df_top['Sett_cm'] = df_top['Sett'] * 100
        st.dataframe(
            df_top[['B', 'L', 'Df', 'Vol_Conc', 'Vol_Exc', 'FS', 'Sett_cm']]
            .style.background_gradient(subset=['Vol_Conc'], cmap='Greens_r'), 
            use_container_width=True
        )

    # ==========================================================================
    # PESTAÑA 2: SENSIBILIDAD
    # ==========================================================================
    with tab2:
        st.subheader("Impacto de la Geometría en los Volúmenes")
        
        widths = np.linspace(1.0, 4.0, 50)
        vols_c = []
        q_adms = []
        
        for w in widths:
            vol = w * w * 0.5
            q_u, _, _ = calc_geotecnia(w, w, best['Df'])
            vols_c.append(vol)
            q_adms.append(q_u/FS_OBJETIVO)

        fig2, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.plot(widths, vols_c, 'g-', lw=2, label='Volumen Concreto')
        ax1.set_xlabel("Ancho Zapata B (m)")
        ax1.set_ylabel("Volumen Concreto (m³)", color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        
        ax2 = ax1.twinx()
        ax2.plot(widths, q_adms, 'b--', lw=2, label='Q Admisible')
        ax2.set_ylabel("Q adm (kPa)", color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        plt.title("Relación Tamaño vs Volumen de Material")
        st.pyplot(fig2)

    # ==========================================================================
    # PESTAÑA 3: MEMORIA
    # ==========================================================================
    with tab3:
        st.subheader("Formulación Técnica")
        
        st.markdown("**1. Capacidad Portante (Vesic)**")
        st.latex(r"q_{ult} = c N_c s_c + q N_q s_q + 0.5 \gamma B N_\gamma s_\gamma")
        
        st.markdown("**2. Asentamiento Elástico (Inmediato)**")
        st.latex(r"S_e = q_{act} B \frac{1-\nu^2}{E} I_f")
        st.caption("Donde If = 0.88 para zapata rígida cuadrada.")
        
        st.markdown("**3. Consolidación (Si aplica)**")
        st.latex(r"S_c = \sum m_v \cdot \Delta \sigma \cdot H")
        st.info("La optimización busca minimizar la función objetivo: $F = Vol_{concreto} + 0.1 \cdot Vol_{excavacion}$")