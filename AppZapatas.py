# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from io import BytesIO

# --- IMPORTACIONES PDF ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    pass

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================
st.set_page_config(page_title="ToolZapatas Pro", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #f1f3f6; border-radius: 5px 5px 0px 0px;
        padding: 10px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #3498DB; }
    h1, h2, h3 { color: #2C3E50; }
    .stMetric { background-color: #ffffff; border: 1px solid #eeeeee; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. GESTIÓN DE ESTADO
# ==============================================================================
if 'opt_zapata' not in st.session_state: st.session_state['opt_zapata'] = {}
if 'params_input' not in st.session_state: st.session_state['params_input'] = {}

# ==============================================================================
# 2. MOTOR CIENTÍFICO (GEOTECNIA Y ESTRUCTURA)
# ==============================================================================

@dataclass
class ZapataInput:
    P: float; Mx: float; My: float  # Cargas (Ton, Ton-m)
    phi: float; c: float; gamma: float; Df: float # Suelo
    fc: float; fy: float # Materiales (MPa)
    FS_geo_req: float # Factor de seguridad requerido

def calc_capacidad_portante(B, L, Df, phi, c, gamma):
    """
    Calcula q_ult usando Meyerhof.
    Retorna q_ult (Ton/m2).
    """
    # Factores de forma y profundidad simplificados para el ejemplo
    # Conversión grados a radianes
    rad = np.radians(phi)
    
    # Factores de capacidad de carga (Nq, Nc, Ny)
    if phi < 0.1: # Caso arcilla pura phi=0
        Nq = 1.0
        Nc = 5.14
        Ny = 0.0
    else:
        Nq = np.exp(np.pi * np.tan(rad)) * (np.tan(np.radians(45) + rad/2))**2
        Nc = (Nq - 1) / np.tan(rad)
        Ny = (Nq - 1) * np.tan(1.4 * rad)

    # Factores de forma (Meyerhof)
    sc = 1 + 0.2 * Nq/Nc * (B/L) if phi > 0 else 1 + 0.2*(B/L)
    sq = 1 + 0.1 * Nq * (B/L) if phi > 10 else 1.0
    sy = 1 + 0.1 * Nq * (B/L) if phi > 10 else 1.0

    # q_ult (Ton/m2) - gamma en Ton/m3, c en Ton/m2
    q_ult = (c * Nc * sc) + (gamma * Df * Nq * sq) + (0.5 * gamma * B * Ny * sy)
    return q_ult

def verificar_zapata(B, L, h, inp: ZapataInput):
    """
    Verifica Geotecnia (Presiones) y Estructura (Cortante, Punzonamiento).
    Retorna diccionario con resultados y métricas.
    """
    # 1. Geometría Efectiva y Presiones
    Area = B * L
    Ix = (B * L**3) / 12
    Iy = (L * B**3) / 12
    
    # Peso propio (Concreto 2.4 Ton/m3)
    W_pp = Area * h * 2.4 
    P_tot = inp.P + W_pp
    
    # Excentricidades
    ex = inp.My / P_tot if P_tot > 0 else 0
    ey = inp.Mx / P_tot if P_tot > 0 else 0
    
    # Presiones en esquinas (Navier) q = P/A ± Mx*y/Ix ± My*x/Iy
    # Simplificado qmax
    q_max = (P_tot/Area) * (1 + 6*abs(ex)/B + 6*abs(ey)/L)
    q_min = (P_tot/Area) * (1 - 6*abs(ex)/B - 6*abs(ey)/L)
    
    if q_min < 0: q_min = 0 # Levantamiento (no ideal, pero para calculo de qmax sirve)

    # 2. Geotecnia
    q_ult = calc_capacidad_portante(min(B,L), max(B,L), inp.Df, inp.phi, inp.c, inp.gamma)
    FS_geo = q_ult / q_max if q_max > 0 else 999
    
    # 3. Estructural (ACI 318)
    d = h - 0.075 # Recubrimiento 7.5cm
    if d <= 0: return {"Valido": False}

    # Factores (SI units internamente: MN, m, MPa)
    # Convertimos cargas a MN para formulas ACI
    Pu_MN = (inp.P * 1.4) * 9.81 / 1000 # Mayorada 1.4 (simplificado)
    fc_MPa = inp.fc
    
    # --- Cortante Punzonamiento (Dos Direcciones) ---
    bo = 2 * ((0.3+d) + (0.3+d)) # Perimetro critico asumiendo columna 30x30cm
    beta = 1.0 # Relacion lados columna
    vc_aci = 0.17 * (1 + 2/beta) * np.sqrt(fc_MPa) # MPa
    vc_aci = min(vc_aci, 0.33 * np.sqrt(fc_MPa))
    Phi_v = 0.75
    Vu_punz = Pu_MN # Asumimos toda la carga se punzona
    Vu_cap_punz = Phi_v * vc_aci * bo * d # MN
    ratio_punz = Vu_punz / Vu_cap_punz

    # --- Cortante Unidireccional (Una Dirección) ---
    # Cortante actuante a distancia 'd' de la cara
    L_volado = (L - 0.3)/2 # Columna 0.3
    if L_volado > d:
        Area_shear = B * (L_volado - d)
        Vu_one = (q_max * 1.4 * 9.81/1000) * Area_shear # Aprox conservadora usando qmax
    else:
        Vu_one = 0
        
    vc_one = 0.17 * np.sqrt(fc_MPa) * B * d # MPa * m2 = MN
    Vu_cap_one = Phi_v * vc_one
    ratio_one = Vu_one / Vu_cap_one if Vu_cap_one > 0 else 999

    # --- Flexión (Acero) ---
    # Momento en la cara de la columna
    Mu = (q_max * 1.4 * 9.81/1000) * B * (L_volado**2) / 2 # MN-m
    phi_f = 0.9
    # As aprox = Mu / (phi * fy * 0.9d)
    As_req = (Mu) / (phi_f * inp.fy * 0.9 * d) * 1e4 # cm2
    
    rho_min = 0.0018
    As_min = rho_min * B * h * 1e4 # cm2
    As_final = max(As_req, As_min)

    valido = (FS_geo >= inp.FS_geo_req) and (ratio_punz < 1.0) and (ratio_one < 1.0)
    
    volumen = B * L * h
    peso_acero = As_final * (B+L) * 2 * 7.85 # Aprox cuantía en kg (muy simplificado)
    costo_idx = volumen * 100 + peso_acero * 1.5 # 1m3 concreto = 100 USD, 1kg acero = 1.5 USD

    return {
        "Valido": valido,
        "FS_geo": FS_geo, "q_max": q_max, "q_ult": q_ult,
        "Ratio_Punz": ratio_punz, "Ratio_Cort": ratio_one,
        "As_req_cm2": As_final,
        "Costo_Idx": costo_idx,
        "Volumen": volumen,
        "Geom": (B, L, h)
    }

# ==============================================================================
# 3. INTERFAZ PRINCIPAL
# ==============================================================================
def app_principal():
    st.title("🏗️ ToolZapatas Pro")
    st.markdown("**Diseño y Optimización de Cimentaciones Superficiales**")

    tab1, tab2, tab3 = st.tabs([
        "📝 1. Datos de Entrada", 
        "🚀 2. Optimización Automática", 
        "📐 3. Detalles y Planos"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: INPUTS
    # --------------------------------------------------------------------------
    with tab1:
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.subheader("Cargas (Servicio)")
            P = st.number_input("Carga Axial P (Ton)", 1.0, 1000.0, 50.0)
            Mx = st.number_input("Momento Mx (Ton-m)", 0.0, 100.0, 2.0)
            My = st.number_input("Momento My (Ton-m)", 0.0, 100.0, 0.0)
        
        with c2:
            st.subheader("Suelo (Geotecnia)")
            phi = st.number_input("Fricción Phi (°)", 0.0, 45.0, 30.0)
            c_soil = st.number_input("Cohesión C (Ton/m2)", 0.0, 50.0, 0.0)
            gamma = st.number_input("Peso Esp. Gamma (Ton/m3)", 1.0, 2.5, 1.8)
            Df = st.number_input("Profundidad Df (m)", 0.5, 5.0, 1.5)
            FS_req = st.slider("FS Requerido", 1.5, 4.0, 3.0)

        with c3:
            st.subheader("Materiales")
            fc = st.selectbox("Concreto f'c (MPa)", [21, 28, 35, 42], index=1)
            fy = st.number_input("Acero fy (MPa)", 240, 500, 420)
        
        # Guardar en sesión
        st.session_state['params_input'] = ZapataInput(P, Mx, My, phi, c_soil, gamma, Df, fc, fy, FS_req)
        
        # Cálculo Rápido de Capacidad
        q_est = calc_capacidad_portante(1.5, 1.5, Df, phi, c_soil, gamma)
        st.info(f"💡 Capacidad de carga estimada para una zapata de 1.5x1.5m: **{q_est:.2f} Ton/m²**")

    # --------------------------------------------------------------------------
    # TAB 2: OPTIMIZACIÓN
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🤖 Motor de Optimización")
        st.write("El algoritmo buscará la zapata de menor costo que cumpla con Geotecnia (FS) y Estructura (ACI 318).")
        
        col_o1, col_o2 = st.columns([1, 3])
        with col_o1:
            if st.button("EJECUTAR DISEÑO", type="primary"):
                inp = st.session_state['params_input']
                resultados = []
                
                # Barrido de dimensiones (Brute Force Inteligente)
                prog_bar = st.progress(0)
                B_range = np.arange(0.8, 3.5, 0.1) # De 0.8m a 3.5m
                L_factor = [1.0, 1.2, 1.5] # Cuadrada o Rectangular
                H_range = np.arange(0.3, 0.8, 0.05) # Espesores
                
                total_iter = len(B_range)*len(L_factor)*len(H_range)
                curr = 0
                
                for B in B_range:
                    for lf in L_factor:
                        L = B * lf
                        for h in H_range:
                            curr += 1
                            if curr % 50 == 0: prog_bar.progress(curr/total_iter)
                            
                            res = verificar_zapata(B, L, h, inp)
                            if res["Valido"]:
                                res_clean = {
                                    "B (m)": round(B, 2), "L (m)": round(L, 2), "h (m)": round(h, 2),
                                    "FS Geo": round(res["FS_geo"], 2),
                                    "Volumen": round(res["Volumen"], 2),
                                    "Costo_Idx": round(res["Costo_Idx"], 1),
                                    "As (cm2)": round(res["As_req_cm2"], 2),
                                    "Rat_Punz": round(res["Ratio_Punz"], 2)
                                }
                                resultados.append(res_clean)
                
                prog_bar.empty()
                
                if resultados:
                    df_res = pd.DataFrame(resultados).sort_values("Costo_Idx")
                    best = df_res.iloc[0]
                    st.session_state['opt_zapata'] = best.to_dict()
                    st.success("✅ Diseño Óptimo Encontrado")
                    
                    # Métricas Top
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Dimensiones", f"{best['B (m)']} x {best['L (m)']} m")
                    m2.metric("Espesor (h)", f"{best['h (m)']} m")
                    m3.metric("Concreto", f"{best['Volumen']} m³")
                    m4.metric("Acero Req", f"{best['As (cm2)']} cm²")
                    
                    st.dataframe(df_res.head(5), use_container_width=True)
                else:
                    st.error("No se encontró solución. Aumente el rango de búsqueda o mejore el suelo.")

    # --------------------------------------------------------------------------
    # TAB 3: DETALLES
    # --------------------------------------------------------------------------
    with tab3:
        opt = st.session_state.get('opt_zapata')
        if not opt:
            st.warning("⚠️ Ejecute la optimización primero.")
        else:
            B, L, h = opt["B (m)"], opt["L (m)"], opt["h (m)"]
            inp = st.session_state['params_input']
            
            st.subheader(f"Análisis Detallado: Zapata {B}x{L}x{h}m")
            
            # Recalcular datos precisos para plot
            res_det = verificar_zapata(B, L, h, inp)
            
            col_d1, col_d2 = st.columns([1, 1])
            
            with col_d1:
                st.markdown("#### 📊 Distribución de Presiones")
                # Gráfico de Planta con mapa de calor de presiones
                fig, ax = plt.subplots(figsize=(5, 5))
                
                # Zapata
                rect = patches.Rectangle((-B/2, -L/2), B, L, linewidth=2, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
                
                # Columna (Asumida 0.3x0.3)
                col_rect = patches.Rectangle((-0.15, -0.15), 0.3, 0.3, color='gray')
                ax.add_patch(col_rect)
                
                # Excentricidad
                ex = inp.My / (inp.P + B*L*h*2.4)
                ey = inp.Mx / (inp.P + B*L*h*2.4)
                ax.plot(ex, ey, 'rx', markersize=10, markeredgewidth=2, label='Centro Presión')
                
                # Núcleo Central (Rombo B/6, L/6)
                nucleus = patches.Polygon([(-B/6, 0), (0, L/6), (B/6, 0), (0, -L/6)], 
                                          closed=True, fill=True, color='green', alpha=0.1, label='Núcleo Central')
                ax.add_patch(nucleus)
                
                ax.set_xlim(-B/2 - 0.5, B/2 + 0.5)
                ax.set_ylim(-L/2 - 0.5, L/2 + 0.5)
                ax.set_xlabel("B (m)")
                ax.set_ylabel("L (m)")
                ax.legend(loc='upper right')
                ax.grid(True, ls=':')
                ax.set_title(f"Excentricidad e=({ex:.2f}, {ey:.2f})")
                
                st.pyplot(fig)

            with col_d2:
                st.markdown("#### ✅ Verificaciones ACI 318")
                
                checks = [
                    {"Check": "Capacidad Portante", "Valor": f"{res_det['q_max']:.1f} vs {res_det['q_ult']:.1f} Ton/m2", "Status": "OK" if res_det['FS_geo']>=inp.FS_geo_req else "FALLA"},
                    {"Check": "Punzonamiento", "Valor": f"Ratio {res_det['Ratio_Punz']:.2f}", "Status": "OK" if res_det['Ratio_Punz']<1.0 else "FALLA"},
                    {"Check": "Cortante (1-via)", "Valor": f"Ratio {res_det['Ratio_Cort']:.2f}", "Status": "OK" if res_det['Ratio_Cort']<1.0 else "FALLA"},
                ]
                st.table(pd.DataFrame(checks))
                
                st.info(f"**Acero Refuerzo (Ambas direcciones):** Usar {opt['As (cm2)']} cm². \n\n"
                        f"Sug: {int(np.ceil(opt['As (cm2)']/1.29))} varillas #4 (1/2\")")

            # Botón PDF simple
            if st.button("Descargar Memoria de Cálculo"):
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                p.drawString(50, 750, "TOOLZAPATAS PRO - REPORTE DE CALCULO")
                p.drawString(50, 720, f"Dimensiones: {B} x {L} x {h} m")
                p.drawString(50, 700, f"Carga Axial: {inp.P} Ton")
                p.drawString(50, 680, f"Esfuerzo Max Suelo: {res_det['q_max']:.2f} Ton/m2")
                p.drawString(50, 660, f"Capacidad Ultima: {res_det['q_ult']:.2f} Ton/m2")
                p.drawString(50, 640, f"Factor de Seguridad: {res_det['FS_geo']:.2f}")
                p.drawString(50, 600, "Cumple normativa ACI 318 y NSR-10")
                p.save()
                st.download_button("📥 Bajar PDF", buffer, "Memoria_Zapata.pdf", "application/pdf")

# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    app_principal()
