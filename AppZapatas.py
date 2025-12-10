# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# 0. CONFIGURACIÓN Y ESTILOS
# ==============================================================================
st.set_page_config(page_title="ToolZapatas v3.2 Safe", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2C3E50; font-family: 'Helvetica', sans-serif; }
    .stMetric { background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
    .bloque-teoria { background-color: #e8f6f3; padding: 15px; border-radius: 5px; border-left: 5px solid #1abc9c; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. GESTIÓN DE ESTADO (SESSION STATE)
# ==============================================================================
if 'proyecto' not in st.session_state:
    st.session_state['proyecto'] = {
        "cargas": {"P": 50.0},
        "geometria": {"B": 1.5, "L": 1.5, "Df": 1.5, "h": 0.4},
        "suelo": pd.DataFrame(),
    }

# ==============================================================================
# 2. BIBLIOTECA DE FUNCIONES CIENTÍFICAS
# ==============================================================================

def dibujar_mecanismo_meyerhof():
    """Genera una ilustración teórica del mecanismo de falla."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axhline(0, color='black', linewidth=1) # Suelo
    
    # Zapata
    B, Df = 2.0, 1.0
    rect = patches.Rectangle((-B/2, 0), B, 0.5, facecolor='#95a5a6', edgecolor='black')
    ax.add_patch(rect)
    ax.text(0, 0.25, "Zapata", ha='center', color='white', weight='bold')

    # Zona I: Activa
    ax.add_patch(patches.Polygon([(-B/2, 0), (B/2, 0), (0, -1.5)], closed=True, facecolor='#d7bde2', alpha=0.6, label='Zona I: Activa'))
    # Zona II: Radial (Simplificada)
    ax.add_patch(patches.Wedge((B/2, 0), 2.5, 270, 360, facecolor='#f9e79f', alpha=0.6, label='Zona II: Radial'))
    ax.add_patch(patches.Wedge((-B/2, 0), 2.5, 180, 270, facecolor='#f9e79f', alpha=0.6))
    # Zona III: Pasiva
    ax.add_patch(patches.Polygon([(B/2, 0), (3, 0), (2.1, -1.8)], closed=True, facecolor='#aed6f1', alpha=0.6, label='Zona III: Pasiva'))
    ax.add_patch(patches.Polygon([(-B/2, 0), (-3, 0), (-2.1, -1.8)], closed=True, facecolor='#aed6f1', alpha=0.6))

    ax.set_xlim(-4, 4); ax.set_ylim(-3, 1); ax.axis('off')
    ax.legend(loc='lower center', ncol=3, fontsize='small')
    ax.set_title("Mecanismo Teórico de Falla (Meyerhof)", fontsize=10)
    return fig

def calc_incremento_esfuerzo(P, B, L, z):
    """Método 2:1 Boussinesq simplificado."""
    if z <= 0: return P / (B*L)
    area_z = (B + z) * (L + z)
    return P / area_z

def calc_asentamiento_elastico(d_sigma, B, Es_ton_m2, nu):
    """Asentamiento inmediato (Schmertmann simplificado centro)."""
    if Es_ton_m2 <= 0: return 0.0
    # Si = q * B * ((1-nu^2)/Es) * If (If approx 1.12 para cuadrado flexible)
    Si = d_sigma * B * ((1 - nu**2) / Es_ton_m2) * 1.12 
    return Si

def calc_consolidacion(H, e0, Cc, sigma_0, delta_sigma):
    """Terzaghi unidimensional."""
    if Cc <= 0 or H <= 0 or sigma_0 <= 0: return 0.0
    val = (sigma_0 + delta_sigma) / sigma_0
    if val <= 0: return 0.0
    return (Cc * H / (1 + e0)) * np.log10(val)

# ==============================================================================
# 3. MÓDULOS DE INTERFAZ
# ==============================================================================

def modulo_sidebar():
    """Configuración Global"""
    st.sidebar.header("🎛️ Parámetros Globales")
    
    # IMPORTANTE: Usamos 'proj' para referenciar el estado. 
    # NO uses 'st = ...' porque rompería la librería streamlit.
    proj = st.session_state['proyecto']
    
    proj['cargas']['P'] = st.sidebar.number_input("Carga Axial P (Ton)", 1.0, 5000.0, 50.0)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Geometría")
    proj['geometria']['B'] = st.sidebar.number_input("Ancho B (m)", 0.5, 10.0, 1.5)
    proj['geometria']['L'] = st.sidebar.number_input("Largo L (m)", 0.5, 10.0, 1.5)
    proj['geometria']['Df'] = st.sidebar.number_input("Desplante Df (m)", 0.0, 10.0, 1.5)
    proj['geometria']['h'] = st.sidebar.number_input("Espesor h (m)", 0.3, 2.0, 0.4)

def modulo_estratigrafia():
    """Tabla de Suelos"""
    st.subheader("1. Caracterización del Subsuelo")
    st.info("Define las capas de arriba hacia abajo.")

    if st.session_state['proyecto']['suelo'].empty:
        # Datos iniciales de ejemplo
        data = [
            {"Tipo": "Relleno", "Esp (m)": 1.0, "Gamma (T/m3)": 1.6, "phi": 28, "c": 0, "Es (T/m2)": 800, "nu": 0.3, "Cc": 0.0, "e0": 0.5},
            {"Tipo": "Arcilla Blanda", "Esp (m)": 4.0, "Gamma (T/m3)": 1.7, "phi": 0, "c": 3, "Es (T/m2)": 300, "nu": 0.45, "Cc": 0.3, "e0": 1.1},
            {"Tipo": "Arena Densa", "Esp (m)": 5.0, "Gamma (T/m3)": 1.95, "phi": 34, "c": 0, "Es (T/m2)": 3000, "nu": 0.3, "Cc": 0.0, "e0": 0.6},
        ]
        df_init = pd.DataFrame(data)
    else:
        df_init = st.session_state['proyecto']['suelo']

    df_edit = st.data_editor(df_init, num_rows="dynamic", use_container_width=True)
    st.session_state['proyecto']['suelo'] = df_edit
    
    # Dibujo simple del perfil
    if not df_edit.empty:
        z = 0
        fig, ax = plt.subplots(figsize=(8, 1.5))
        for i, r in df_edit.iterrows():
            try:
                esp = float(r['Esp (m)'])
                ax.barh(0, esp, left=z, height=0.5, align='center', edgecolor='black', alpha=0.5)
                ax.text(z + esp/2, 0, str(r['Tipo']), ha='center', va='center', fontsize=8)
                z += esp
            except: pass
        ax.set_xlim(0, z if z>0 else 1); ax.set_yticks([]); ax.set_xlabel("Profundidad Acumulada (m)")
        ax.invert_xaxis()
        st.pyplot(fig)

def modulo_capacidad():
    """Cálculo Meyerhof"""
    st.subheader("2. Capacidad Portante (Meyerhof)")
    
    proj = st.session_state['proyecto']
    df = proj['suelo']
    if df.empty: st.error("Faltan datos de suelo."); return

    Df = proj['geometria']['Df']
    B = proj['geometria']['B']
    L = proj['geometria']['L']
    
    # Buscar estrato en la punta
    z_acum = 0
    suelo_punta = df.iloc[-1]
    q_sobrecarga = 0 # gamma * Df
    
    try:
        # Cálculo preciso de q (sobrecarga) acumulando gammas
        for _, r in df.iterrows():
            dz = float(r['Esp (m)'])
            z_top = z_acum
            z_bot = z_acum + dz
            
            if Df >= z_bot:
                q_sobrecarga += float(r['Gamma (T/m3)']) * dz
            elif z_top <= Df < z_bot:
                q_sobrecarga += float(r['Gamma (T/m3)']) * (Df - z_top)
                suelo_punta = r
                break
            z_acum += dz
    except Exception as e:
        st.error(f"Error leyendo datos numéricos del suelo: {e}")
        return

    phi = float(suelo_punta['phi'])
    c = float(suelo_punta['c'])
    gamma_punta = float(suelo_punta['Gamma (T/m3)'])

    # Factores Meyerhof
    rad = np.radians(phi)
    kp = np.tan(np.radians(45) + rad/2)**2
    
    if phi < 0.1: # Arcilla phi=0
        Nq, Nc, Ny = 1.0, 5.14, 0.0
    else:
        Nq = np.exp(np.pi * np.tan(rad)) * kp
        Nc = (Nq - 1) / np.tan(rad)
        Ny = (Nq - 1) * np.tan(1.4 * rad)

    # Factores de Forma
    sc = 1 + 0.2 * kp * (B/L)
    sq = 1 + 0.1 * kp * (B/L) if phi > 10 else 1.0
    sy = sq
    
    # Factores Profundidad (Simplificado Df/B < 1)
    k_d = Df/B if Df/B <=1 else 1.0
    dc = 1 + 0.4 * np.sqrt(kp) * k_d
    dq = 1 + 0.1 * np.sqrt(kp) * k_d
    dy = dq

    # Ecuación General
    q_ult = (c * Nc * sc * dc) + (q_sobrecarga * Nq * sq * dq) + (0.5 * gamma_punta * B * Ny * sy * dy)
    
    P_act = proj['cargas']['P']
    Area = B * L
    Peso_Zapata = Area * proj['geometria']['h'] * 2.4
    q_act = (P_act + Peso_Zapata) / Area
    
    FS = q_ult / q_act if q_act > 0 else 999

    # Resultados UI
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.metric("Q Última", f"{q_ult:.2f} Ton/m²")
        st.metric("Q Actuante", f"{q_act:.2f} Ton/m²", delta="Incluye P.Propio")
        st.metric("FS Global", f"{FS:.2f}", delta="OK" if FS>=3.0 else "Revisar")
    
    with c2:
        st.pyplot(dibujar_mecanismo_meyerhof())
        st.caption(f"Apoyado en: {suelo_punta['Tipo']} (Phi={phi}°, c={c})")

def modulo_asentamientos():
    """Cálculo de Asentamientos"""
    st.subheader("3. Estimación de Asentamientos")
    
    proj = st.session_state['proyecto']
    df = proj['suelo']
    if df.empty: return

    P = proj['cargas']['P']
    B, L, Df = proj['geometria']['B'], proj['geometria']['L'], proj['geometria']['Df']
    
    z_abs_top = 0
    total_Se = 0
    total_Sc = 0
    tabla_res = []

    for _, r in df.iterrows():
        # --- BLINDAJE DE VARIABLES ---
        # Inicializamos a cero ANTES de cualquier cálculo para evitar UnboundLocalError
        Se = 0.0
        Sc = 0.0
        # -----------------------------
        
        try:
            esp = float(r['Esp (m)'])
            z_abs_bot = z_abs_top + esp
            
            # Analizamos si el estrato está debajo de la zapata (al menos parcialmente)
            if z_abs_bot > Df:
                # Definir tramo efectivo del estrato bajo la zapata
                z_start = max(z_abs_top, Df)
                z_end = z_abs_bot
                H_eff = z_end - z_start
                
                if H_eff > 0:
                    # Punto medio de este sub-estrato (desde la superficie)
                    z_mid_abs = z_start + H_eff/2
                    # Distancia desde la base de la zapata hasta el punto medio
                    z_local = z_mid_abs - Df
                    
                    # 1. Delta Sigma
                    d_sigma = calc_incremento_esfuerzo(P, B, L, z_local)
                    
                    # 2. Sigma Inicial (Geostático)
                    sigma_0 = 0
                    z_temp = 0
                    for _, r_inner in df.iterrows():
                        esp_inner = float(r_inner['Esp (m)'])
                        gam_inner = float(r_inner['Gamma (T/m3)'])
                        if z_temp + esp_inner < z_mid_abs:
                            sigma_0 += esp_inner * gam_inner
                            z_temp += esp_inner
                        else:
                            remaining = z_mid_abs - z_temp
                            sigma_0 += remaining * gam_inner
                            break
                    
                    # 3. Asentamientos
                    Es = float(r['Es (T/m2)'])
                    nu = float(r['nu'])
                    Se = calc_asentamiento_elastico(d_sigma, B, Es, nu)
                    
                    Cc = float(r['Cc'])
                    if Cc > 0:
                        e0 = float(r['e0'])
                        Sc = calc_consolidacion(H_eff, e0, Cc, sigma_0, d_sigma)

                    total_Se += Se
                    total_Sc += Sc
                    
                    tabla_res.append({
                        "Estrato": r['Tipo'],
                        "H efec (m)": round(H_eff, 2),
                        "σ'0 (T/m2)": round(sigma_0, 2),
                        "Δσ (T/m2)": round(d_sigma, 2),
                        "S. Elástico (cm)": round(Se*100, 2),
                        "S. Consol (cm)": round(Sc*100, 2)
                    })
            
            z_abs_top = z_abs_bot
            
        except Exception as e:
            st.warning(f"Error calculando estrato {r.get('Tipo', '?')}: {e}")

    st.table(pd.DataFrame(tabla_res))
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Elástico (Inmediato)", f"{total_Se*100:.2f} cm")
    k2.metric("Consolidación (Largo Plazo)", f"{total_Sc*100:.2f} cm")
    k3.metric("Asentamiento Total", f"{(total_Se+total_Sc)*100:.2f} cm")

# ==============================================================================
# 4. RUN
# ==============================================================================
def main():
    st.title("🏗️ ToolZapatas v3.2 Modular")
    modulo_sidebar()
    
    tabs = st.tabs(["🌍 1. Estratigrafía", "⚙️ 2. Capacidad Portante", "📉 3. Asentamientos"])
    with tabs[0]: modulo_estratigrafia()
    with tabs[1]: modulo_capacidad()
    with tabs[2]: modulo_asentamientos()

if __name__ == "__main__":
    main()
