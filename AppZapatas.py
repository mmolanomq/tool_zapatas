# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# 0. CONFIGURACIÓN Y ESTILOS
# ==============================================================================
st.set_page_config(page_title="ToolZapatas v3.0 Modular", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2C3E50; font-family: 'Helvetica', sans-serif; }
    .stMetric { background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
    .bloque-teoria { background-color: #e8f6f3; padding: 15px; border-radius: 5px; border-left: 5px solid #1abc9c; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CLASES DE DATOS (STATE)
# ==============================================================================
# Usamos SessionState para mantener la data accesible entre módulos

if 'proyecto' not in st.session_state:
    st.session_state['proyecto'] = {
        "cargas": {"P": 50.0},
        "geometria": {"B": 1.5, "L": 1.5, "Df": 1.5, "h": 0.4},
        "suelo": pd.DataFrame(),
        "resultados_geo": {}
    }

# ==============================================================================
# 2. BIBLIOTECA DE FUNCIONES CIENTÍFICAS
# ==============================================================================

def dibujar_mecanismo_meyerhof():
    """Genera una ilustración teórica del mecanismo de falla."""
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Suelo
    ax.axhline(0, color='black', linewidth=1)
    
    # Zapata
    B = 2.0
    Df = 1.0
    rect = patches.Rectangle((-B/2, 0), B, 0.5, facecolor='#95a5a6', edgecolor='black')
    ax.add_patch(rect)
    ax.text(0, 0.25, "Zapata", ha='center')

    # Zona I: Cuña Activa (Triangular)
    triangle = patches.Polygon([(-B/2, 0), (B/2, 0), (0, -1.5)], closed=True, 
                               facecolor='#d7bde2', alpha=0.6, label='Zona I: Activa')
    ax.add_patch(triangle)

    # Zona II: Corte Radial (Log Espiral - Simplificado visualmente como abanico)
    wedge_r = patches.Wedge((B/2, 0), 2.5, 270, 360, facecolor='#f9e79f', alpha=0.6, label='Zona II: Radial')
    wedge_l = patches.Wedge((-B/2, 0), 2.5, 180, 270, facecolor='#f9e79f', alpha=0.6)
    ax.add_patch(wedge_r); ax.add_patch(wedge_l)

    # Zona III: Pasiva Rankine (Triangular hacia superficie)
    poly_pass_r = patches.Polygon([(B/2, 0), (3, 0), (2.1, -1.8)], closed=True, facecolor='#aed6f1', alpha=0.6, label='Zona III: Pasiva')
    poly_pass_l = patches.Polygon([(-B/2, 0), (-3, 0), (-2.1, -1.8)], closed=True, facecolor='#aed6f1', alpha=0.6)
    ax.add_patch(poly_pass_r); ax.add_patch(poly_pass_l)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 1)
    ax.axis('off')
    ax.legend(loc='lower center', ncol=3, fontsize='small')
    ax.set_title("Mecanismo Teórico de Falla (Meyerhof)", fontsize=10)
    return fig

def calc_incremento_esfuerzo(P, B, L, z):
    """Método 2:1 para distribución de esfuerzos."""
    if z == 0: return P / (B*L)
    area_z = (B + z) * (L + z)
    return P / area_z

def calc_asentamiento_elastico(q_neto, B, Es_ton_m2, nu):
    """Calcula asentamiento inmediato (centro zapata flexible)."""
    # Formula simplificada Si = q * B * ((1-nu^2)/Es) * If
    # Asumimos If aprox 1.0 para centro
    if Es_ton_m2 <= 0: return 0.0
    Si = q_neto * B * ((1 - nu**2) / Es_ton_m2) * 1.12 # 1.12 factor forma cuadrado
    return Si

def calc_consolidacion(H, e0, Cc, sigma_0, delta_sigma):
    """Formula Clásica de Terzaghi."""
    if Cc <= 0 or H <= 0: return 0.0
    return (Cc * H / (1 + e0)) * np.log10((sigma_0 + delta_sigma) / sigma_0)

# ==============================================================================
# 3. MÓDULOS DE LA INTERFAZ (UI BLOCKS)
# ==============================================================================

def modulo_sidebar():
    """Módulo 1: Entrada Global"""
    st.sidebar.header("🎛️ Parámetros Globales")
    
    # Acceso al estado
    st = st.session_state['proyecto']
    
    st['cargas']['P'] = st.sidebar.number_input("Carga Axial P (Ton)", 1.0, 1000.0, 50.0)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Geometría")
    st['geometria']['B'] = st.sidebar.number_input("Ancho B (m)", 0.5, 10.0, 1.5)
    st['geometria']['L'] = st.sidebar.number_input("Largo L (m)", 0.5, 10.0, 1.5)
    st['geometria']['Df'] = st.sidebar.number_input("Desplante Df (m)", 0.0, 10.0, 1.5)
    st['geometria']['h'] = st.sidebar.number_input("Espesor h (m)", 0.3, 2.0, 0.4)

def modulo_estratigrafia():
    """Módulo 2: Suelo (Datos extendidos para asentamientos)"""
    st.subheader("1. Caracterización Estratigráfica")
    st.info("Ingrese los parámetros. Para asentamientos, asegurese de llenar Es, Cc y e0.")

    # Datos por defecto con columnas extendidas
    if st.session_state['proyecto']['suelo'].empty:
        data = [
            {"Tipo": "Relleno", "Esp (m)": 1.0, "Gamma (T/m3)": 1.6, "phi": 28, "c": 0, "Es (T/m2)": 800, "nu": 0.3, "Cc": 0.0, "e0": 0.5},
            {"Tipo": "Arcilla", "Esp (m)": 4.0, "Gamma (T/m3)": 1.8, "phi": 5, "c": 4, "Es (T/m2)": 400, "nu": 0.4, "Cc": 0.25, "e0": 0.9},
            {"Tipo": "Arena", "Esp (m)": 5.0, "Gamma (T/m3)": 1.9, "phi": 32, "c": 0, "Es (T/m2)": 2000, "nu": 0.3, "Cc": 0.0, "e0": 0.6},
        ]
        df_init = pd.DataFrame(data)
    else:
        df_init = st.session_state['proyecto']['suelo']

    df_edit = st.data_editor(df_init, num_rows="dynamic", use_container_width=True)
    st.session_state['proyecto']['suelo'] = df_edit
    
    # Dibujar perfil rápido
    z_acum = 0
    fig, ax = plt.subplots(figsize=(8, 2))
    for _, r in df_edit.iterrows():
        ax.barh(-z_acum - r['Esp (m)']/2, 1, height=r['Esp (m)'], align='center', edgecolor='black', alpha=0.5)
        ax.text(0.5, -z_acum - r['Esp (m)']/2, r['Tipo'], ha='center', va='center')
        z_acum += r['Esp (m)']
    ax.set_yticks([]); ax.set_xlabel("Perfil"); ax.set_title("Visualización Rápida")
    st.pyplot(fig)

def modulo_capacidad_carga():
    """Módulo 3: Capacidad Portante (Meyerhof)"""
    st.subheader("2. Capacidad Portante (Meyerhof)")
    
    proj = st.session_state['proyecto']
    df = proj['suelo']
    if df.empty: st.warning("Defina suelo primero."); return

    # Buscar estrato en punta
    Df = proj['geometria']['Df']
    z_acum = 0; suelo_punta = df.iloc[-1]
    for _, r in df.iterrows():
        if z_acum <= Df < (z_acum + r['Esp (m)']): suelo_punta = r; break
        z_acum += r['Esp (m)']

    # Cálculos
    B, L = proj['geometria']['B'], proj['geometria']['L']
    phi = suelo_punta['phi']; c = suelo_punta['c']; gamma = suelo_punta['Gamma (T/m3)']
    
    # --- Lógica Meyerhof (resumida para brevedad, igual a v2.0) ---
    rad = np.radians(phi)
    Nq = np.exp(np.pi*np.tan(rad))*(np.tan(np.radians(45)+rad/2))**2
    Nc = (Nq-1)/np.tan(rad) if phi>0 else 5.14
    Ny = (Nq-1)*np.tan(1.4*rad)
    
    sc = 1+0.2*(B/L); sq = 1+0.1*np.sqrt(Nq)*(B/L) if phi>10 else 1.0; sy = sq
    q_ult = (c*Nc*sc) + (gamma*Df*Nq*sq) + (0.5*gamma*B*Ny*sy)
    
    q_act = proj['cargas']['P'] / (B*L)
    FS = q_ult / q_act
    
    # --- Layout Resultados ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Resultados")
        st.metric("Q Última", f"{q_ult:.2f} Ton/m²")
        st.metric("Q Actuante", f"{q_act:.2f} Ton/m²")
        st.metric("Factor de Seguridad", f"{FS:.2f}", delta="OK" if FS>=3 else "BAJO")
    
    with c2:
        st.markdown("#### Mecanismo de Falla")
        st.pyplot(dibujar_mecanismo_meyerhof()) # Llamada a la ilustración teórica
        with st.expander("Ver Ecuaciones"):
            st.latex(r"q_{ult} = c N_c s_c + q N_q s_q + 0.5 \gamma B N_\gamma s_\gamma")

def modulo_asentamientos():
    """Módulo 4: Asentamientos (NUEVO)"""
    st.subheader("3. Análisis de Asentamientos")
    
    proj = st.session_state['proyecto']
    df = proj['suelo']
    if df.empty: return

    P = proj['cargas']['P']
    B = proj['geometria']['B']; L = proj['geometria']['L']; Df = proj['geometria']['Df']
    q_neto = P / (B*L) # Simplificado

    st.markdown("##### Desglose por Estrato (Bajo Cimentación)")
    
    resultados_asent = []
    z_abs_top = 0
    
    total_Se = 0
    total_Sc = 0
    
    # Tabla de cálculo
    filas_tabla = []

    for idx, row in df.iterrows():
        z_abs_bot = z_abs_top + row['Esp (m)']
        
        # Analizar solo estratos debajo de Df
        if z_abs_bot > Df:
            # Espesor efectivo afectado
            z_ini_eff = max(z_abs_top, Df)
            z_fin_eff = z_abs_bot
            H_eff = z_fin_eff - z_ini_eff
            
            # Profundidad media del sub-estrato para calcular esfuerzo
            z_mid_from_zapata = (z_ini_eff + H_eff/2) - Df
            
            # 1. Delta Sigma (2:1)
            d_sigma = calc_incremento_esfuerzo(P, B, L, z_mid_from_zapata)
            
            # 2. Sigma Inicial (Geostático aprox en el medio)
            # Esto requeriría integrar gammas anteriores. Simplificamos:
            sigma_0 = z_abs_top * row['Gamma (T/m3)'] + (H_eff/2)*row['Gamma (T/m3)'] # Muy simplificado
            
            # 3. Asentamiento Elástico
            Se = calc_asentamiento_elastico(d_sigma, B, row['Es (T/m2)'], row['nu']) if row['Es (T/m2)']>0 else 0
            
            # 4. Asentamiento Consolidación (Solo si es arcilla/cohesivo y Cc > 0)
            Sc = 0
            if row['Cc'] > 0:
                Sc = calc_consolidacion(H_eff, row['e0'], row['Cc'], sigma_0, d_sigma)
            
            total_Se += Se
            total_Sc += Sc
            
            filas_tabla.append({
                "Estrato": row['Tipo'],
                "H efec (m)": round(H_eff, 2),
                "Δσ (T/m2)": round(d_sigma, 2),
                "S. Elástico (cm)": round(Se*100, 2),
                "S. Consol (cm)": round(Sc*100, 2)
            })
            
        z_abs_top = z_abs_bot

    st.table(pd.DataFrame(filas_tabla))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Asent. Elástico Total", f"{total_Se*100:.2f} cm")
    c2.metric("Asent. Consolidación Total", f"{total_Sc*100:.2f} cm")
    c3.metric("Asentamiento TOTAL", f"{(total_Se+total_Sc)*100:.2f} cm", delta_color="inverse", delta="Máx sugerido 2.5cm")

    with st.expander("📘 Teoría utilizada"):
        st.markdown("**1. Distribución de Esfuerzos (Método 2:1):**")
        st.latex(r"\Delta \sigma_z = \frac{P}{(B+z)(L+z)}")
        st.markdown("**2. Asentamiento Elástico:**")
        st.latex(r"S_e = q B \frac{1-\nu^2}{E_s} I_f")
        st.markdown("**3. Consolidación (Terzaghi):**")
        st.latex(r"S_c = \frac{C_c H}{1+e_0} \log \left( \frac{\sigma'_0 + \Delta \sigma}{\sigma'_0} \right)")


# ==============================================================================
# 4. CONTROLADOR PRINCIPAL (MAIN LOOP)
# ==============================================================================

def main():
    st.title("🏗️ ToolZapatas v3.0 | Arquitectura Modular")
    st.markdown("Sistema extensible de cálculo de cimentaciones superficiales.")
    
    # 1. Ejecutar Sidebar (Configuración)
    modulo_sidebar()
    
    # 2. Definir Pestañas (Aquí se agregan nuevos módulos fácilmente)
    tabs = st.tabs(["🌍 1. Estratigrafía", "⚙️ 2. Capacidad Portante", "📉 3. Asentamientos"])
    
    with tabs[0]:
        modulo_estratigrafia()
        
    with tabs[1]:
        modulo_capacidad_carga()
        
    with tabs[2]:
        modulo_asentamientos()

if __name__ == "__main__":
    main()
