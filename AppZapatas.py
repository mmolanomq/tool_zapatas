# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
st.set_page_config(page_title="ToolZapatas v2.0", layout="wide", page_icon="🏗️")

st.markdown("""
<style>
    .stApp { background-color: #fcfcfc; }
    h1, h2, h3 { color: #2C3E50; font-family: 'Segoe UI', sans-serif; }
    .stMetric { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; }
    .stDataFrame { border: 1px solid #d0d0d0; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. FUNCIONES GEOTÉCNICAS (MOTOR DE CÁLCULO)
# ==============================================================================

def obtener_suelo_en_punta(df_suelo, Df):
    """
    Determina las propiedades del suelo justo en la base de la zapata (Df).
    """
    z_acum = 0
    propiedades = None
    
    # Si Df es 0 o menor (superficie), toma el primer estrato
    if Df <= 0:
        return df_suelo.iloc[0]

    for idx, row in df_suelo.iterrows():
        z_top = z_acum
        z_bot = z_acum + row['Espesor (m)']
        
        # Verificar si la punta cae en este estrato
        # Usamos < z_bot para atraparlo, pero si es igual al limite, 
        # asumimos que apoya en el de arriba (o abajo según criterio, aquí conservador: el de arriba)
        if z_top <= Df <= z_bot:
            propiedades = row
            break
        
        z_acum = z_bot
    
    # Si Df es más profundo que el último estrato, asumimos el último
    if propiedades is None:
        propiedades = df_suelo.iloc[-1]
        
    return propiedades

def calcular_meyerhof(B, L, Df, gamma, c, phi, carga_axial_ton):
    """
    Calcula q_ult y Factores de Meyerhof con detalle.
    Retorna diccionario completo.
    """
    # Conversiones y ajustes
    phi_rad = np.radians(phi)
    kp = np.tan(np.radians(45) + phi_rad/2)**2
    
    # 1. Factores de Capacidad de Carga (N)
    if phi < 0.1: # Caso No Drenado (Arcilla Phi=0)
        Nq = 1.0
        Nc = 5.14
        Ny = 0.0
    else:
        Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.radians(45) + phi_rad/2))**2
        Nc = (Nq - 1) / np.tan(phi_rad)
        Ny = (Nq - 1) * np.tan(1.4 * phi_rad)

    # 2. Factores de Forma (Shape - s)
    # Meyerhof standard:
    if phi == 0:
        sq = 1.0
        sc = 1 + 0.2 * (B/L)
        sy = 1.0
    else:
        sc = 1 + 0.2 * kp * (B/L)
        sq = 1 + 0.1 * kp * (B/L)
        sy = 1 + 0.1 * kp * (B/L)

    # 3. Factores de Profundidad (Depth - d)
    # Para Df/B <= 1 (Simplificado)
    if Df/B <= 1:
        k_d = Df/B
    else:
        k_d = np.arctan(Df/B) # En radianes para Df/B > 1 (simplificación común)

    if phi == 0:
        dc = 1 + 0.4 * k_d
        dq = 1.0
        dy = 1.0
    else:
        dq = 1 + 0.1 * np.sqrt(kp) * k_d
        sc_param = 1 if phi == 0 else sc # Ajuste formula general
        dc = 1 + 0.4 * np.sqrt(kp) * k_d
        dy = 1 + 0.1 * np.sqrt(kp) * k_d

    # 4. Cálculo de q_ult
    # Termino Cohesión: c * Nc * sc * dc
    term_c = c * Nc * sc * dc
    # Termino Sobrecarga: q * Nq * sq * dq (q = gamma * Df efectivo)
    q_sobrecarga = gamma * Df
    term_q = q_sobrecarga * Nq * sq * dq
    # Termino Peso: 0.5 * gamma * B * Ny * sy * dy
    term_y = 0.5 * gamma * B * Ny * sy * dy
    
    q_ult = term_c + term_q + term_y
    
    return {
        "q_ult": q_ult,
        "Nq": Nq, "Nc": Nc, "Ny": Ny,
        "sc": sc, "sq": sq, "sy": sy,
        "dc": dc, "dq": dq, "dy": dy,
        "term_c": term_c, "term_q": term_q, "term_y": term_y
    }

# ==============================================================================
# 2. VISUALIZACIÓN
# ==============================================================================

def plot_perfil_estratigrafico(df_suelo, Df, B, L_zapata, h_zapata):
    """Genera gráfico Matplotlib con estratos y zapata."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    z_acum = 0
    max_width = B * 3 if B > 0 else 5
    center_x = max_width / 2
    
    # Colores por tipo
    colors = {"Relleno": "#E5E7E9", "Arcilla": "#D7BDE2", "Arena": "#F9E79F", "Grava": "#A3E4D7", "Roca": "#AED6F1"}
    
    # 1. DIBUJAR ESTRATOS
    max_depth = df_suelo['Espesor (m)'].sum()
    if max_depth < Df + h_zapata + 2: max_depth = Df + h_zapata + 2
    
    for _, row in df_suelo.iterrows():
        esp = row['Espesor (m)']
        tipo = row['Tipo']
        c_fill = colors.get(tipo, "#F2F3F4")
        
        # Rectangulo del estrato
        rect = patches.Rectangle((0, z_acum), max_width, esp, facecolor=c_fill, edgecolor='#566573', alpha=0.6)
        ax.add_patch(rect)
        
        # Texto del estrato
        ax.text(0.2, z_acum + esp/2, f"{tipo}\n$\gamma$={row['Gamma (Ton/m3)']}\n$c$={row['Cohesion (Ton/m2)']}", 
                va='center', fontsize=9, color='#17202A')
        
        z_acum += esp
        
    # 2. DIBUJAR ZAPATA
    # Coordenadas esquina superior izquierda de la zapata
    # La base está en Df. La parte superior está en Df - h
    z_base = Df
    z_top = Df - h_zapata
    
    # Si Df < h, la zapata sobresale, manejamos visualización
    
    # Zapata (Gris oscuro)
    zapata_rect = patches.Rectangle((center_x - B/2, z_top), B, h_zapata, 
                                    facecolor='#5D6D7E', edgecolor='black', linewidth=1.5, label='Zapata')
    ax.add_patch(zapata_rect)
    
    # Pedestal / Columna (simbólico)
    col_w = 0.3
    col_rect = patches.Rectangle((center_x - col_w/2, z_top - 1.0), col_w, 1.0, 
                                 facecolor='#ABB2B9', edgecolor='black', linestyle='--')
    ax.add_patch(col_rect)
    
    # Línea de cota Df
    ax.annotate(f'Df = {Df}m', xy=(center_x + B/2, Df), xytext=(center_x + B/2 + 0.5, Df),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Configuración Ejes
    ax.set_ylim(max_depth, -1) # Invertir eje Y para profundidad
    ax.set_xlim(0, max_width)
    ax.set_ylabel("Profundidad (m)")
    ax.set_xlabel("Ancho (m)")
    ax.set_title("Perfil Estratigráfico y Posición de Cimentación")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    return fig

# ==============================================================================
# 3. INTERFAZ PRINCIPAL
# ==============================================================================

def main():
    st.title("🏗️ ToolZapatas v2.0: Diseño Geotécnico Detallado")
    
    # --- BARRA LATERAL: CARGAS Y GEOMETRÍA ---
    with st.sidebar:
        st.header("1. Datos de Carga y Geometría")
        st.info("Defina las cargas de servicio y la geometría inicial.")
        P_serv = st.number_input("Carga Axial P (Ton)", value=50.0)
        B_input = st.number_input("Ancho B (m)", value=1.5, step=0.1)
        L_input = st.number_input("Largo L (m)", value=1.5, step=0.1)
        h_input = st.number_input("Espesor h (m)", value=0.4, step=0.05)
        Df_input = st.number_input("Profundidad Desplante Df (m)", value=1.5, step=0.1)
        FS_req = st.slider("FS Requerido", 2.0, 4.0, 3.0)

    # --- TABS ---
    tab_suelo, tab_calc, tab_opt = st.tabs(["🌍 Estratigrafía", "📐 Cálculos y Resultados", "🚀 Optimización"])

    # ----------------------------------------------------------------------
    # TAB 1: ESTRATIGRAFÍA (NUEVO)
    # ----------------------------------------------------------------------
    with tab_suelo:
        col_s1, col_s2 = st.columns([1, 1])
        
        with col_s1:
            st.subheader("Definición de Capas de Suelo")
            st.markdown("Edite la tabla para agregar estratos. **El orden es de arriba hacia abajo.**")
            
            # DataFrame Inicial
            data_inicial = [
                {"Tipo": "Relleno", "Espesor (m)": 1.0, "Gamma (Ton/m3)": 1.6, "Cohesion (Ton/m2)": 0.0, "Phi (°)": 28.0},
                {"Tipo": "Arcilla", "Espesor (m)": 3.0, "Gamma (Ton/m3)": 1.8, "Cohesion (Ton/m2)": 5.0, "Phi (°)": 15.0},
                {"Tipo": "Arena", "Espesor (m)": 5.0, "Gamma (Ton/m3)": 1.9, "Cohesion (Ton/m2)": 0.0, "Phi (°)": 32.0},
            ]
            
            df_estratos = st.data_editor(
                data_inicial,
                column_config={
                    "Tipo": st.column_config.SelectboxColumn(options=["Relleno", "Arcilla", "Arena", "Grava", "Roca"]),
                    "Phi (°)": st.column_config.NumberColumn(min_value=0, max_value=45),
                    "Cohesion (Ton/m2)": st.column_config.NumberColumn(min_value=0.0)
                },
                num_rows="dynamic",
                use_container_width=True
            )
            
            # Convertir a DataFrame de Pandas seguro
            df_soil = pd.DataFrame(df_estratos)
        
        with col_s2:
            st.subheader("Perfil Gráfico")
            if not df_soil.empty:
                fig_soil = plot_perfil_estratigrafico(df_soil, Df_input, B_input, L_input, h_input)
                st.pyplot(fig_soil)
            else:
                st.warning("Ingrese al menos un estrato.")

    # ----------------------------------------------------------------------
    # TAB 2: CÁLCULOS (MOTOR TRANSPARENTE)
    # ----------------------------------------------------------------------
    with tab_calc:
        st.subheader("Memoria de Cálculo Geotécnico")
        
        if df_soil.empty:
            st.error("Por favor defina la estratigrafía primero.")
        else:
            # 1. Obtener suelo en la punta
            suelo_punta = obtener_suelo_en_punta(df_soil, Df_input)
            
            # Mostrar propiedades detectadas
            st.markdown(f"""
            **Suelo detectado en la cota de cimentación ($D_f = {Df_input}$ m):**
            * Tipo: `{suelo_punta['Tipo']}`
            * Cohesión ($c$): {suelo_punta['Cohesion (Ton/m2)']} Ton/m²
            * Fricción ($\phi$): {suelo_punta['Phi (°)']}°
            * Peso Unitario ($\gamma$): {suelo_punta['Gamma (Ton/m3)']} Ton/m³
            """)
            st.divider()

            # 2. Ejecutar Cálculo Meyerhof
            res = calcular_meyerhof(
                B_input, L_input, Df_input, 
                suelo_punta['Gamma (Ton/m3)'], 
                suelo_punta['Cohesion (Ton/m2)'], 
                suelo_punta['Phi (°)'], 
                P_serv
            )
            
            q_ult = res['q_ult']
            q_adm = q_ult / FS_req
            area = B_input * L_input
            peso_propio = area * h_input * 2.4 # Concreto
            q_act = (P_serv + peso_propio) / area
            FS_calc = q_ult / q_act if q_act > 0 else 999

            # 3. Mostrar Ecuaciones
            st.markdown("### 1. Ecuación General de Capacidad de Carga (Meyerhof)")
            st.latex(r"q_{ult} = c N_c s_c d_c + q N_q s_q d_q + 0.5 \gamma B N_\gamma s_\gamma d_\gamma")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Factores de Capacidad (N)**")
                st.latex(f"N_c = {res['Nc']:.2f}")
                st.latex(f"N_q = {res['Nq']:.2f}")
                st.latex(f"N_\gamma = {res['Ny']:.2f}")
            with c2:
                st.markdown("**Factores de Forma (s)**")
                st.latex(f"s_c = {res['sc']:.2f}")
                st.latex(f"s_q = {res['sq']:.2f}")
                st.latex(f"s_\gamma = {res['sy']:.2f}")
            with c3:
                st.markdown("**Factores de Profundidad (d)**")
                st.latex(f"d_c = {res['dc']:.2f}")
                st.latex(f"d_q = {res['dq']:.2f}")
                st.latex(f"d_\gamma = {res['dy']:.2f}")

            st.markdown("### 2. Contribución por Término (Ton/m²)")
            t1, t2, t3 = st.columns(3)
            t1.metric("Cohesión", f"{res['term_c']:.2f}")
            t2.metric("Sobrecarga (q)", f"{res['term_q']:.2f}")
            t3.metric("Fricción/Peso", f"{res['term_y']:.2f}")
            
            st.divider()
            
            # 4. Resultados Finales
            st.markdown("### 3. Verificación Final")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Q Última", f"{q_ult:.2f} Ton/m²")
            k2.metric("Q Admisible", f"{q_adm:.2f} Ton/m²")
            k3.metric("Q Actuante", f"{q_act:.2f} Ton/m²", delta_color="inverse", delta=f"P.Propio: {peso_propio:.1f}T")
            
            estado = "✅ CUMPLE" if FS_calc >= FS_req else "❌ FALLA"
            k4.metric("Factor Seguridad", f"{FS_calc:.2f}", delta=estado)
            
            if FS_calc < FS_req:
                st.error(f"La zapata falla por capacidad portante. Aumente B, L o Df. (FS Actual: {FS_calc:.2f} < {FS_req})")
            else:
                st.success("Diseño Geotécnico Satisfactorio.")

    # ----------------------------------------------------------------------
    # TAB 3: OPTIMIZACIÓN SIMPLE
    # ----------------------------------------------------------------------
    with tab_opt:
        st.subheader("Buscador de Dimensiones Óptimas")
        if st.button("Buscar B óptimo para la carga actual"):
            if df_soil.empty:
                st.error("Defina estratigrafía primero.")
            else:
                suelo_opt = obtener_suelo_en_punta(df_soil, Df_input)
                
                found = False
                # Iterar B desde 0.5 hasta 5.0m
                for b_try in np.arange(0.5, 5.0, 0.1):
                    # Asumimos zapata cuadrada para optimización rápida
                    res_opt = calcular_meyerhof(
                        b_try, b_try, Df_input,
                        suelo_opt['Gamma (Ton/m3)'], suelo_opt['Cohesion (Ton/m2)'], suelo_opt['Phi (°)'], P_serv
                    )
                    
                    q_u = res_opt['q_ult']
                    area_opt = b_try**2
                    q_act_opt = (P_serv + (area_opt*h_input*2.4)) / area_opt
                    fs_opt = q_u / q_act_opt
                    
                    if fs_opt >= FS_req:
                        st.success(f"✅ Dimensión Óptima Encontrada: B = L = {b_try:.2f} m")
                        st.write(f"FS Resultante: {fs_opt:.2f}")
                        found = True
                        break
                
                if not found:
                    st.error("No se encontró solución razonable (B < 5m). Revise suelo o cargas.")

if __name__ == "__main__":
    main()
