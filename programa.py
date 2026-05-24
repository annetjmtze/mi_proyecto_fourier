import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import io

# Configuración de la página
st.set_page_config(page_title="Análisis de Fourier - Armónicos", layout="wide")

# Título
st.title("🔍 Análisis de Series de Fourier")
st.markdown("Encuentra las componentes senoidales (magnitud, ángulo y fase) de 1 a 100 Hz")

# Cargar datos
uploaded_file = st.file_uploader("📂 Cargar archivo CSV (columna 'Muestras')", type=["csv"])

if uploaded_file is not None:
    # Leer CSV
    df = pd.read_csv(uploaded_file)
    
    # Verificar columna
    if 'Muestras' not in df.columns:
        st.error("El archivo debe tener una columna llamada 'Muestras'")
        st.stop()

    # Convertir a string, reemplazar comas por puntos y convertir a numérico
    df['Muestras'] = pd.to_numeric(df['Muestras'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['Muestras'])

    if len(df) == 0:
        st.error("No se encontraron datos numéricos válidos. Revisa el formato de tu CSV.")
        st.stop()

    muestras = df['Muestras'].values
    media_original = np.mean(muestras)
    muestras_centradas = muestras - media_original # Restar media para ver mejor los armónicos

    N = len(muestras)
    st.success(f"Datos cargados: {N} muestras")
    
    # Parámetros
    st.sidebar.header("⚙️ Parámetros")
    fs = st.sidebar.number_input("Frecuencia de muestreo (Hz)", min_value=1.0, max_value=10000.0, value=1000.0, step=10.0)
    f_min = st.sidebar.number_input("Frecuencia mínima (Hz)", min_value=1, max_value=100, value=1)
    f_max = st.sidebar.number_input("Frecuencia máxima (Hz)", min_value=1, max_value=100, value=100)
    
    # Calcular duración
    T = N / fs
    t = np.linspace(0, T, N, endpoint=False)
    
    # Prepara resultados
    frecuencias = np.arange(f_min, f_max + 1)

    with st.spinner("Calculando componentes de Fourier..."):
        # Cálculo vectorizado para mayor eficiencia (O(F*N) usando álgebra lineal)
        # Se genera una matriz de ángulos (frecuencias x tiempo)
        angles = 2 * np.pi * frecuencias[:, np.newaxis] * t
        
        a_n_vals = (2 / N) * np.dot(np.cos(angles), muestras_centradas)
        b_n_vals = (2 / N) * np.dot(np.sin(angles), muestras_centradas)
        
        magnitudes = np.sqrt(a_n_vals**2 + b_n_vals**2)
        angulos = np.degrees(np.arctan2(b_n_vals, a_n_vals))
        fases = -angulos
    
    # Crear DataFrame
    resultados = pd.DataFrame({
        "Frecuencia (Hz)": frecuencias,
        "Magnitud": magnitudes,
        "Ángulo (°)": angulos,
        "Fase (°)": fases,
        "a_n": a_n_vals,
        "b_n": b_n_vals
    })
    
    # Mostrar tabla
    st.subheader("📊 Resultados de Fourier")
    st.dataframe(resultados.style.format({
        "Magnitud": "{:.4f}",
        "Ángulo (°)": "{:.2f}",
        "Fase (°)": "{:.2f}",
        "a_n": "{:.4f}",
        "b_n": "{:.4f}"
    }), height=400, use_container_width=True)
    
    # Top armónicos
    st.subheader("🎵 Principales armónicos (mayor magnitud)")
    top_n = st.slider("Mostrar top N", min_value=5, max_value=20, value=10)
    top = resultados.nlargest(top_n, "Magnitud")[["Frecuencia (Hz)", "Magnitud", "Ángulo (°)", "Fase (°)"]]
    st.dataframe(top.style.format({
        "Magnitud": "{:.4f}",
        "Ángulo (°)": "{:.2f}",
        "Fase (°)": "{:.2f}"
    }), use_container_width=True)
    
    # Gráfica de magnitud vs frecuencia
    st.subheader("📈 Espectro de magnitud")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stem(frecuencias, magnitudes, basefmt=" ")
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    ax.set_title("Espectro de Fourier (1-100 Hz)")
    ax.grid(True)
    st.pyplot(fig)
    
    # Gráfica de fase
    st.subheader("📉 Fase vs frecuencia")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(frecuencias, fases, 'o-', markersize=3)
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Fase (°)")
    ax2.set_title("Fase de componentes")
    ax2.grid(True)
    st.pyplot(fig2)
    
    # Señal original vs reconstruida (con los armónicos principales)
    st.subheader("🔄 Señal original vs reconstruida")
    num_armonicos = st.slider("Usar top N armónicos para reconstruir", 1, 50, 10)
    top_freqs = resultados.nlargest(num_armonicos, "Magnitud")
    
    reconstruida = np.full_like(t, media_original) # Empezar con la componente DC (media)
    for _, row in top_freqs.iterrows():
        f = row["Frecuencia (Hz)"]
        mag = row["Magnitud"]
        fase_rad = np.radians(row["Fase (°)"])
        reconstruida += mag * np.cos(2 * np.pi * f * t + fase_rad)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t, muestras, label="Original", alpha=0.7)
    ax3.plot(t, reconstruida, label=f"Reconstruida ({num_armonicos} armónicos)", alpha=0.7)
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("Amplitud")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)
    
    # Exportar resultados
    st.subheader("💾 Exportar resultados")
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        csv_output = io.BytesIO()
        resultados.to_csv(csv_output, index=False)
        st.download_button(
            label="📥 Descargar CSV completo",
            data=csv_output.getvalue(),
            file_name="resultados_fourier.csv",
            mime="text/csv"
        )

    def get_table_export(df_to_export, format_type='png'):
        # Formatear los valores para la exportación visual
        df_temp = df_to_export.copy()
        for col in ["Magnitud", "a_n", "b_n"]:
            if col in df_temp.columns: df_temp[col] = df_temp[col].map("{:.4f}".format)
        for col in ["Ángulo (°)", "Fase (°)"]:
            if col in df_temp.columns: df_temp[col] = df_temp[col].map("{:.2f}".format)
        
        fig, ax = plt.subplots(figsize=(10, len(df_temp) * 0.4 + 1))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df_temp.values, colLabels=df_temp.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        buf = io.BytesIO()
        plt.savefig(buf, format=format_type, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return buf.getvalue()

    with col_exp2:
        st.download_button(
            label="🖼️ Descargar Tabla Top (PNG)",
            data=get_table_export(top, 'png'),
            file_name="tabla_fourier_top.png",
            mime="image/png"
        )

    with col_exp3:
        st.download_button(
            label="📄 Descargar Tabla Top (PDF)",
            data=get_table_export(top, 'pdf'),
            file_name="tabla_fourier_top.pdf",
            mime="application/pdf"
        )
    
else:
    st.info("👈 Por favor, carga tu archivo CSV con la columna 'Muestras'")
    st.markdown("""
    ### Formato esperado del CSV:
    ```csv
    Muestras
    1.613472399
    1.849850658
    0.896787487
    ...
    ```
    """)