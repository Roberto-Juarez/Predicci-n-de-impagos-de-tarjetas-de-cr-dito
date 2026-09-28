import numpy as np
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



gradient_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
}
</style>
"""
st.markdown(gradient_bg, unsafe_allow_html=True)

glow_button = """
<style>
div.stButton > button:first-child {
    background: linear-gradient(45deg, #f39c12, #ee0979);
    color: white;
    border-radius: 20px;
    padding: 0.7em 1.5em;
    font-size: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
}
div.stButton > button:first-child:hover {
    box-shadow: 0px 6px 15px rgba(238,9,121,0.6);
}
</style>
"""
st.markdown(glow_button, unsafe_allow_html=True)

upload_style = """
<style>
[data-testid="stFileUploader"] {
    background: rgba(172, 182, 229, 0.5);
    border: #74ebd5;
    border-radius: 15px;
    padding: 1.5em;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
[data-testid="stFileUploader"] > div {
    color: #333;
    font-weight: bold;
    text-align: center;
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #74ebd5 0%, #ACB6E5 100%);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: bold;
    transition: 0.3s;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(90deg, #ACB6E5 0%, #74ebd5 100%);
    transform: scale(1.05);
}
</style>
"""
st.markdown(upload_style, unsafe_allow_html=True)

st.markdown("""
<style>
/* Fondo del sidebar con transparencia */
[data-testid="stSidebar"] {
    background-color: rgba(247, 249, 252, 0.35); /* gris-azulado con 85% opacidad */
    border-right: 1px solid rgba(220, 227, 234, 0.6);
}

/* Texto dentro del sidebar */
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] div, 
[data-testid="stSidebar"] span {
    color: #333333;
    font-size: 14px;
}

/* Encabezados */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #0077B6;
    font-weight: bold;
}

/* Expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: rgba(238, 243, 248, 0.7); /* más transparente */
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar.expander("Diccionario de columnas"):
    
    df_demo = pd.read_csv("dataset_20filas.csv")

    
    st.download_button(
        label="Descargar dataset de prueba",
        data=df_demo.to_csv(index=False).encode("utf-8"),
        file_name="dataset_prueba.csv",
        mime="text/csv"
    )
    st.write("""
    **LIMIT_BAL**: Monto del crédito otorgado.
    
    **SEX**: Género (1=masculino, 2=femenino).
    
    **EDUCATION**: Nivel educativo (1=posgrado, 2=universidad, 3=preparatoria, 4-6=otros).
   
    **MARRIAGE**: Estado civil (1=casado, 2=soltero, 3=otros).
    
    **AGE**: Edad en años.
   
    **PAY_0**: Historial de pagos por mes.
       - -2, -1 = pago anticipado
       - 0 = sin retraso
       - 1 = un mes de retraso
       - hasta 8 = ocho meses de retraso
   
    **BILL_AMT1–6**: Monto facturado en cada mes.
    
    **PAY_AMT1–6**: Monto pagado en cada mes.
    """)

with st.container():
    c1, c2, c3 = st.columns([1,6,1]) 
    
    
    with c2:
        st.header("Predicción con el mejor modelo")
        st.write("Esta aplicación predice la probabilidad de que un cliente no pague su tarjeta usando un modelo entrenado con datos históricos.")

predictor = joblib.load("m_credito.pkl")
df_demo = pd.read_csv("dataset_20filas.csv")
df_resultados = pd.DataFrame()
archivo = st.file_uploader("Cargar dataset en formato CSV", type=["csv"])
   
 
co1, co2, co3, co4, co5 = st.columns(5)

with co2:
    generar_archivo = st.button("Predicciones desde archivo")

if generar_archivo:
    
    if archivo is not None:
        X_input = pd.read_csv(archivo)
   
        
        pred = predictor.predict(X_input)
        probas = predictor.predict_proba(X_input)
        st.markdown("---") 
         
    
        df_probas = pd.DataFrame(probas, columns=["Prob_Paga","Prob_NoPaga"])
        df_pred = pd.DataFrame(pred, columns=['Predicción'])
        
        df_resultados = pd.concat([df_pred, df_probas, X_input.reset_index(drop=True), ], axis=1)
    
         
        paga = (df_resultados['Predicción'] == 0).sum()
        no_paga = (df_resultados['Predicción'] == 1).sum()
        
        labels = ["Clientes que pagan", "Clientes que no pagan"]
        sizes = [paga, no_paga]  
        colors = sns.color_palette("pastel")[0:2]  
       
       # Gráfica de pastel
       
        col1, col2  = st.columns(2)

        with col1:
           st.markdown(f"<p style='font-size:16px; color:#333;'>Clientes que pagan: {paga}</p>", unsafe_allow_html=True)
           st.markdown(f"<p style='font-size:16px; color:#333;'>Clientes que no pagan: {no_paga}</p>", unsafe_allow_html=True)
           
          
        with col2:
           fig, ax = plt.subplots(figsize=(2,2))

           
           ax.set_facecolor("#acb6e5")   # azul muy claro
           
           
           fig.patch.set_facecolor("#acb6e5")  # gris-azulado suave
           
          
           ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={'fontsize': 8})
           
           ax.set_title("Distribución de clientes según predicción", fontsize=10)
           ax.axis("equal")  
           
           st.pyplot(fig)    
         
            
        df_estilizado = df_resultados.style.set_properties(**{
            'background-color': '#acb6e5',
            'color': 'black',
            'border': '1px solid #ffffff'
        }).set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#acb6e5'),
                    ('color', 'black'),
                    ('font-weight', 'bold'),
                    ('padding', '10px'),
                    ('border', '1px solid #ffffff')
                ]
            }
        ])


        st.markdown("""
        <style>
            .tabla-contenedor {
                overflow-x: auto;
                max-width: 100%;
                margin-bottom: 20px;
            }
            .tabla-contenedor table {
                width: 100%;
                border-collapse: collapse;
            }
        </style>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No hay dataset cargado. Por favor sube un archivo CSV antes de presionar el botón.")

with co4:
    generar = st.button("Generar caso aleatorio")
    num_filas = st.slider("Número de casos a generar", min_value=1, max_value=50, value=1)

if generar:   
    
    X_random = pd.DataFrame([
            [
            np.random.randint(10000,1000000), np.random.choice([2,1]),
            np.random.choice([0,1,2,3,4,5,6]), np.random.choice([0,1,2,3]),
            np.random.randint(21,79), np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8]),
            np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8]), np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8]),
            np.random.choice([-2,-1,0,1,2,3,4,5,6,7,8]), np.random.choice([-2,-1,0,2,3,4,5,6,7,8]),
            np.random.choice([-2,-1,0,2,3,4,5,6,7,8]), np.random.randint(-165580,964511), 
            np.random.randint(-69777,983931), np.random.randint(-157264,1664089), 
            np.random.randint(-170000,891586), np.random.randint(-81334,927171),
            np.random.randint(-339603,961664), np.random.randint(0,873552), 
            np.random.randint(0,1684259), np.random.randint(0,896040), 
            np.random.randint(0,621000), np.random.randint(0,426529), 
            np.random.randint(0,528666)
            
            ]
            
            for _ in range(num_filas)
            
            ], columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
                       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
        
    pred = predictor.predict(X_random)
    probas = predictor.predict_proba(X_random)
    st.markdown("---") 
        
    df_probas = pd.DataFrame(probas, columns=["Prob_Paga","Prob_NoPaga"])
    df_pred = pd.DataFrame(pred, columns=['Predicción'])

    df_resultados = pd.concat([df_pred, df_probas, X_random.reset_index(drop=True), ], axis=1)
        
    paga = (df_resultados['Predicción'] == 0).sum()
    no_paga = (df_resultados['Predicción'] == 1).sum()
    
    labels = ["Clientes que pagan", "Clientes que no pagan"]
    sizes = [paga, no_paga]  
    colors = sns.color_palette("pastel")[0:2]  
    
    # Gráfica de pastel
    
    col1, col2  = st.columns(2)

    with col1:
        st.markdown(f"<p style='font-size:16px; color:#333;'>Clientes que pagan: {paga}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:16px; color:#333;'>Clientes que no pagan: {no_paga}</p>", unsafe_allow_html=True)
        
       
    with col2:
        fig, ax = plt.subplots(figsize=(2,2))

       
        ax.set_facecolor("#acb6e5")   
        
        
        fig.patch.set_facecolor("#acb6e5")  
        
       
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={'fontsize': 8})
        
        ax.set_title("Distribución de clientes según predicción", fontsize=10)
        ax.axis("equal")  
        
        st.pyplot(fig)  
        
        
df_estilizado = df_resultados.style.set_properties(**{
    'background-color': '#acb6e5',
    'color': 'black',
    'border': '1px solid #ffffff'
}).set_table_styles([
    {
        'selector': 'th',
        'props': [
            ('background-color', '#acb6e5'),
            ('color', 'black'),
            ('font-weight', 'bold'),
            ('padding', '10px'),
            ('border', '1px solid #ffffff')
        ]
    }
])


st.markdown("""
<style>
    .tabla-contenedor {
        overflow-x: auto;
        max-width: 100%;
        margin-bottom: 20px;
    }
    .tabla-contenedor table {
        width: 100%;
        border-collapse: collapse;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="tabla-contenedor">{df_estilizado.to_html()}</div>', 
    unsafe_allow_html=True
)
