import streamlit as st
import joblib
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Predicción de Diabetes", layout="wide")

# CSS personalizado para estilo y adaptabilidad
st.markdown(
    """
    <style>
    /* Ocultar elementos por defecto */
    MainMenu, footer, header { visibility: hidden; }
    
    /* Centrar títulos */
    h1, h2, h3 { text-align: center; }
   
    /* Botón predecir centralizado */
    div.stButton > button:first-child {
        padding: 2rem 4rem;
        border: none;
        border-radius: 12px;
        font-size: 18px;
        display: block;
        margin: 1rem auto;
        width: 100%; 
        max-width: 300px;
    }
    h2{
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        padding: 1rem !important;
        margin:1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cargar modelo
@st.cache_resource
def load_model(path="modelo76SinCols.pkl"):
    return joblib.load(path)

model = load_model()
features = list(model.feature_names_in_)

# Dominios para inputs
binary_vars = {"Diabetes_binary","HighBP","HighChol","CholCheck","Smoker",
               "Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies",
               "HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","DiffWalk","Sex"}
discrete_domains = {
    "GenHlth": [1,2,3,4,5],
    "MentHlth": [0,1,2,3,4,5,7,10,15,30],
    "PhysHlth": [0,1,2,3,4,5,6,10,15,30],
    "Age": [1,2,3,4,5,6,7,8,9,10,11,12,13],
    "Education": [1,2,3,4,5,6],
    "Income": [1,2,3,4,5,6,7,8]
}
continuous_domains = {"BMI": (18.0, 60.0)}

#traduccion y descripcion de variables
descriptions={
    "HighBP":["Presión Alta","¿Alguna vez le han dicho que tiene presión arterial alta?"],
    "HighChol":["Colesterol Alto","¿Alguna vez le han dicho que tiene colesterol alto?"],
    "CholCheck":["Chequeo de Colesterol","¿Se ha hecho un chequeo de colesterol en los últimos 5 años?"],
    "Smoker":["Fumador","¿Es usted fumador?"],
    "Stroke":["Accidente Cerebrovascular","¿Alguna vez ha tenido un accidente cerebrovascular?"],
    "HeartDiseaseorAttack":["Enfermedad Cardiaca","¿Alguna vez le han dicho que tiene enfermedad cardíaca?"],
    "PhysActivity":["Actividad Física","¿Realiza alguna actividad física regularmente?"],
    "Fruits":["Consumo de Frutas","¿Consume frutas al menos una vez al día?"],
    "Veggies":["Consumo de Verduras","¿Consume verduras al menos una vez al día?"],
    "HvyAlcoholConsump":["Consumo Excesivo de Alcohol","¿Consume alcohol en exceso?"],
    "AnyHealthcare":["Atención Médica","¿Tiene acceso a atención médica?"],
    "NoDocbcCost":["Costo de Atención Médica","¿Alguna vez no ha podido ver a un médico debido a costos?"],
    "DiffWalk":["Dificultad para Caminar","¿Tiene dificultad para caminar o subir escaleras?"],
    "Sex":["Sexo","¿Cuál es su sexo?"],
    "GenHlth":["Salud General","¿Cómo calificaría su salud en general?"],
    "MentHlth":["Salud Mental","¿Cuántos días en el último mes ha tenido problemas de salud mental?"],
    "PhysHlth":["Salud Física","¿Cuántos días en el último mes ha tenido problemas de salud física?"],
    "Age":["Edad","¿Cuál es su grupo de edad?"],
    "Education":["Nivel Educativo","¿Cuál es su nivel educativo?"],
    "Income":["Ingreso Familiar","¿Cuál es su nivel de ingreso familiar?"],
    "BMI":["Índice de Masa Corporal","¿Cuál es su índice de masa corporal (IMC)?"],
}



def get_input(var):
    label,desc = descriptions[var]
    st.markdown(f"""
<div style='display: flex; gap: 8px; align-items: baseline;margin-top:1rem;'>
    <strong>{label}</strong>
    <span style='color: gray; font-size: 0.9em; padding-left:.5rem'>{desc}</span>
</div>
""", unsafe_allow_html=True)


    if var in binary_vars:
        if var == "Sex":
            return 1.0 if st.radio("", ("Hombre","Mujer"), horizontal=True , label_visibility="collapsed")=="Hombre" else 0.0
        return 1.0 if st.radio("", ("Sí","No"), horizontal=True,key =f"{var}_radio")=="Sí" else 0.0
    if var in discrete_domains:
        return st.select_slider("", discrete_domains[var])
    if var in continuous_domains:
        mi, ma = continuous_domains[var]
        return st.number_input("", min_value=mi, max_value=ma,
                               value=(mi+ma)/2, step=0.1, format="%.1f")
    return st.number_input("", value=0.0)

# Título principal y descripción
st.markdown("<h1>Predicción de Riesgo de Diabetes</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Completa el formulario con tus datos para obtener una estimación personalizada del riesgo de diabetes.</p>", unsafe_allow_html=True)
inputs = {}

i_section  =1
def make_section(title, variables):
    global i_section
    with st.container():
        
        st.markdown(f"<h2>{i_section}. {title}</h2>", unsafe_allow_html=True)
        cols = st.columns(2)
        col1,col2 = variables
        with cols[0]:
            for var in col1:
                if var in features: inputs[var] = get_input(var)
        with cols[1]:
            for var in col2:
                if var in features: inputs[var] = get_input(var)
        st.divider()

    i_section += 1


# make_section("Datos Demográficos", (["Age","Income"], ["Education","Sex"]))
# make_section("Salud y Bienestar", (["BMI","MentHlth","PhysHlth","AnyHealthcare","PhysActivity"],
#                 ["Fruits","Veggies","HvyAlcoholConsump","NoDocbcCost","Smoker"]))
# make_section("Antecedentes Médicos", (["CholCheck","HighBP","HighChol","Stroke"],
#                 ["HeartDiseaseorAttack","DiffWalk"]))
# make_section("Evaluación General", (["GenHlth"], ["Diabetes_binary"]))

make_section("Datos Demográficos", (["Age","Income"], ["Education","Sex"]))
make_section("Salud y Bienestar", (["BMI","PhysHlth"],
                ["HvyAlcoholConsump","GenHlth"]))
make_section("Antecedentes Médicos", (["CholCheck","HighBP","HighChol","Stroke"],
                ["HeartDiseaseorAttack","DiffWalk"]))


# Botón para predecir
enviar = st.button("Predecir Riesgo")

# Procesar predicción y mostrar resultado
if enviar:
    try:
        df = pd.DataFrame([inputs])
        X = df[features]
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else float(model.predict(X)[0])
        pred = model.predict(X)[0]
        # Mostrar en contenedor estilizado
        st.divider()
        st.markdown('<h2>Resultado</h2>', unsafe_allow_html=True)
        cols = st.columns([1,2,1])
        with cols[1]:
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 1.5rem; font-weight: bold;   ">
                    Probabilidad de Diabetes: {prob*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
            if pred == 1:
                st.markdown("<p style='text-align:center; font-size:1.2rem; color:#d32f2f;'>🟥 Alto riesgo de diabetes</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='text-align:center; font-size:1.2rem; color:#388e3c;'>🟩 Bajo riesgo de diabetes</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
