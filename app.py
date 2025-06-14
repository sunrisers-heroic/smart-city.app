import streamlit as st
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from datetime import datetime
from fpdf import FPDF

# Language translations
LANGUAGES = {
    "en": {
        "title": "🌿 Sustainability Smart City Assistant",
        "subtitle": "Ask about urban planning, green tech, sustainability, and smart city solutions.",
        "home_welcome": "🌿 Welcome to Your Smart City Assistant",
        "highlights": "### 🧠 Highlights:",
        "chat": "🤖 AI Chatbot",
        "urban_planning": "🏙️ Urban Planning",
        "green_energy": "🌳 Green Energy",
        "transportation": "🚇 Transportation",
        "water_management": "💧 Water Management",
        "waste_management": "🗑️ Waste Management",
        "reports": "📊 Progress Reports",
        "settings": "⚙️ Settings & Preferences",
        "footer": "© 2025 SmartCity Assistant | Built with ❤️ using Streamlit & Watsonx",
        "save_profile": "Save Preferences",
        "generate_ai_report": "Generate AI Report Summary",
        "export_pdf": "📄 Export Report as PDF"
    },
    "es": {
        "title": "🌿 Asistente de Ciudad Sostenible",
        "subtitle": "Pregunte sobre planificación urbana, tecnología verde y soluciones inteligentes para ciudades.",
        "home_welcome": "🌿 Bienvenido a su Asistente de Ciudad Inteligente",
        "highlights": "### 🧠 Destacados:",
        "chat": "🤖 Chatbot con IA",
        "urban_planning": "🏙️ Planificación Urbana",
        "green_energy": "🌳 Energía Verde",
        "transportation": "🚇 Transporte Urbano",
        "water_management": "💧 Gestión del Agua",
        "waste_management": "🗑️ Gestión de Residuos",
        "reports": "📊 Informes de Progreso",
        "settings": "⚙️ Configuración y Preferencias",
        "footer": "© 2025 Asistente de Ciudad Inteligente | Hecho con ❤️ usando Streamlit & Watsonx",
        "save_profile": "Guardar Preferencias",
        "generate_ai_report": "Generar Informe con IA",
        "export_pdf": "📄 Exportar Informe como PDF"
    },
    "fr": {
        "title": "🌿 Assistant Ville Durable",
        "subtitle": "Posez des questions sur l'urbanisme, les énergies vertes et les solutions intelligentes pour villes.",
        "home_welcome": "🌿 Bienvenue dans votre Assistant Ville Intelligente",
        "highlights": "### 🧠 Points forts :",
        "chat": "🤖 Chatbot avec IA",
        "urban_planning": "🏙️ Aménagement Urbain",
        "green_energy": "🌳 Énergie Renouvelable",
        "transportation": "🚇 Transports Urbains",
        "water_management": "💧 Gestion de l'Eau",
        "waste_management": "🗑️ Gestion des Déchets",
        "reports": "📊 Rapports de Suivi",
        "settings": "⚙️ Paramètres et Préférences",
        "footer": "© 2025 Assistant Ville Intelligente | Réalisé avec ❤️ en utilisant Streamlit & Watsonx",
        "save_profile": "Enregistrer les Préférences",
        "generate_ai_report": "Générer un Résumé IA",
        "export_pdf": "📄 Exporter le Rapport en PDF"
    }
}

# Page config
st.set_page_config(page_title="🌿 Smart City Assistant", layout="wide", page_icon="🌱")

# Custom CSS for modern UI
st.markdown("""
    <style>
        body { background-color: #f0f9f4; font-family: 'Segoe UI', sans-serif; }
        .main { background-color: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .card { background-color: #ffffff; padding: 25px; margin: 20px 0; border-left: 6px solid #2ecc71; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .navbar { display: flex; justify-content: center; gap: 20px; padding: 15px 0; background: linear-gradient(to right, #2ecc71, #27ae60); border-radius: 10px; margin-bottom: 25px; }
        .nav-button { background-color: #ffffff; color: #2ecc71; border: none; width: 50px; height: 50px; font-size: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.3s ease; }
        .nav-button:hover { background-color: #eafaf1; transform: scale(1.1); }
        h1, h2, h3 { color: #2c3e50; }
        label { font-weight: bold; color: #34495e; }
        input, select, textarea { border-radius: 8px; border: 1px solid #ccc; padding: 10px; width: 100%; font-size: 14px; }
        button { background-color: #2ecc71; color: white; border: none; padding: 10px 20px; font-size: 14px; border-radius: 8px; cursor: pointer; }
        button:hover { background-color: #27ae60; }
        .user-bubble, .bot-bubble { padding: 10px 15px; border-radius: 12px; max-width: 70%; margin: 6px 0; font-size: 14px; }
        .user-bubble { background-color: #d6eaff; align-self: flex-end; }
        .bot-bubble { background-color: #e6f0ff; align-self: flex-start; }
        .chat-container { display: flex; flex-direction: column; gap: 10px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "current_section" not in st.session_state:
    st.session_state.current_section = "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "city_data" not in st.session_state:
    st.session_state.city_data = {}
if "language" not in st.session_state:
    st.session_state.language = "en"

# Load Watsonx credentials
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]

    model_map = {
        "urban_planning": "ibm/granite-3-2-13b-instruct",
        "green_energy": "ibm/granite-3-2-8b-instruct",
        "transportation": "ibm/granite-3-2-code-instruct",
        "water_management": "ibm/granite-3-2-3b-instruct",
        "waste_management": "ibm/granite-3-2-8b-instruct",
        "chat": "ibm/granite-3-2-3b-instruct"
    }

    def get_llm(model_name):
        return WatsonxLLM(
            model_id=model_map[model_name],
            url=credentials.get("url"),
            apikey=credentials.get("apikey"),
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.TEMPERATURE: 0,
                GenParams.MIN_NEW_TOKENS: 5,
                GenParams.MAX_NEW_TOKENS: 250,
                GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
            },
        )

except KeyError:
    st.warning("⚠️ Watsonx credentials missing.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

# Navigation Bar
lang = st.session_state.language
st.markdown('<div class="navbar">', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    if st.button("챗", key="btn_chat"): st.session_state.current_section = "chat"
with col2:
    if st.button("🏙️", key="btn_urban"): st.session_state.current_section = "urban_planning"
with col3:
    if st.button("🌳", key="btn_green"): st.session_state.current_section = "green_energy"
with col4:
    if st.button("🚇", key="btn_transport"): st.session_state.current_section = "transportation"
with col5:
    if st.button("💧", key="btn_water"): st.session_state.current_section = "water_management"
with col6:
    if st.button("🗑️", key="btn_waste"): st.session_state.current_section = "waste_management"
with col7:
    if st.button("📊", key="btn_reports"): st.session_state.current_section = "reports"
st.markdown('</div>', unsafe_allow_html=True)

# Header
st.markdown(f'<h1 style="text-align:center;">{LANGUAGES[lang]["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="text-align:center; font-size:16px;">{LANGUAGES[lang]["subtitle"]}</p>', unsafe_allow_html=True)

# Function to export data as PDF
def export_smart_city_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Smart City Report", ln=True, align='C')
    pdf.ln(10)
    for key, value in st.session_state.city_data.items():
        pdf.cell(0, 10, txt=f"{key.capitalize()}: {value}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# ------------------------------ SETTINGS ------------------------------
if st.session_state.current_section == "settings":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2>⚙️ {LANGUAGES[lang]["settings"]}</h2>', unsafe_allow_html=True)
    language = st.selectbox("Language", options=["en", "es", "fr"], format_func=lambda x: {"en": "English", "es": "Español", "fr": "Français"}[x])
    theme = st.selectbox("Theme", ["Light", "Dark"])
    font_size = st.slider("Font Size", 12, 24)
    if st.button(LANGUAGES[lang]["save_profile"]):
        st.session_state.language = language
        st.success("Preferences updated!")
    st.markdown('</div>')

# ------------------------------ PROGRESS REPORTS ------------------------------
elif st.session_state.current_section == "reports":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2>📊 {LANGUAGES[lang]["reports"]}</h2>', unsafe_allow_html=True)
    traffic = st.slider("Traffic Volume", 0, 100000, step=1000)
    pollution = st.slider("Air Pollution Index", 0, 500)
    energy_use = st.slider("Energy Consumption (kWh)", 0, 10000)
    water_use = st.slider("Water Usage (L)", 0, 50000)
    waste_recycled = st.slider("Recycled Waste (%)", 0, 100)
    if st.button("Save Data"):
        st.session_state.city_data.update({
            "traffic_volume": traffic,
            "air_pollution": pollution,
            "energy_consumption": energy_use,
            "water_usage": water_use,
            "waste_recycled": waste_recycled
        })
        st.success("Data saved successfully.")
    if st.button(LANGUAGES[lang]["generate_ai_report"]):
        summary = get_llm("chat").invoke(f"Give a short sustainability report based on: {st.session_state.city_data}")
        st.markdown(f"🧠 **AI Analysis:**\n{summary}")
    if st.session_state.city_data:
        st.download_button(LANGUAGES[lang]["export_pdf"], data=export_smart_city_report(), file_name="smart_city_report.pdf", mime="application/pdf")
    st.markdown('</div>')

# ------------------------------ CHATBOT ------------------------------
elif st.session_state.current_section == "chat":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🤖 AI Chatbot</h2>', unsafe_allow_html=True)

    # Display chat messages
    for role, content in st.session_state.messages:
        bubble_class = "user-bubble" if role == "user" else "bot-bubble"
        st.markdown(f'<div class="{bubble_class}"><b>{role.capitalize()}:</b> {content}</div>', unsafe_allow_html=True)

    # Input form
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Your question:", placeholder="Type something like 'What makes a city sustainable?'...")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        st.session_state.messages.append(("user", user_input))
        with st.spinner("Thinking..."):
            try:
                llm = get_llm("chat")
                response = llm.invoke(user_input)
                st.session_state.messages.append(("assistant", response))
                st.rerun()
            except Exception as e:
                st.session_state.messages.append(("assistant", f"Error: {str(e)}"))
                st.rerun()
    st.markdown('</div>')

# ------------------------------ URBAN PLANNING ------------------------------
elif st.session_state.current_section == "urban_planning":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🏙️ Urban Planning Insights</h2>', unsafe_allow_html=True)
    query = st.text_area("Describe your city plan or ask a question:")
    if st.button("Get Advice"):
        llm = get_llm("urban_planning")
        res = llm.invoke(query)
        st.markdown(f"🧠 **AI Response:**\n{res}")
    st.markdown('</div>')

# ------------------------------ GREEN ENERGY ------------------------------
elif st.session_state.current_section == "green_energy":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🌳 Green Energy Suggestions</h2>', unsafe_allow_html=True)
    energy_query = st.text_input("Ask about solar, wind, or other green energy ideas:")
    if st.button("Generate Ideas"):
        llm = get_llm("green_energy")
        res = llm.invoke(energy_query)
        st.markdown(f"💡 **Suggestions:**\n{res}")
    st.markdown('</div>')

# ------------------------------ TRANSPORTATION ------------------------------
elif st.session_state.current_section == "transportation":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🚇 Smart Transportation Solutions</h2>', unsafe_allow_html=True)
    transport_query = st.text_input("Describe transportation issue or idea:")
    if st.button("Analyze Transport Idea"):
        llm = get_llm("transportation")
        res = llm.invoke(transport_query)
        st.markdown(f"🚀 **AI Analysis:**\n{res}")
    st.markdown('</div>')

# ------------------------------ WATER MANAGEMENT ------------------------------
elif st.session_state.current_section == "water_management":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>💧 Water Resource Planning</h2>', unsafe_allow_html=True)
    water_query = st.text_input("Ask about water systems or conservation:")
    if st.button("Get Water Plan"):
        llm = get_llm("water_management")
        res = llm.invoke(water_query)
        st.markdown(f"📊 **Plan:**\n{res}")
    st.markdown('</div>')

# ------------------------------ WASTE MANAGEMENT ------------------------------
elif st.session_state.current_section == "waste_management":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>🗑️ Waste Reduction Strategies</h2>', unsafe_allow_html=True)
    waste_query = st.text_input("Describe your waste challenge:")
    if st.button("Get Strategy"):
        llm = get_llm("waste_management")
        res = llm.invoke(waste_query)
        st.markdown(f"♻️ **Recommendations:**\n{res}")
    st.markdown('</div>')

# Footer
st.markdown(f'<p style="text-align:center; font-size:14px;">{LANGUAGES[lang]["footer"]}</p>', unsafe_allow_html=True)
