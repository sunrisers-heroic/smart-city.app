import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Page config
st.set_page_config(page_title="🌿 Sustainability Smart City Assistant", layout="centered", page_icon="🌱")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f4f8fb;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .chat-box {
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 10px;
        }
        .user-bubble {
            background-color: #dcf8c6;
            align-self: flex-end;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 70%;
            word-wrap: break-word;
        }
        .ai-bubble {
            background-color: #ececec;
            align-self: flex-start;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 70%;
            word-wrap: break-word;
        }
        .message-container {
            display: flex;
            flex-direction: column;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌿 Sustainability Smart City Assistant")
st.markdown("Ask anything about smart cities, sustainability, green tech, or urban planning!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Watsonx credentials from secrets.toml
try:
    credentials = {
        "url": st.secrets["WATSONX_URL"],
        "apikey": st.secrets["WATSONX_APIKEY"]
    }
    project_id = st.secrets["WATSONX_PROJECT_ID"]

    # Initialize Watsonx LLM
    llm = WatsonxLLM(
        model_id="ibm/granite-3-2-8b-instruct",
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
    st.warning("⚠️ Watsonx credentials are missing. Please set them in `.streamlit/secrets.toml` or Streamlit Cloud.")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error initializing LLM: {str(e)}")
    st.stop()

# Display chat messages
def render_chat():
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                st.markdown(f'<div class="message-container"><div class="user-bubble"><b>You:</b> {message.content}</div></div>', unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.markdown(f'<div class="message-container"><div class="ai-bubble"><b>Bot:</b> {message.content}</div></div>', unsafe_allow_html=True)

# Chat input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Your question:", placeholder="Type something like 'What makes a city sustainable?'...")
    submit_button = st.form_submit_button(label="Send")

# Handle submission
if submit_button and user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    try:
        ai_response = llm.invoke(user_input)
    except Exception as e:
        ai_response = f"Error generating response: {str(e)}"
    
    st.session_state.messages.append(AIMessage(content=ai_response))

# Render chat history
render_chat()
