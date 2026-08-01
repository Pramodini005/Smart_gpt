import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ---------------------------------------------------------
# System Prompt
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are PragyanAI Assistant.

You are an expert AI assistant.

Answer professionally.

Use markdown.

If you don't know the answer,
say you don't know.
"""


# ---------------------------------------------------------
# Get Groq API Key
# ---------------------------------------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None


# ---------------------------------------------------------
# Check API Key
# ---------------------------------------------------------
if not groq_api_key:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Please add it to Streamlit Secrets."
    )
    st.stop()


# ---------------------------------------------------------
# Initialize LLM
# ---------------------------------------------------------
@st.cache_resource
def load_llm():

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
        api_key=groq_api_key
    )


llm = load_llm()


# ---------------------------------------------------------
# Initialize Chat History
# ---------------------------------------------------------
if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 I am PragyanAI Assistant. How can I help you?"
        }
    ]


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🤖 PragyanAI Assistant")

st.markdown(
    """
    **Powered by LangChain + Groq + Llama 3.3 70B**
    """
)


# ---------------------------------------------------------
# Clear Chat Button
# ---------------------------------------------------------
if st.button("🗑️ Clear Chat"):

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 I am PragyanAI Assistant. How can I help you?"
        }
    ]

    st.rerun()


# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------
user_input = st.chat_input("Ask anything...")


if user_input:

    # Display user message
    with st.chat_message("user"):

        st.markdown(user_input)

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # -----------------------------------------------------
    # Build LangChain Messages
    # -----------------------------------------------------
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    for message in st.session_state.messages:

        if message["role"] == "user":

            messages.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":

            messages.append(
                AIMessage(
                    content=message["content"]
                )
            )

    # -----------------------------------------------------
    # Generate Response
    # -----------------------------------------------------
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = llm.invoke(messages)

                answer = response.content

            except Exception as e:

                answer = f"Error: {str(e)}"

        st.markdown(answer)

    # -----------------------------------------------------
    # Store Assistant Response
    # -----------------------------------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
