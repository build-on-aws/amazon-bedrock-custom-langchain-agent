import time

import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

import agent_aws

st.title("Agent AWS")


@st.cache_resource
def load_llm():
    return agent_aws.setup_full_agent()


model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help??"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        st_callback = StreamlitCallbackHandler(st.container())

        result = agent_aws.interact_with_agent_st(
            model, prompt, st.session_state.messages, st_callback
        )

        # Simulate stream of response with milliseconds delay
        for chunk in result.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
