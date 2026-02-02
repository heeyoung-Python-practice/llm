import streamlit as st          # ← 이 줄이 빠졌음
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("openai_key")  # 🔥 이것도 중요 (아래 설명)
client = OpenAI(api_key=api_key)

# 세션 초기화 (안 하면 또 에러남)
if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.messages = st.session_state.messages[-10:]

@st.cache_resource
def load_model():
    return client


st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("기본 챗봇을 streamlit으로 구현한다.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "너는 친절한 한국어 챗봇이다."}
    ]

# 대화 출력
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 입력
if prompt := st.chat_input("대화내용 입력"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=st.session_state.messages
    )

    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)

    # 히스토리 제한
    st.session_state.messages = st.session_state.messages[-12:]