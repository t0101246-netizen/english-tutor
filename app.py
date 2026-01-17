import streamlit as st
import asyncio
import edge_tts
import os
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import tempfile

st.set_page_config(page_title="Mama AI", page_icon="🇺🇸", layout="centered")
st.title("🇺🇸 Mama AI: Native English Environment")
st.caption("I am your American parent. Speak to me!")

# 獲取 API Key
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("請在 Streamlit 設定 Secrets: GROQ_API_KEY")
    st.stop()

client = Groq(api_key=api_key)

# 初始化
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system", 
        "content": "You are a patient American parent. I am your child. 1. NEVER speak Chinese. 2. Use simple words. 3. Correct me gently. 4. Keep answers short."
    }]

# TTS 函數
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "en-US-AnaNeural", rate="-10%")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        await communicate.save(fp.name)
        return fp.name

# 介面
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

st.write("---")
c1, c2 = st.columns([1, 3])
with c1:
    audio = mic_recorder(start_prompt="🔴 Speak", stop_prompt="⏹️ Stop", key='recorder')

if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(audio['bytes'])
        audio_filename = fp.name

    try:
        # 聽
        with open(audio_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_filename, file.read()),
                model="distil-whisper-large-v3-en",
                response_format="json"
            )
        user_text = transcription.text
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        # 想
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=st.session_state.messages,
            temperature=0.7,
            max_tokens=200,
        )
        ai_response = completion.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.write(ai_response)

        # 說
        audio_file = asyncio.run(text_to_speech(ai_response))
        st.audio(audio_file, format="audio/mp3", autoplay=True)

    except Exception as e:
        st.error(f"Error: {str(e)}"
