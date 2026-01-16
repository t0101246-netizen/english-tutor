import streamlit as st
import asyncio
import edge_tts
import os
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import tempfile

# --- 設定頁面 ---
st.set_page_config(page_title="My AI American Parent", page_icon="🇺🇸")
st.title("🇺🇸 Immersive English Environment")
st.caption("Speak to me like a child. I will teach you.")

# --- 初始化 Groq ---
# 這裡會自動從雲端環境讀取密碼
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# --- 核心大腦設定 (System Prompt) ---
system_prompt = """
You are a patient, warm American parent. I am your 5-year-old child learning to speak English.
Your Goal: Create a 100% English immersion environment.

Rules:
1. NEVER speak Chinese. Even if I speak Chinese, guess what I mean and reply in English.
2. Use simple vocabulary (CEFR A1/A2 level).
3. Speak slowly and clearly.
4. If I make a grammar mistake, gently repeat the correct sentence (Recasting).
5. Keep answers short (1-2 sentences).
6. Always end with a simple question to encourage me to speak more.
"""

# --- 初始化對話紀錄 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# --- 函數：文字轉語音 (TTS) ---
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "en-US-AnaNeural", rate="-10%") # 使用溫柔女聲，語速放慢
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        await communicate.save(fp.name)
        return fp.name

# --- 介面佈局 ---
# 1. 錄音按鈕
c1, c2 = st.columns([1, 3])
with c1:
    st.write("### 🗣️ Speak:")
    # 錄音組件
    audio = mic_recorder(
        start_prompt="🔴 Record",
        stop_prompt="⏹️ Stop",
        key='recorder'
    )

# 2. 處理邏輯
if audio:
    # 存下錄音檔
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(audio['bytes'])
        audio_filename = fp.name

    # A. 耳朵：用 Groq Whisper 聽懂你說什麼
    try:
        with open(audio_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_filename, file.read()),
                model="distil-whisper-large-v3-en", # 免費且強大的聽力模型
                response_format="json"
            )
        user_text = transcription.text
        st.success(f"You said: {user_text}")

        # B. 大腦：思考回應
        st.session_state.messages.append({"role": "user", "content": user_text})
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192", # 免費且極速的模型
            messages=st.session_state.messages,
            temperature=0.7,
            max_tokens=1024,
        )
        ai_response = completion.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        st.info(f"Mom says: {ai_response}")

        # C. 嘴巴：合成語音
        audio_file = asyncio.run(text_to_speech(ai_response))
        st.audio(audio_file, format="audio/mp3", autoplay=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# 顯示歷史對話 (可選)
with st.expander("Conversation History"):
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.write(f"**{msg['role']}**: {msg['content']}")