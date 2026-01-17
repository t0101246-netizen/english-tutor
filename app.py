import streamlit as st
import asyncio
import edge_tts
import os
from groq import Groq
from streamlit_mic_recorder import mic_recorder
import tempfile

# --- 1. 頁面與字體設定 (CSS 魔法) ---
st.set_page_config(page_title="Mama AI", page_icon="🇺🇸", layout="centered")

# 這裡強制把字體放大到 24px，讓你不用戴老花眼鏡
st.markdown("""
    <style>
    .stChatMessage p {
        font-size: 24px !important;
        line-height: 1.5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🇺🇸 Mama AI")
st.caption("Press Start -> Speak English -> Press Stop")

# --- 2. 獲取 API Key ---
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("請在 Streamlit 設定 Secrets: GROQ_API_KEY")
    st.stop()

client = Groq(api_key=api_key)

# --- 3. 初始化 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system", 
        "content": "You are a patient American parent. I am your child. 1. NEVER speak Chinese. 2. Use simple words. 3. Correct me gently. 4. Keep answers short."
    }]

# --- 4. TTS 函數 ---
async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, "en-US-AnaNeural", rate="-10%")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        await communicate.save(fp.name)
        return fp.name

# --- 5. 介面顯示 ---
# 顯示歷史對話
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

st.write("---")
st.write("### 👇 Click to Speak (按一下開始，講完按停止)")

# 錄音按鈕
c1, c2 = st.columns([1, 3])
with c1:
    audio = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop & Send",
        key='recorder'
    )

# --- 6. 處理邏輯 ---
if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(audio['bytes'])
        audio_filename = fp.name

    try:
        # 聽 (Whisper Large V3)
        with open(audio_filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(audio_filename, file.read()),
                model="whisper-large-v3",
                response_format="json"
            )
        user_text = transcription.text
        
        # 防呆機制：如果沒聽到聲音或聽到幻覺，就不回應
        if len(user_text) < 2 or "Halo" in user_text or "Amara" in user_text:
            st.warning("⚠️ I didn't hear you clearly. Please speak louder! (沒聽清楚，請大聲一點)")
        else:
            # 顯示使用者說的話
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.write(user_text)

            # 想 (Llama 3.1)
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.messages,
                temperature=0.7,
                max_tokens=200,
            )
            ai_response = completion.choices[0].message.content
            
            # 顯示 AI 回應
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.write(ai_response)

            # 說 (Edge TTS)
            audio_file = asyncio.run(text_to_speech(ai_response))
            st.audio(audio_file, format="audio/mp3", autoplay=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
