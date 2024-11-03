import streamlit as st
from langchain_core.messages import ChatMessage
from agent import modelCreation
from openai import OpenAI
import pygame
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tempfile
from transformers import pipeline
from audiorecorder import audiorecorder




st.set_page_config(page_title="Ready2Interview", page_icon="💬")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small")
emotion_recognizer = pipeline(model="superb/wav2vec2-base-superb-er", task="audio-classification")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()



audio = audiorecorder("🎤", "🛑")

if len(audio) > 0: 

    # To save audio to a file, use pydub export method:
    audio.export("audio.wav", format="wav")

def print_history():
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)


def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

def read_outloud(stream_response):
    client = OpenAI()

    # Create a temporary file for speech.mp3
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        speech_file_path = temp_file.name

    response = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=stream_response
    )

    # Stream the response directly to the file
    response.stream_to_file(speech_file_path)


    play_mp3(speech_file_path)


def play_mp3(path):
    pygame.mixer.init()  # Initialize the mixer module
    pygame.mixer.music.load(str(path))  # Load the MP3 file
    pygame.mixer.music.play()  # Play the MP3 file

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

def generate_question():
    add_history("user", "give me a single interview question")
    st.chat_message("user").write("give me a single interview question")

    with st.chat_message("assistant"):
        chat_container = st.empty()

        with_message_history = RunnableWithMessageHistory(
            st.session_state["chain"],
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # 스트림 응답을 처리하여 결과를 생성
        stream_response = with_message_history.invoke(
            {"input": "give me a single interview question"},
            config={"configurable": {"session_id": "abc123"}},
        )["output"]

        ai_answer = ""
        for chunk in stream_response:
            ai_answer += chunk
            if st.session_state.get("sound_button") == "Sound":
                chat_container.markdown("Sound 옵션이 선택되었습니다.")
            else:
                chat_container.markdown(ai_answer)

        # AI 응답을 기록하고, 필요 시 읽기 기능 수행
        add_history("ai", stream_response)
        if st.session_state.get("sound_button") != "Text":
            read_outloud(stream_response)

 # 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환




def recognize_emotion(audio_file_path):
    # 감정 예측
    emotions = emotion_recognizer(audio_file_path)

    for emotion in emotions:
        label = emotion['label']
        score = emotion['score']
        print(f"Emotion: {label}, Confidence: {score:.2f}")

    return emotions[0]['label']






uploaded_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")
if not uploaded_file:
    st.warning("Please upload a PDF file.")
else:
    with st.sidebar:
        st.image("./Logo.png",width=300)
        
        # 인터뷰 유형 및 기타 설정 입력
        role_type = st.selectbox("Role", ["Software Engineer", "Data Scientist", "MLops Engineer"])
        interviewer_type = st.selectbox("interviewer", ["hiring manager", "technical lead"])
        company = st.text_input("Company", value="Meta")
        sound_button = st.selectbox("Text/Sound", ["Text", "Sound", "Both"])
        st.session_state["sound_button"] = sound_button

        submit_btn = st.button(" Save changes ")
        clear_btn = st.button("Clear chat history")

        # # 버튼 텍스트를 세션 상태로 초기화
        # if "button_text" not in st.session_state:
        #     st.session_state["button_text"] = "Sound only"

        # # 버튼 클릭 이벤트 처리
        # if st.button(st.session_state["button_text"]):
        #     # 버튼 클릭 시 텍스트 변경
        #     if st.session_state["button_text"] == "Sound only":
        #         st.session_state["button_text"] = "Read only "
        #     else:
        #         st.session_state["button_text"] = "Sound only"

    stream_response = "This is a sample response text to be read out loud."


    if clear_btn:
        retriever = st.session_state["messages"].clear()

        
    if submit_btn:
        chain = modelCreation(role_type, interviewer_type, company, uploaded_file)
        del st.session_state["store"]   
        if chain:
            st.session_state["chain"] = chain
            print(f"Chain created {company}")

    if "chain" not in st.session_state:
        st.session_state["chain"] = modelCreation(role_type, interviewer_type, company, uploaded_file)


    print_history()

    # modelCreation 함수 수정은 이전 코드에서 동일

   

    

    

    if user_input := st.chat_input():
        add_history("user", user_input)
        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            chat_container = st.empty()

            with_message_history = (
                RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
                    st.session_state["chain"],  # 실행할 Runnable 객체
                    get_session_history,  # 세션 기록을 가져오는 함수
                    input_messages_key="input",  # 입력 메시지의 키
                    history_messages_key="history",  # 기록 메시지의 키
                )
            )

            stream_response = with_message_history.invoke(
                {"input": "Answer of your question: " + user_input},
                config = {"configurable": {"session_id": "abc123"}},
                )["output"]
            ai_answer = ""
            for chunk in stream_response:
                ai_answer += chunk
                if st.session_state["sound_button"] == "Sound":
                    chat_container.markdown("Sound 옵션이 선택되었습니다.")
                else:
                    chat_container.markdown(ai_answer)
                    
            add_history("ai", stream_response)
            if not st.session_state["sound_button"] == "Text":
                read_outloud(stream_response)


    if "chain" in st.session_state:

        # 버튼 클릭 시 실행되는 함수


        # 텍스트와 버튼을 위한 빈 컨테이너를 생성
        text_container = st.container()

        # # 텍스트를 먼저 렌더링
        

        # 버튼을 텍스트 아래에서 렌더링
        if st.button("Generate Interview Questions"):
            with text_container:
                generate_question()  # 텍스트 출력 및 질문 생성

        # `Record your response` 버튼 클릭 시 녹음 버튼 표시
        if st.button("Answer by recorded audio"):
                        
            audio_path = "./audio.wav"

            # Hugging Face의 Whisper 모델 불러오기
            

            # 텍스트 변환
            transcription = transcriber(audio_path)["text"]
            emotion = recognize_emotion(audio_path)

            audioPrompt = f"{transcription} \n\n Emotion: {emotion}"
            
            add_history("user", transcription)
            st.chat_message("user").write(transcription)
            with st.chat_message("assistant"):
                chat_container = st.empty()

                with_message_history = (
                    RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
                        st.session_state["chain"],  # 실행할 Runnable 객체
                        get_session_history,  # 세션 기록을 가져오는 함수
                        input_messages_key="input",  # 입력 메시지의 키
                        history_messages_key="history",  # 기록 메시지의 키
                    )
                )

                stream_response = with_message_history.invoke(
                    {"input": "Answer of your question: " + audioPrompt},
                    config = {"configurable": {"session_id": "abc123"}},
                    )["output"]
                ai_answer = ""
                for chunk in stream_response:
                    ai_answer += chunk
                    if st.session_state["sound_button"] == "Sound":
                        chat_container.markdown("Sound 옵션이 선택되었습니다.")
                    else:
                        chat_container.markdown(ai_answer)
                        
                add_history("ai", stream_response)
                if not st.session_state["sound_button"] == "Text":
                    read_outloud(stream_response)         
            

        
    