import streamlit as st                                   # Streamlit 웹앱 프레임워크
from audiorecorder import audiorecorder                  # 브라우저에서 음성 녹음하는 라이브러리
from openai_service import stt, ask_gpt, tts             # 우리가 만든 STT, GPT, TTS 함수 import

def main():                                              # 프로그램 시작 함수
    st.set_page_config(                                  # Streamlit 페이지 기본 설정
        page_title='Voice Chatbot',                      # 브라우저 탭 제목
        page_icon='🎤',                                  # 브라우저 탭 아이콘
        layout='wide'                                    # 화면을 넓게 사용
    )

    st.header('🎤Voice Chatbot🎤')                      # 화면 상단 제목 출력
    st.markdown('---')                                   # 구분선 출력

    with st.expander('Voice Chatbot 프로그램 처리절차', expanded=False):  # 접었다 펼 수 있는 설명창
        st.write("""                                     # 프로그램 동작 순서 안내 텍스트
            1. 녹음하기 버튼을 눌러 질문을 녹음합니다.
            2. 녹음이 완료되면 자동으로 Whisper모델을 이용해 음성을 텍스트로 변환합니다. 
            3. 변환된 텍스트로 LLM에 질의후 응답을 받습니다.
            4. LLM의 응답을 다시 TTS모델을 사용해 음성으로 변환하고 이를 사용자에게 들려줍니다.
            5. 모든 질문/답변은 채팅형식의 텍스트로 제공합니다.
        """)

    system_prompt = '당신은 친절한 챗봇입니다. 사용자의 질문에 50단어 이내로 간결하게 답변해주세요.'  # GPT 기본 성격 설정

    if 'messages' not in st.session_state:               # 세션에 messages가 없으면
        st.session_state['messages'] = [                 # 최초 대화 기록 생성
            {'role': 'system', 'content': system_prompt} # system 역할로 프롬프트 저장
        ]

    if 'check_reset' not in st.session_state:            # 초기화 체크 변수 없으면
        st.session_state['check_reset'] = False          # 기본값 False

    with st.sidebar:                                    # 사이드바 영역
        model = st.radio(                               # 라디오 버튼으로 모델 선택
            label='GPT 모델',
            options=['gpt-4.1-mini', 'gpt-5-nano', 'gpt-5.2'],
            index=0
        )

        if st.button(label='초기화'):                        # 초기화 버튼 클릭 시
            st.session_state['messages'] = [                # messages를 다시 system만 남김
                {'role': 'system', 'content': system_prompt}
            ]
            st.session_state['check_reset'] = True          # 초기화 상태 True

    col1, col2 = st.columns(2)                          # 화면을 좌/우 두 컬럼으로 분리

    with col1:                                          # 왼쪽 컬럼
        st.subheader('녹음하기')                         # 소제목
        audio = audiorecorder()                         # 녹음 UI 생성

    # 녹음이 1초이상 있고, 리셋 직후가 아니면
    if (audio.duration_seconds > 0) and (not st.session_state['check_reset']):  # 녹음이 있고 초기화가 아닐 때만 실행
        st.audio(audio.export().read())                 # 녹음한 음성 재생
        query: str = stt(audio)                        # STT 실행 : 음성 → 텍스트
        st.session_state['messages'].append({          # 사용자 질문 messages에 추가
            'role': 'user',
            'content': query
        })

        response: str = ask_gpt(st.session_state['messages'], model)  # GPT에게 질문 보내고 답변 받기
        st.session_state['messages'].append({          # GPT 응답 messages에 추가
            'role': 'assistant',
            'content': response
        })

        base64_encoded_audio: str = tts(response)      # TTS 실행 : 텍스트 → Base64 인코딩된 오디오
        st.markdown(f"""                               # HTML 오디오 태그 삽입
            <audio autoplay>
                <source src="data:audio/mp3;base64,{base64_encoded_audio}">
            </audio>
        """, unsafe_allow_html=True)

    else:
        st.session_state['check_reset'] = False        # 초기화 상태 해제

    with col2:                                         # 오른쪽 컬럼
        st.subheader('질문/답변')                       # 소제목

        if (audio.duration_seconds > 0) and (not st.session_state['check_reset']):  # 녹음 후일 때만 출력
            for message in st.session_state['messages']:  # 모든 대화 기록 순회
                role = message['role']                  # 역할 추출 (user/assistant/system)
                content = message['content']            # 내용 추출
                if role == 'system':                   # system 메시지는 화면에 출력하지 않음
                    continue
                with st.chat_message(role):            # 채팅 UI에 맞게 출력
                    st.markdown(content)               # 메시지 내용 출력
if __name__ == '__main__':                            # 프로그램 시작 지점
    main()                                            # main 함수 실행