# STT(Speech -> Text) + GPT 응답 + TTS(Text -> Speech) 파이프라인 함수 모음
import base64                       # mp3 이진데이터를 base64 문자열로 인코딩
from dotenv import load_dotenv      # .env 환경변수 로드
from openai import OpenAI           # OpenAI 클라이언트 클래스
import os

load_dotenv()       # .env 파일을 읽어서 환경변수 등록
OPENAI_API_KEY = os.environ['openai_key']   # .env에 저장된 openai_key 값을 가져옴
client = OpenAI(api_key = OPENAI_API_KEY)   # OpenAI 클라이언트 객체 생성 (키 직접 주입)

# 오디오 객체를 Whisper로 SST 하여 텍스트로 변환하는 함수
def stt(audio):
    output_filepath = 'input.mp3'                   # 임시 저장용 파일
    audio.export(output_filepath, format='mp3')     # 오디오 객체를 mp3 파일로 저장
    
    with open(output_filepath, 'rb') as f:          # 저장도니 파일을 바이너리로 열기
        # STT 요청
        transcription = client.audio.transcriptions.create(
            model = 'whisper-1',
            file=f
        )
    os.remove(output_filepath)  # 임시로 만든 파일 삭제
    
    return transcription.text   # STTS결과 텍스트 반환

# 메시지 히스토리와 모델을 받아 해당 GPT로 응답을 생성하는 함수
def ask_gpt(messages, model):
    # GPT 채팅 응답 반환
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,              # 생성 다양상 (높을수록 랜덤)
        top_p=1,                    # nucleus sampling(1이면 제한 없음)
        max_tokens=4096   # 생성 토큰 최대치
    ).choices[0].message.content    # 첫 번쨰 응답 텍스트
    
# 텍스트를 TTS로 mp3 생성후 base64 문자열로 반환하는 함수
def tts(response:str):
    filename = 'output.mp3'     # TTS 결과 mp3 파일명 지정
    # 스트리밍 방식의 TTS 요청
    with client.audio.speech.with_streaming_response.create(
        model = 'tts-1',
        voice = 'alloy',        # 음성 톤/케릭터
        input = response        # 음성으로 변환할 텍스트
    ) as resp:
        resp.stream_to_file(filename)   # 스트리밍 결과를 mp3 파일로 저장
        
    with open(filename, 'rb') as f:     # 생성된 mp3 파일을 바이너리로 읽기
        data = f.read()                 # mp3 이진데이터 일ㄺ
        b64_encoded = base64.b64encode(data).decode()   # 이진 -> base64 이진 -> 문자열로 디코딩
        
    os.remove(filename) # 생성한 출력 mp3 파일 삭제
    
    return b64_encoded  # base64 문자열 반환 (웹/엡에서 바로 재생용)

################################################################################################
# 바이너리(binary)는 사람이 읽는 문자(text)가 아니라, 0과 1 (바이트로 된 원본 데이터
# mp3, jpg, png, pdf 같은 파일은 대부분 바이너리 파일이다.
# - 텍스트 형식 : 사람이 읽을 수 있는 글자 데이터(예: "hello", JSON 문자열, CSV내용)
# - 바이너리 형식 : 파일 자체의 원본 바이트(bytes) 데이터(예: mp3, jpg, png, pdf, zip 압축 데이터)

# 업로드된 음성 파일을 텍스트로 변환(STT)해서 변환하는 함수
def stt_file(uploaded_file) -> str:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=(uploaded_file.name, uploaded_file.getvalue())     # (파일명, 파일바이트) 튜플
    )
    return transcription.text   # STT 결과 테스트만 반환