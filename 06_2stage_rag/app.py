import streamlit as st
from ai_wine_sommelier import ai_wine_sommelier_rag

# 제목과 설명
st.title("🍷AI Wine Sommelier🍷")
st.write("🍖음식 이미지 URL을 작성하면, 어울리는 와인🍷을 추천해드립니다.")

# 폼 생성
with st.form(key="img_form"):
    img_url = st.text_input("이미지 URL 입력:", placeholder="예: https://example.com/food.jpg")
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    if img_url:
        try:
            # 이미지 노출
            st.image(img_url)

            # AI 메시지 출력 공간
            st.subheader("AI 와인 추천:")
            # Spinner 처리
            with st.spinner("와인 검색중..."):
                query = {
                    'text': '',
                    'image_urls': [img_url]
                }
                gen_response = ai_wine_sommelier_rag(query)
                st.write_stream(gen_response) # stream객체 전달
        except Exception as e:
            st.error(f"이미지를 로드하는 중 오류가 발생했습니다: {e}")
    else:
        st.warning("이미지 URL을 입력해주세요!")