import streamlit as st
from classes import NewsAnalysisAssistant, NewsAnalysisUI
import os


def main():
    # 앱 설정
    st.set_page_config(
        page_title="뉴스 분석 AI 어시스턴트", page_icon="📰", layout="wide"
    )

    # 인스턴스 생성
    assistant = NewsAnalysisAssistant()
    ui = NewsAnalysisUI(assistant)

    # API 키 확인
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = st.text_input(
            "OpenAI API 키 입력:", type="password"
        )
        return

    # UI 렌더링
    keywords, days = ui.render_sidebar()
    tab1, tab2, tab3 = ui.render_main_page()

    # 각 탭 렌더링
    ui.render_news_collection(tab1, keywords, days)
    ui.render_data_analysis(tab2)
    ui.render_ai_insights(tab3)


if __name__ == "__main__":
    main()
