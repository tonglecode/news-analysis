import streamlit as st
from classes import NewsAnalysisUI, NewsAnalysisAssistant


def main():
    # 앱 설정
    st.set_page_config(
        page_title="뉴스 분석 AI 어시스턴트", page_icon="🐳", layout="wide"
    )

    # 인스턴스 생성
    assistant = NewsAnalysisAssistant()
    ui = NewsAnalysisUI(assistant)

    # UI 렌더링
    keywords, days = ui.render_sidebar()
    tab1, tab2, tab3 = ui.render_main_page()

    # 각 탭 렌더링
    ui.render_news_collection(tab1, keywords, days)
    ui.render_data_analysis(tab2)


if __name__ == "__main__":
    main()
