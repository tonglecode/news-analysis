import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import plotly.express as px
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import re
import os


def convert_relative_date(date_str):
    """상대적 날짜(예: '1일 전', '3시간 전')를 절대 날짜로 변환"""
    date_str = str(date_str).strip()

    # 날짜 패턴이 아닐 경우 None 반환
    if not re.search(r"\d{4}-\d{2}-\d{2}|(\d+일 전)|(\d+시간 전)", date_str):
        return None

    now = datetime.now()
    days_match = re.search(r"(\d+)일 전", date_str)
    hours_match = re.search(r"(\d+)시간 전", date_str)

    if days_match:
        days_ago = int(days_match.group(1))
        return (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    elif hours_match:
        hours_ago = int(hours_match.group(1))
        return (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M")

    return date_str  # 이미 날짜 형식이면 그대로 반환


class NewsAnalysisAssistant:
    def __init__(self):
        self.setup_credentials()
        self.initialize_session_state()

    def setup_credentials(self):
        """API 키 설정"""
        st.sidebar.header("API 설정")
        api_service = st.sidebar.selectbox(
            "사용할 AI 서비스", ["Google Gemini", "OpenAI GPT"]
        )
        api_key = st.sidebar.text_input("API Key", type="password")

        if api_key:
            if api_service == "Google Gemini":
                os.environ["GOOGLE_API_KEY"] = api_key
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )  # 수정된 부분

            elif api_service == "OpenAI GPT":
                os.environ["OPENAI_API_KEY"] = api_key
                self.llm = ChatOpenAI(model_name="gpt-4")
                self.embeddings = OpenAIEmbeddings()

    def initialize_session_state(self):
        """세션 상태 초기화"""
        if "news_data" not in st.session_state:
            st.session_state.news_data = []
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None


class NewsCrawler:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def crawl_news(self, keyword, days=7):
        """뉴스 기사 크롤링"""
        news_list = []

        try:
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}"
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, "html.parser")

            articles = soup.select("div.news_wrap.api_ani_send")

            for article in articles:
                try:
                    title = article.select_one("a.news_tit")
                    summary = article.select_one("a.api_txt_lines.dsc_txt_wrap")
                    press = article.select_one("a.info.press")
                    date = article.select_one("span.info")

                    if all([title, summary, press, date]):
                        news_list.append(
                            {
                                "title": title.text,
                                "summary": summary.text,
                                "press": press.text,
                                "date": date.text,
                                "keyword": keyword,
                            }
                        )
                except Exception as e:
                    continue

            return news_list

        except Exception as e:
            st.error(f"크롤링 중 오류 발생: {str(e)}")
            return []


class DataProcessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def process_news_data(self, news_list):
        """뉴스 데이터 전처리"""
        try:
            # DataFrame 생성
            df = pd.DataFrame(news_list)

            # 텍스트 결합
            texts = []
            for _, row in df.iterrows():
                text = f"제목: {row['title']}\n요약: {row['summary']}\n"
                texts.append(text)

            # 텍스트 분할
            doc_chunks = self.text_splitter.split_text("\n\n".join(texts))

            # 벡터 저장소 생성
            vectorstore = FAISS.from_texts(doc_chunks, self.embeddings)

            return df, vectorstore

        except Exception as e:
            st.error(f"데이터 처리 중 오류 발생: {str(e)}")
            return None, None

    def analyze_trends(self, df):
        """트렌드 분석"""
        try:
            # 언론사별 기사 수
            press_counts = df["press"].value_counts()

            # 날짜별 기사 수
            df["date"] = pd.to_datetime(df["date"])
            date_counts = df["date"].dt.date.value_counts()

            # 키워드별 기사 수
            keyword_counts = df["keyword"].value_counts()

            return {
                "press_counts": press_counts,
                "date_counts": date_counts,
                "keyword_counts": keyword_counts,
            }

        except Exception as e:
            st.error(f"트렌드 분석 중 오류 발생: {str(e)}")
            return None


class NewsAnalysisUI:
    def __init__(self, assistant):
        self.assistant = assistant

    def render_sidebar(self):
        """사이드바 UI"""
        st.sidebar.header("검색 설정")

        keywords = st.sidebar.text_area(
            "검색 키워드 (줄바꿈으로 구분)", "인공지능\n빅데이터\n메타버스"
        ).split("\n")

        days = st.sidebar.slider("검색 기간 (일)", 1, 30, 7)

        return keywords, days

    def render_main_page(self):
        """메인 페이지 UI"""
        st.title("📰 뉴스 분석 AI 어시스턴트")

        tab1, tab2, tab3 = st.tabs(["뉴스 수집", "데이터 분석", "AI 인사이트"])

        return tab1, tab2, tab3

    def render_news_collection(self, tab, keywords, days):
        """뉴스 수집 탭"""
        with tab:
            if st.button("뉴스 수집 시작"):
                with st.spinner("뉴스를 수집하고 있습니다..."):
                    crawler = NewsCrawler()
                    all_news = []

                    progress_bar = st.progress(0)
                    for idx, keyword in enumerate(keywords):
                        news = crawler.crawl_news(keyword, days)
                        all_news.extend(news)
                        progress_bar.progress((idx + 1) / len(keywords))

                    st.session_state.news_data = all_news
                    st.success(f"총 {len(all_news)}개의 뉴스를 수집했습니다!")

            if st.session_state.news_data:
                st.dataframe(pd.DataFrame(st.session_state.news_data))

    def render_data_analysis(self, tab):
        """데이터 분석 탭"""
        with tab:
            if not st.session_state.news_data:
                st.warning("먼저 뉴스를 수집해주세요.")
                return

            df = pd.DataFrame(st.session_state.news_data)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("키워드별 뉴스 분포")
                fig = px.pie(df, names="keyword")
                st.plotly_chart(fig)

            with col2:
                st.subheader("언론사별 뉴스 수")
                fig = px.bar(df["press"].value_counts().head(10))
                st.plotly_chart(fig)

            st.subheader("일자별 뉴스 추이")
            df["date"] = df["date"].apply(convert_relative_date)

            # 유효한 날짜만 필터링
            df = df.dropna(subset=["date"])

            # datetime 변환
            df["date"] = pd.to_datetime(
                df["date"], format="mixed", errors="coerce"
            )  # 날짜혼용 "YYYY-MM-DD" 형식 일부 날짜는 "YYYY-MM-DD HH:MM" 형식
            daily_counts = df.groupby([df["date"].dt.date, "keyword"]).size().unstack()
            fig = px.line(daily_counts)
            st.plotly_chart(fig)

    def render_ai_insights(self, tab):
        """AI 인사이트 탭"""
        with tab:
            if not st.session_state.news_data:
                st.warning("먼저 뉴스를 수집해주세요.")
                return

            if not st.session_state.vectorstore:
                with st.spinner("데이터를 분석하고 있습니다..."):
                    processor = DataProcessor(self.assistant.embeddings)
                    df, vectorstore = processor.process_news_data(
                        st.session_state.news_data
                    )
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore

            st.subheader("뉴스 데이터 분석")
            query = st.text_input(
                "질문을 입력하세요 (예: 최근 주요 트렌드는 무엇인가요?)"
            )

            if query and st.session_state.vectorstore:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.assistant.llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(),
                )

                with st.spinner("분석 중..."):
                    response = qa_chain.run(query)
                    st.write(response)
