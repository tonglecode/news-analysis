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


class NewsAnalysisAssistant:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        if "news_data" not in st.session_state:
            st.session_state.news_data = []


class NewsCrawler:
    def __init__(self):
        self.header = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def crawl_news(self, keyword, days=7):
        news_list = []

        try:
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}"
            response = requests.get(url, headers=self.header)
            soup = BeautifulSoup(response.text, "html.parser")

            articles = soup.select("div.news_wrap.api_ani_send")

            for article in articles:
                try:
                    title = article.select_one("a.news_tit")
                    summary = article.select_one(
                        "a.api_txt_lines.dsc_txt_wrap")
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


class NewsAnalysisUI:
    def __init__(self, assistant):
        self.assistant = assistant

    def render_sidebar(self):
        st.sidebar.header("검색 설정")

        keywords = st.sidebar.text_area(
            "검색 키워드 (줄바꿈으로 구분)", "인공지능\n빅데이터\n메타버스"
        ).split("\n")

        days = st.sidebar.slider("검색 기간 (일)", 1, 30, 7)

        return keywords, days

    def render_main_page(self):
        st.title("📰 뉴스 분석 AI 어시스턴트")

        tab1, tab2, tab3 = st.tabs(["뉴스 수집", "데이터 분석", "AI 인사이트"])

        return tab1, tab2, tab3

    def render_news_collection(self, tab, keywords, days):
        with tab:
            if st.button("뉴스 수집 시작"):
                with st.spinner("뉴스를 수집하고 있습니다..."):
                    crawler = NewsCrawler()
                    all_news = []

                    progress_bar = st.progress(0)
                    for idx, keyword in enumerate(keywords):
                        news = crawler.crawl_news(keyword, days)
                        all_news.extend(news)
                        progress_bar.progress((idx + 1) / len(keyword))

                    st.session_state.news_data = all_news
                    st.success(f"총 {len(all_news)}개의 뉴스를 수집했습니다.")

            if st.session_state.news_data:
                st.dataframe(pd.DataFrame(st.session_state.news_data))
