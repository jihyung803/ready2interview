import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from PyPDF2 import PdfReader
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

import tempfile

load_dotenv()


def modelCreation(role = "Software Engineer", interviewer = "Hiring manager", company = "Meta", uploaded_file = None):
        # 사용자에게 파일 업로드 요청
    if not uploaded_file:
        st.warning("Please upload a PDF file.")
        return

    if uploaded_file is not None:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # PyMuPDFLoader로 임시 파일 경로에서 PDF 파일 로드
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()  # 이 단계에서 문서 객체 형태로 변환됩니다.


    # # PDF 파일 로드
    # loader = PyMuPDFLoader(uploaded_file)
    # docs = loader.load()

    # 단계 1: 검색(Search) 도구 생성
    search = TavilySearchResults(max_results=5)

    # 단계 2: 텍스트 분할(Text Splitting)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # 단계 5: 도구 생성
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="Resume에 대한 질문이 나오면 이 도구를 사용해야해",
    )

    # tools 정의
    tools = [retriever_tool, search]


    # examples = [
    #     {
    #         "question": "메타 인터뷰 질문 만들어줘",
    #         "answer": """At Meta, we work extensively on detecting harmful content across multiple formats and 
    #         languages. In your research on identifying hidden meanings and hate speech in memes, how did you approach the 
    #         integration of language and visual models to uncover subtle cues of hateful intent? What potential would you 
    #         see in applying this research to content moderation on social platforms?""",
    #     },
    #     {
    #         "question": "메타 인터뷰 질문 만들어줘",
    #         "answer": """Meta leverages large language models to enhance user experiences in virtual assistance and 
    #         information retrieval. In your project using RAG and the Llama model, what strategies did you use to manage 
    #         the accuracy and relevance of real-time data in trip planning? How might your experience with LangChain and 
    #         real-time data integration contribute to building intelligent assistants for global users?""",
    #     },
    #     {
    #         "question": "구글 인터뷰 질문 만들어줘",
    #         "answer": """Google often deals with real-time data integration in applications like Maps and Search. 
    #         In your project using Retrieval-Augmented Generation with the Llama model, what techniques did you employ 
    #         to ensure data relevance and user-specific recommendations in trip planning? How might your skills with 
    #         LangChain and RAG apply to solving data retrieval and personalization challenges at Google’s scale?""",
    #     },
    # ]

    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            f"""You are an {interviewer} interviewer responsible for hiring for the role of {role} at {company}. Read the candidate’s resume, 
            using the pdf_search tool to gather information about the candidate, and the search tool to obtain information about {company}. 
            Based on these two sources, create interview questions for the candidate.

            if user answer your question, Read the candidate’s resume, using the pdf_search 
            tool to gather information about the candidate, and the search tool to obtain information about {company}. 
            Based on these two sources, evaluate the answer and give them feedback.
            if there is Emotion label, give additional feedback for the emotion of the candidate.
            
            Example1: 
            question: give me a single interview question.
            answer: At Meta, we work extensively on detecting harmful content across multiple formats and 
            languages. In your research on identifying hidden meanings and hate speech in memes, how did you approach the 
            integration of language and visual models to uncover subtle cues of hateful intent? What potential would you 
            see in applying this research to content moderation on social platforms?

            Example2:
            question: give me a single interview question.
            answer: Google often deals with real-time data integration in applications like Maps and Search. 
            In your project using Retrieval-Augmented Generation with the Llama model, what techniques did you employ 
            to ensure data relevance and user-specific recommendations in trip planning? How might your skills with 
            LangChain and RAG apply to solving data retrieval and personalization challenges at Google’s scale?

            Example3:
            question: In my project utilizing Retrieval-Augmented Generation (RAG) with the Llama model, 
            I focused on delivering personalized trip planning recommendations by combining user-specific data 
            with real-time retrieval from relevant sources. Through LangChain, I integrated online sources to access up-to-date 
            information such as local prices and activity options. This approach ensured users received tailored suggestions 
            that were not only accurate but also contextually relevant to the current data landscape.

            At Google’s scale, these skills can address challenges in data retrieval and personalization by leveraging LangChain’s 
            modularity for scalable retrieval pipelines and RAG’s generative abilities for context-aware suggestions. Given Google’s 
            need for high-volume, real-time updates in applications like Maps and Search, my experience developing an AI Trip 
            Planner provides insights into optimizing data flows and dynamically tuning model recommendations based on live data. 
            This background equips me to contribute effectively to Google’s data integration and personalization efforts by ensuring 
            that user interactions remain both relevant and timely.
            answer: Your response effectively highlights relevant skills in LangChain, RAG, and real-time data retrieval, 
            aligning well with Google’s focus on user personalization in large-scale applications like Maps and Search. To enhance it:

            1.	Add More on Scale and Optimization: Google handles vast amounts of data. Mentioning how you might adapt LangChain’s 
            retrieval architecture to optimize for Google-scale demands, such as using distributed systems or indexing techniques, would strengthen the answer.
            2.	Specificity in Personalization: Emphasize any specific methods in your RAG project for tailoring recommendations. 
            Adding details on query refinement or ranking methods you used (or could use) to personalize at scale would demonstrate depth.
            3.	Tie Research Experience to Real-Time Relevance: You have a strong research background in AI and NLP, as shown in your 
            resume. Connecting this to potential Google challenges, like handling diverse data types or ensuring low-latency responses, 
            would underscore your academic rigor and technical adaptability.

            These additions would make your answer more robust for Google’s expectations in handling complex, large-scale systems.
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    from langchain.agents import create_tool_calling_agent

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # tool calling agent 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    from langchain.agents import AgentExecutor

    # AgentExecutor 생성
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        max_execution_time=10,
        handle_parsing_errors=True,
    )

    return agent_executor