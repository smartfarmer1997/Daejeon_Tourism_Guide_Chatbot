import os
import streamlit as st
import pandas as pd
import shutil

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.documents import Document
from openai import OpenAI

#Chroma tenant 오류 방지 위한 코드
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysplite3')

from langchain_chroma import Chroma

from chromadb.config import Settings

client_settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)

vectorstore = Chroma.from_documents(
    split_docs,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory=persist_directory,
    client_settings=client_settings
)

# 오픈AI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="대전 여행 추천 챗봇")
st.header("📍 대전 맞춤 여행 코스 추천 챗봇")

# 📄 사이드바에서 PDF 업로드 받기
with st.sidebar:
    st.markdown("📄 PDF 파일 업로드")
    uploaded_pdf = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

# 📄 PDF 로딩 및 벡터 디렉토리 지정
if uploaded_pdf:
    temp_path = os.path.join("./data", uploaded_pdf.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_pdf.getvalue())
    pdf_loader = PyPDFLoader(temp_path)
    pdf_docs = pdf_loader.load()
    file_key = uploaded_pdf.name.replace(".pdf", "").replace(" ", "_")
    persist_directory = f"./chroma_db_{file_key}"
else:
    default_pdf_path = os.path.join("./data", "꿀잼도시대전가이드북_웹배포용.pdf")
    pdf_loader = PyPDFLoader(default_pdf_path)
    pdf_docs = pdf_loader.load()
    persist_directory = "./chroma_db_default"

# 📊 CSV 로드
def load_csvs_as_documents(folder_path):
    files = {
        "숙박": "대전광역시_문화관광(숙박정보)_전처리완료.csv",
        "로컬 맛집": "대전광역시_문화관광(모범음식점)_전처리완료.csv",
        "사진 찍기": "대전광역시_문화관광(관광지)_전처리완료.csv",
        "쇼핑": "대전광역시_문화관광(쇼핑)_전처리완료.csv"
    }
    df_list = []
    for category, filename in files.items():
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="cp949")
        df["category"] = category
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    text = merged_df.to_string(index=False)
    documents = [Document(page_content=text, metadata={"source": "통합 CSV"})]
    return documents, merged_df

FOLDER_PATH = "./data"
docs, df = load_csvs_as_documents(FOLDER_PATH)
all_docs = pdf_docs + docs

# 🔍 벡터스토어 생성
def create_vector_store(_docs, persist_directory):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(_docs)
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=persist_directory
    )
    return vectorstore

def get_vector_store(_docs, persist_directory):
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
        )
    else:
        return create_vector_store(_docs, persist_directory)

# 🔗 RAG 체인 초기화
def initialize_components(selected_model, docs, persist_directory):
    vectorstore = get_vector_store(docs, persist_directory)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. \
       Use the following pieces of retrieved context to answer the question. \
       If you don't know the answer, just say that you don't know. \
       Keep the answer perfect. please use imogi with the answer. \
       대답은 한국어로 하고, 존댓말을 써줘. \
       만약 사용자가 날씨나 활동 추천, 실내/실외 선택 등과 관련된 질문을 한다면, 업로드된 PDF 내용(예: 기상정보 문서, 가이드북 등)을 적극적으로 참고해서 답변해줘.\n\n{context}"""),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# 🧠 모델 선택 및 체인 실행
option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))

if os.path.exists(FOLDER_PATH):
    with st.spinner("데이터를 불러오는 중입니다..."):
        rag_chain = initialize_components(option, all_docs, persist_directory)
        chat_history = StreamlitChatMessageHistory(key="chat_messages")

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "무엇이든 물어보세요!"}]

        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input("질문을 입력하세요"):
            st.chat_message("human").write(prompt)
            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    config = {"configurable": {"session_id": "default"}}
                    response = conversational_rag_chain.invoke({"input": prompt}, config)
                    st.write(response["answer"])
                    with st.expander("참고 문서 보기"):
                        for doc in response["context"]:
                            st.markdown(doc.metadata.get("source", "알 수 없음"), help=doc.page_content)
else:
    st.error(f"❌ 폴더 경로를 찾을 수 없습니다: {FOLDER_PATH}")

# 🎯 선택 기반 코스 추천
options = st.multiselect("어떤 걸 중심으로 여행하고 싶나요?", ["숙박", "사진 찍기", "로컬 맛집", "쇼핑"])
client = OpenAI()

if options:
    filtered_context = ""
    for opt in options:
        filtered_df = df[df["category"].str.contains(opt, na=False)]
        for _, row in filtered_df.iterrows():
            filtered_context += f"{row['name']} ({row['address']})\n"

    prompt = f"""
    너는 대전 여행 전문가야.
    사용자가 {', '.join(options)} 테마로 여행을 하고 싶대.
    아래 대전 관광지 데이터 참고해서,
    추천 여행 코스와 일정 만들어줘.

    관광지 목록:
    {filtered_context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.markdown(response.choices[0].message.content)
else:
    st.info("관심 있는 테마를 선택해주세요!")
