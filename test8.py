__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import os
import streamlit as st
import pandas as pd
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


from langchain_chroma import Chroma

#오픈AI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

# CSV 불러오기 + category 추가 + Document로 변환
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

# 벡터 저장소 생성
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_directory
    )
    return vectorstore

# RAG 체인 초기화
def initialize_components(selected_model, docs):
    vectorstore = create_vector_store(docs)
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
대답은 한국어로 하고, 존댓말을 써줘.\n\n{context}"""),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.set_page_config(page_title="대전 여행 추천 챗봇")
st.header("📍 대전 맞춤 여행 코스 추천 챗봇")

# 모델 선택
option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))





# 로컬 폴더 경로 설정
FOLDER_PATH = "data"

if os.path.exists(FOLDER_PATH):
    with st.spinner("데이터를 불러오는 중입니다..."):
        docs, df = load_csvs_as_documents(FOLDER_PATH)
        rag_chain = initialize_components(option, docs)
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
                    response = conversational_rag_chain.invoke(
                        {"input": prompt},
                        config
                    )
                    st.write(response['answer'])
                    with st.expander("참고 문서 보기"):
                        for doc in response['context']:
                            st.markdown(doc.metadata.get('source', '알 수 없음'), help=doc.page_content)
else:
    st.error(f"❌ 폴더 경로를 찾을 수 없습니다:\n{FOLDER_PATH}")


# 
options = st.multiselect(
    "어떤 걸 중심으로 여행하고 싶나요?",
    ["숙박", "사진 찍기", "로컬 맛집", "쇼핑"]
)

# OpenAI 클라이언트
client = OpenAI()

if options:
    # CSV에서 선택된 옵션 필터링
    filtered_context = ""
    for opt in options:
        filtered_df = df[df['category'].str.contains(opt, na=False)]
        for _, row in filtered_df.iterrows():
            filtered_context += f"{row['name']} ({row['address']})\n"

    # GPT-MINI에게 요청
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
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # 결과 출력
    st.markdown(response.choices[0].message.content)
else:
    st.info("관심 있는 테마를 선택해주세요!")
