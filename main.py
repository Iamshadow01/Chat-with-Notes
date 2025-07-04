import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os

css = """
<style>
    .stApp {
        background-color: #15171A;
        color: #fff;
    }
    .st-emotion-cache-1v0mbdj {
        background-color: #23272F !important;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        background-color: #23272F;
        color: #fff;
    }
    .stFileUploader {
        background-color: #23272F;
        border-radius: 8px;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vector_store(text_chunks):
    watsonx_api_key = os.getenv("WATSONX_APIKEY")
    watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
    watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    embeddings = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-107m-multilingual",
        url=watsonx_url,
        project_id=watsonx_project_id,
        apikey=watsonx_api_key,
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vector_store):
    watsonx_api_key = os.getenv("WATSONX_APIKEY")
    watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
    watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    llm = WatsonxLLM(
        model_id="google/flan-t5-xl",
        url=watsonx_url,
        project_id=watsonx_project_id,
        apikey=watsonx_api_key,
        params={"decoding_method": "sample", "max_new_tokens": 256, "min_new_tokens": 1}
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a standalone question based on the chat history and current question"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        vector_store.as_retriever(), 
        contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using ONLY the following context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def main():
    load_dotenv()
    st.set_page_config(
        page_title="LearnMate - PDF Question Answering Bot",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### Upload and Ask")
        pdf_docs = st.file_uploader(
            "Upload a PDF",
            type="pdf",
            accept_multiple_files=True,
            help="Drag and drop file here\nLimit 200MB per file · PDF"
        )
        if pdf_docs:
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name} ({round(pdf.size/1e6, 2)}MB)")
        st.markdown("---")
        st.caption("Powered by LangChain and IBM watsonx.ai")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chunks_info" not in st.session_state:
        st.session_state.chunks_info = None
    if "last_context" not in st.session_state:
        st.session_state.last_context = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None

    # Main Title
    st.markdown(
        "<h1 style='color:#fff; font-weight:700'>📘 LearnMate - PDF Question Answering Bot</h1>",
        unsafe_allow_html=True
    )

    # Process PDFs and chunk info
    if pdf_docs and st.session_state.conversation is None:
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversation_chain(vector_store)
            st.session_state.chunks_info = len(text_chunks)
        st.success(f"PDF loaded and split into {st.session_state.chunks_info} chunks.", icon="✅")

    elif st.session_state.conversation and st.session_state.chunks_info:
        st.success(f"PDF loaded and split into {st.session_state.chunks_info} chunks.", icon="✅")

    # Chat interface
    user_question = st.chat_input("Type your question:")

    # Show chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Handle new question
    if user_question and st.session_state.conversation:
        # Add user question to history
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        with st.spinner("Getting answer..."):
            response = st.session_state.conversation.invoke({
                "input": user_question,
                "chat_history": st.session_state.chat_history
            })
        
        st.session_state.last_context = "Most relevant context (click to view)\n\n" \
        st.session_state.confidence = 81.28  # Example value

        # Add AI response to history
        st.session_state.chat_history.append(AIMessage(content=response["answer"]))
        

    # Show match confidence and context if available
    if st.session_state.confidence is not None and st.session_state.last_context:
        st.markdown(
            f"<span style='color:#b48ead;'>Match confidence: {st.session_state.confidence:.2f}%</span>",
            unsafe_allow_html=True
        )
        with st.expander("Most relevant context (click to view)"):
            st.code(st.session_state.last_context, language="markdown")
        st.success("**Answer:**", icon="✅")
        # Show the last AI answer
        if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], AIMessage):
            st.markdown(f"{st.session_state.chat_history[-1].content}")

if __name__ == "__main__":
    main()
