import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

css = """
<style>
    .stApp {
        background-color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
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
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #3498db;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #ecf0f1;
        margin-right: 20%;
    }
    .stSpinner>div>div {
        border-color: #3498db transparent transparent transparent;
    }
</style>
"""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
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
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    
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
        page_title="PDF Chat Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(css, unsafe_allow_html=True)
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.title("📚 PDF Chat Assistant")
        st.markdown("---")
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader(
            "Drag & drop PDF files here",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF files to analyze"
        )
        
        if st.button("Process Documents", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
            else:
                with st.spinner("Analyzing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)
                    st.success("Documents processed successfully!")
                    st.balloons()
        
        st.markdown("---")
        st.subheader("How to Use")
        st.markdown("""
        1. Upload PDF documents
        2. Click **Process Documents**
        3. Ask questions about the content in the chat
        """)
        
        st.markdown("---")
        st.caption("Powered by LangChain and OpenAI")

    # Main chat area
    st.header("Chat with your PDFs 📚")
    
    # Chat history display
    chat_container = st.container(height=500)
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if isinstance(message, HumanMessage):
                st.markdown(
                    f'<div class="chat-message user-message">👤 {message.content}</div>', 
                    unsafe_allow_html=True
                )
            elif isinstance(message, AIMessage):
                st.markdown(
                    f'<div class="chat-message bot-message">🤖 {message.content}</div>', 
                    unsafe_allow_html=True
                )
    
    # User input with clear button
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
    with col2:
        submit_btn = st.button("Ask", use_container_width=True)
    
    # Handle question submission
    if (submit_btn or user_question) and st.session_state.conversation:
        if not user_question.strip():
            st.warning("Please enter a question")
        else:
            # Add user question to history immediately
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            
            # Get and display response
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.invoke({
                    "input": user_question,
                    "chat_history": st.session_state.chat_history
                })
                
            # Add AI response to history
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))
            
            # Rerun to update chat display
            st.rerun()

if __name__ == "__main__":
    main()
