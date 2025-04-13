__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import os
import tempfile
from streamlit_extras.buy_me_a_coffee import button
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#load_dotenv()
st.title("PDF QA")
st.write("---")

openai_key = st.text_input("OPEN_AI_API_KEY", type="password")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages
    
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20
    )
    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    db = Chroma.from_documents(texts, embeddings)

    from langchain.callbacks.base import BaseCallbackHandler
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)
    
    st.header("PDF 에게 질문 해보세요")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner("질문을 하는 중입니다..."):
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)
            llm = ChatOpenAI(temperature=0.5, model="gpt-4", openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(),
            )
            qa_chain({"query": question})

button(username="inzin823", floating=True, width=220)



    




