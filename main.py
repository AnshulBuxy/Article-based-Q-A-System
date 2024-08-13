from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import transformers
#from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
import streamlit as st
from langchain_community.vectorstores import FAISS
import pickle 
import nltk



from dotenv import load_dotenv
load_dotenv()

st.title("Q/A Question Answering System")
st.sidebar.title("Upload Your URL")
urlt=[]
url=st.sidebar.text_input(f"url ")
urlt.append(url)
process_url_clicked= st.sidebar.button("process_url")
file_path="faiss_store_hugging.pkl"
# loadi= UnstructuredURLLoader(urls=["https://www.indiatoday.in/business/story/adani-group-statement-allegation-hindenburg-report-sebi-2580520-2024-08-11"])
# data= loadi.load()
# r_split= RecursiveCharacterTextSplitter(
#     chunk_size=1100,
#     chunk_overlap=200,
#     separators=["\n\n","\n"," ",""]
#     )
# docs= r_split.split_documents(data)
# print(len(docs))
model = 'google/flan-t5-large'
model_kwargs = {'temperature': 0.4,'max_length': 300}
hub_llm = HuggingFaceHub(repo_id=model, model_kwargs=model_kwargs)
main_placefolder=st.empty()
if process_url_clicked:
    loader= UnstructuredURLLoader(urls=urlt)
    main_placefolder.text("Processing URL...")
    data= loader.load()
    r_split= RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n","\n"," ",""]
    )
    docs= r_split.split_documents(data)
    main_placefolder.text("Reading the data...")
    embedd = HuggingFaceEndpointEmbeddings()
    vector_store=FAISS.from_documents(docs,embedd)
    with open(file_path, "wb") as f:
        pickle.dump(vector_store,f)
    main_placefolder.text("Data saved successfully]...")

query=main_placefolder.text_input("Enter your question")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)
            chain= RetrievalQAWithSourcesChain.from_llm(llm=hub_llm,retriever=vector_store.as_retriever())
            result= chain({"question":query},return_only_outputs=True)
            answer = result.get('answer', '').strip()
            sources = result.get('sources', '').strip()
            st.header("Answer")
            if answer:
                st.write(result["answer"])
            else:
                st.write(result["sources"])
            
       

