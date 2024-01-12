from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
#from langchain.callbacks import get_openai_callback

def main():
  load_dotenv()
  st.set_page_config(page_title="Ask your PDF")
  st.header("Ask your PDF ")

  pdf = st.file_uploader("Uplaod your PDF", type="pdf")    #upload the file
  
  #extract the text from pdf
  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()


    #
    #slplit into chunk of text
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(text)

    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # show user input
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question:
      docs = knowledge_base.similarity_search(user_question)
      st.write(docs)

      llm = OpenAI()
      chain = load_qa_chain(llm, chain_type="stuff")
      #with get_openai_callback() as cb:
        #response = chain.run(input_documents=docs, question=user_question)
        #print(cb)

      #st.write(response)

if __name__ == '__main__':
  main()
