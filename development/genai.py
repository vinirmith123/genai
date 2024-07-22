import os
import streamlit as st
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Set up OpenAI Embeddings
embedding = OpenAIEmbeddings()

def process_pdfs(pdf_files):
    """Process uploaded PDF files into text chunks."""
    all_text = ""
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()

        # Concatenate text from all pages
        all_text += "\n".join([page.page_content for page in pages])

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(all_text)

    # Convert texts to Document objects
    documents = [Document(page_content=text) for text in texts]

    return documents

def create_vector_db(documents):
    """Create and persist a Chroma vector store."""
    persist_directory = 'db'

    vectordb = Chroma.from_documents(documents=documents, 
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    vectordb.persist()

    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

def get_retriever(vectordb):
    """Retrieve documents from Chroma vector store."""
    return vectordb.as_retriever(search_kwargs={"k": 5})

def process_llm_response(llm_response):
    """Format LLM response for output."""
    result = llm_response['result']
    sources = [source.metadata['source'] for source in llm_response["source_documents"]]
    return result, sources

def main():
    st.set_page_config(page_title="PDF Query Application", page_icon="📄")
    st.title("PDF Query Application")

    # File uploader
    pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if pdf_files:
        with st.spinner("Processing PDFs..."):
            documents = process_pdfs(pdf_files)
            vectordb = create_vector_db(documents)
            retriever = get_retriever(vectordb)

            # User input for query
            query = st.text_input("Ask a question about the PDFs")

            if query:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=OpenAI(), 
                    chain_type="stuff", 
                    retriever=retriever, 
                    return_source_documents=True
                )

                llm_response = qa_chain(query)
                result, sources = process_llm_response(llm_response)

                st.subheader("Answer:")
                st.write(result)

                st.subheader("Sources:")
                for source in sources:
                    st.write(source)

if __name__ == "_main_":
      main() 