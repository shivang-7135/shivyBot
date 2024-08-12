import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# Sidebar contents
with st.sidebar:
    st.title('💬 Shivy Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Hugging Face Transformers](https://huggingface.co/sentence-transformers) models
 
    ''')
    st.write('Made with ❤️ by [Shivang Sinha](https://www.linkedin.com/in/shivang-sinha-92755012b/)')
 
load_dotenv()
 
def main():
    st.header("Chat with PDF 💬")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
 
        if os.path.exists(f"{store_name}_index.pkl") and os.path.exists(f"{store_name}_embeddings.npy"):
            with open(f"{store_name}_index.pkl", "rb") as f:
                index = pickle.load(f)
            embeddings_array = np.load(f"{store_name}_embeddings.npy")
        else:
            # Use SentenceTransformer model for embedding
            
            embeddings = embedding_model.encode(chunks)
            embeddings_array = np.array(embeddings, dtype='float32')

            # Initialize FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            # Save the index and embeddings for future use
            with open(f"{store_name}_index.pkl", "wb") as f:
                pickle.dump(index, f)
            np.save(f"{store_name}_embeddings.npy", embeddings_array)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
 
        if query:
            query_embedding = embedding_model.encode([query])
            distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=3)
            matching_chunks = [chunks[i] for i in indices[0]]
            
            st.write("Here are the top results:")
            for chunk in matching_chunks:
                st.write(chunk)

if __name__ == '__main__':
    main()
