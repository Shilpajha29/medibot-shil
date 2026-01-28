import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"

# pdf loader
def load_pdfs():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())
    return documents

# text splitting
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)



# creating embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
emb = get_embeddings()
print(len(emb.embed_query("hello world")))

# faiss vector store
VECTOR_PATH = "vectorstore"

def create_vectorstore(chunks, embeddings):
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_PATH)

if __name__ == "__main__":
    docs = load_pdfs()
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    create_vectorstore(chunks, embeddings)
    print("Vector store created")
