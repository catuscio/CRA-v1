from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

def context_retriever() : 
    persist_db = Chroma(
        persist_directory="./pdfminerDB",
        embedding_function=UpstageEmbeddings(model="embedding-passage"),
        collection_name="my_db",
    )
    retriever = persist_db.as_retriever()
    return retriever

def search_context(user_input) :
    persist_db = Chroma(
        persist_directory="./pdfminerDB",
        embedding_function=UpstageEmbeddings(model="embedding-passage"),
        collection_name="my_db",
    )
    return persist_db.similarity_search(user_input)