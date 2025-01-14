from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

def init_retriever() : 
    persist_db = Chroma(
        persist_directory="./pdfminerDB",
        embedding_function=UpstageEmbeddings(model="embedding-passage"),
        collection_name="my_db",
    )
    retriever = persist_db.as_retriever()
    return retriever