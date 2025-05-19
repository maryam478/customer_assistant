# from langchain.chains import RetrievalQA
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_openai import ChatOpenAI
# from langchain.docstore.document import Document
# from utils.prompt import custom_prompt
# import os
# from dotenv import load_dotenv

# load_dotenv()
# API_KEY = os.getenv("OPENAI_API_KEY")


# def create_vector_store(chunks):
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     docs = [Document(page_content=chunk) for chunk in chunks]
#     vector_store = FAISS.from_documents(docs, embedding_model)
#     return vector_store
    

# def build_qa_chain(vectorstore):
#     retriever = vectorstore.as_retriever()
#     return RetrievalQA.from_chain_type(
#         llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     openai_api_key=API_KEY
#     ),
#         retriever=retriever,
#         chain_type="stuff",  # you can also experiment with 'map_reduce'
#         chain_type_kwargs={"prompt": custom_prompt},
#         return_source_documents=False
#     )
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever
from utils.prompt import custom_prompt

import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store, docs  # returning docs for BM25/hybrid


def build_qa_chain(vectorstore, retrieval_type="hybrid", bm25_docs=docs):
    """
    Builds a QA chain using different retriever types:
    - "dense": default vector similarity search
    - "mmr": Max Marginal Relevance
    - "bm25": keyword-based sparse retriever
    - "hybrid": combines dense and BM25 results
    """

    if retrieval_type == "dense":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    elif retrieval_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "lambda_mult": 0.5}
        )

    elif retrieval_type == "bm25":
        if bm25_docs is None:
            raise ValueError("BM25 retriever requires bm25_docs.")
        retriever = BM25Retriever.from_documents(bm25_docs)
        retriever.k = 4

    elif retrieval_type == "hybrid":
        if bm25_docs is None:
            raise ValueError("Hybrid retriever requires bm25_docs.")
        bm25 = BM25Retriever.from_documents(bm25_docs)
        bm25.k = 4
        dense = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Simple hybrid by combining results (no rank fusion)
        class HybridRetriever:
            def get_relevant_documents(self, query):
                bm25_results = bm25.get_relevant_documents(query)
                dense_results = dense.get_relevant_documents(query)
                combined = {doc.page_content: doc for doc in bm25_results + dense_results}
                return list(combined.values())

        retriever = HybridRetriever()

    else:
        raise ValueError("Invalid retrieval_type. Choose from: dense, mmr, bm25, hybrid")

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=API_KEY
        ),
        retriever=retriever,
        chain_type="stuff",  # or try "map_reduce"
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=False
    )
