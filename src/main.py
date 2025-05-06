import os
from utils.chunker import Chunker
from utils.embedding import load_pdf_text
from utils.vector_store import create_vector_store, build_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_qa_chain(pdf_path="Owners_Manual.pdf"):
    store_path = "./vector_store/faiss_store"
    index_file = os.path.join(store_path, "index.faiss")
    metadata_file = os.path.join(store_path, "index.pkl")

    if not os.path.exists(index_file) and not os.path.exists(metadata_file):
        print("Loading PDF...")
        all_text = load_pdf_text(pdf_path)

        chunker = Chunker()
        chunks = chunker.recursive(all_text)
        print(f"Chunked into {len(chunks)} pieces")

        print("Creating vector store...")
        vectorstore = create_vector_store(chunks)
        vectorstore.save_local(store_path)
    else:
        print("Using existing vector store...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(store_path, embedding_model, allow_dangerous_deserialization=True)

    print("Setting up chatbot...")
    return build_qa_chain(vectorstore)

# === Optional CLI interface ===
if __name__ == "__main__":
    qa_chain = get_qa_chain()

    print("\nTesla Manual Chatbot is ready. Type your question (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.invoke(query)
        print(f"Tesla Assistant: {answer['result']}")
