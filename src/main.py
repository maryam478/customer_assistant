import os
from utils.chunker import Chunker
from utils.embedding import load_pdf_text
from utils.vector_store import create_vector_store, build_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_qa_chain(pdf_folder='data/'):
    store_path = "./vector_store/faiss_store"
    index_file = os.path.join(store_path, "index.faiss")
    metadata_file = os.path.join(store_path, "index.pkl")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        print("Loading PDFs...")
        all_text = ""

        # ✅ Load all PDFs from the folder
        for file_name in os.listdir(pdf_folder):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, file_name)
                all_text += load_pdf_text(pdf_path) + "\n"

        chunker = Chunker()
        chunks = chunker.recursive(all_text)
        print(f"Chunked into {len(chunks)} pieces")

        print("Creating vector store...")
        vectorstore, docs = create_vector_store(chunks)
        vectorstore.save_local(store_path)
        return build_qa_chain(vectorstore, bm25_docs=docs, retrieval_type="hybrid")

    else:
        print("Using existing vector store...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(store_path, embedding_model, allow_dangerous_deserialization=True)
        return build_qa_chain(vectorstore, retrieval_type="dense")  # fallback if docs aren't available

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
