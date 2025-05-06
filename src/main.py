'''main function, this function
this will receive a query as argument
then
1.we will import embedding function and embed the query
2. calling the 2nd function from vectorstor.py in utils to load the vector store
3. we will call similarity search function and pass embeddded query and vector tsore to do similarity search and we will recive
k resposes,
4. we write generation logic
5. import prompts


 '''
from utils.chunker import Chunker
import os
from utils.embedding import load_pdf_text
from utils.vector_store import create_vector_store
from utils.vector_store import build_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS




if __name__ == "__main__":
    store_path = "./vector_store/faiss_store"
    index_file = os.path.join(store_path, "index.faiss")
    metadata_file = os.path.join(store_path, "index.pkl")
    # === Step 1: Load Tesla manual PDF ===
    # pdf_path = "Owners_Manual.pdf"  # <- Update this path if needed
    if not os.path.exists(index_file) and not os.path.exists(metadata_file):
        pdf_path = "src/Owners_Manual.pdf"

        print("Loading PDF...")
        all_text = load_pdf_text(pdf_path)

        # === Step 2: Choose Chunking Strategy ===
        chunker = Chunker()
        chunks = chunker.recursive(all_text)  # or: .fixed_size(), .nltk_sentence_chunks(), .custom_delimiter()
        print(f"Chunked into {len(chunks)} pieces")

    # # === Step 3: Create Knowledge Base ===


    

    
        
        print("Creating vector store...")
        vectorstore = create_vector_store(chunks)
        vectorstore.save_local(store_path)
        print('done')
        print("Setting up chatbot...")
        qa_chain = build_qa_chain(vectorstore)
    

    # === Step 4: Chat Loop ===
    
    else:
        print('using existing vector store....')
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = FAISS.load_local(store_path, embedding_model,allow_dangerous_deserialization=True)
        print("Setting up chatbot...")
        qa_chain = build_qa_chain(vectorstore)
        

    print("\nTesla Manual Chatbot is ready. Type your question (type 'exit' to quit):")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.invoke(query)
        print(f"Tesla Assistant: {answer['result']}")
