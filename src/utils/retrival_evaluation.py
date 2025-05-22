import mlflow
from langchain_community.vectorstores import FAISS
import json
from langchain_huggingface import HuggingFaceEmbeddings
def evaluate_retrival():
    vector_store_paths = {
        "fixed_chunk": "./src/vector_store/fixed_size",
        "nltk_chunk": "./src/vector_store/nltk_sentence_chunks",
        # "recursive_chunk_store": "./src/vector_store/recursive"
    }

    ground_truth_path = {
        "fixed_chunk": "./src/resources/Ground_truth/fixed_chunk_ground_truth.json",
        "nltk_chunk": "./src/resources/Ground_truth/nltk_chunk_gound_truth.json",
        # "recursive_chunk_store": "./src/vector_store/recursive"
    }


    for key, value in vector_store_paths.items():
        path = ground_truth_path[key]
        with open(path, "r") as f:
            gt_data = json.load(f)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(value, embeddings= embedding_model,allow_dangerous_deserialization=True)

        data = []
        for i in gt_data['queries']:

            queries_data = {}

            question = i['query_text']
            q_chunk = i['relevant_chunks']

            retrive_chunk = vector_store.similarity_search(question,k = 5)

            chunks = [ res.page_content for res in retrive_chunk ]

            queries_data['question'] = question
            queries_data['retrieved_context'] = chunks
            queries_data['ground_truth_context'] = q_chunk

            data.append(queries_data)

        with open(f"src/resources/gt_with_retrival_docs/{key}_gt_rt.json",mode='w',encoding="utf-8") as file:
            json.dump(data,file, ensure_ascii=False, indent=2)
evaluate_retrival()
