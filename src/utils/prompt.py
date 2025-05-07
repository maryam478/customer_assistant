from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use only the information from the following manual excerpt to answer the question.
If the answer is not in the text, say: "I'm sorry, I couldn't find that information in the manual."

Manual Excerpt:
{context}

Question:
{question}

Answer:"""
)
