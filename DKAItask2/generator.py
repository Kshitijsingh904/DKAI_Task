import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("groq_api"),
    temperature=0.3
)

def generate_answer(context_text, query):
    prompt = f"""You are a precise assistant. Use ONLY the following context to answer the question.
If the answer isn't in the context, say "I cannot find that in the database."

Context:
{context_text}

Question:
{query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    sample_context = "Retrieval-Augmented Generation (RAG) combines retrieval and generation."
    sample_query = "What is RAG?"
    answer = generate_answer(sample_context, sample_query)
    print(answer)