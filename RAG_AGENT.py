import ollama
import numpy as np

# ---------- Step 1: Knowledge Base ----------
documents = [
    "The Eiffel Tower is a famous landmark in Paris, France.",
    "Retrieval-Augmented Generation (RAG) enhances LLMs with external data.",
    "OpenChat is an open-source instruction-tuned model for conversational tasks.",
    "Paris is the capital of France and is known for its fashion and history.",
    "Mount Everest is the highest mountain on Earth."
]

# ---------- Step 2: Embed Documents ----------
EMBEDDING_MODEL = 'nomic-embed-text'
vector_database = []

def store_documents(docs):
    for doc_text in docs:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=doc_text)
        vector_database.append((doc_text, response['embedding']))
    print(f"[INFO] {len(vector_database)} documents embedded.")

store_documents(documents)

# ---------- Step 3: Cosine Similarity ----------
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    return dot / (np.linalg.norm(v1) * np.linalg.norm(v2))

def retrieve_chunks(query, top_n=2):
    query_embed = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)['embedding']
    scores = [(text, cosine_similarity(query_embed, emb)) for text, emb in vector_database]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scores[:top_n]]

# ---------- Step 4: Augment Prompt ----------
def augment_prompt(query, chunks):
    context = "\n\n".join(chunks)
    return f"""You are ARYA, a helpful AI assistant.

Context:
{context}

Question: {query}
Answer:"""

# ---------- Step 5: Generate Response ----------
LLM_MODEL = 'openchat'  # or 'llama3'

def generate_answer(prompt):
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response['response']

# ---------- Step 6: Agent Loop ----------
def run_agent():
    print("🤖 ARYA: Hello! I’m your AI assistant. Ask me anything (type 'exit' to quit).")
    while True:
        query = input("\n🧑 You: ")
        if query.lower() in ['exit', 'quit']:
            print("👋 ARYA: Goodbye!")
            break
        chunks = retrieve_chunks(query)
        prompt = augment_prompt(query, chunks)
        answer = generate_answer(prompt)
        print("\n🤖 ARYA:", answer)

if __name__ == '__main__':
    run_agent()
