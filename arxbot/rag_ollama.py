import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"

def format_prompt(query: str, context_docs: list[str]) -> str:
    context_str = "\n\n".join(context_docs)
    return (
        f"You are a helpful research assistant. Use the following documents to answer the question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

def query_ollama(prompt: str, model: str = DEFAULT_MODEL, stream: bool = False) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": stream}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.status_code}: {response.text}")
    
    if stream:
        raise NotImplementedError("Streaming not implemented yet.")
    else:
        return response.json()["response"]

def rag_query(query: str, context_docs: list[str], model: str = DEFAULT_MODEL) -> str:
    prompt = format_prompt(query, context_docs)
    return query_ollama(prompt, model=model)


def rag_with_bm25(query: str, bm25_results, model: str = DEFAULT_MODEL) -> str:
    context_docs = [
        r["metadata"]['title'] + " " + \
        r['metadata']["abstract"] \
        for r in bm25_results
    ]
    print("\n--- Retrieved Context ---")
    for i, doc in enumerate(context_docs):
        print(f"[{i+1}] {doc[:300]}...\n") 
    prompt = format_prompt(query, context_docs)
    return query_ollama(prompt, model=model)