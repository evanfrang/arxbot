from arxbot.rag_ollama import rag_query

context = [
    "In this paper, we study Hawking radiation in Jackiw-Teitelboim gravity for minimally coupled massless and massive scalar fields.",
    "We employ a holography-inspired technique to derive the Bogoliubov coefficients. We consider both black holes in equilibrium and black holes attached to a bath.",
]

query = "What other systems has Hawking Radiation been studied?"

response = rag_query(query, context)
print("Ollama says:\n", response)