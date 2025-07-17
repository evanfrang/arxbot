from arxbot.retriever import ArxbotRetriever

def main():
    retriever = ArxbotRetriever()
    while True:
        q = input("\nEnter your query (or 'exit' to quit): ")
        if q.lower() == "exit":
            break
        res = retriever.query(q)
        for i, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ):
            print(f"\nResult {i+1} (distance: {dist:.4f}):")
            print(f"Title: {meta.get('title', 'N/A')}")
            print(f"Abstract: {doc}")
            print(f"Date: {meta.get('published')}")

if __name__ == "__main__":
    main()