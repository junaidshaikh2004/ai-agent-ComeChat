from src.rag.embedder import get_embedder
from src.rag.vectorstore import load_vectorstore

CONFLICTING_DOCUMENT_PAIRS = [
    ("CARE-2026-01", "PROD-BREEZE-20"),
]

# KNOWN LIMITATION (for the bug diary): conflict detection is document-ID-based
# on the retrieved top-k only. A paraphrased query could theoretically rank one
# half of a conflicting pair outside k and miss the flag. Not fixed, because a
# distance-threshold fix we prototyped (force-fetch the missing partner, only
# count it if its relevance score was at least as good as the worst chunk
# already shown) introduced false positives on unrelated queries (e.g. "how do
# I clean my backpack" incorrectly flagged a tumbler conflict) -- a worse
# failure mode than an occasional missed flag.


def check_conflict(chunks):
    document_ids = {chunk.metadata.get("document_id") for chunk in chunks}
    for doc_a, doc_b in CONFLICTING_DOCUMENT_PAIRS:
        if doc_a in document_ids and doc_b in document_ids:
            return (doc_a, doc_b)
    return None


def retrieve(query, k=5):
    embedder = get_embedder()
    vectorstore = load_vectorstore()

    query_vector = embedder.embed_query(query)

    scored_chunks = vectorstore.similarity_search_by_vector_with_relevance_scores(
        embedding=query_vector,
        k=k,
        filter={"$and": [{"status": "active"}, {"policy_authority": "official"}]},
    )

    conflict_pair = check_conflict([chunk for chunk, _ in scored_chunks])

    return {
        "chunks": scored_chunks,
        "conflict": conflict_pair is not None,
        "conflict_pair": conflict_pair,
    }


if __name__ == "__main__":
    result = retrieve("Can I put the Breeze Tumbler in the dishwasher?", k=3)
    for chunk, score in result["chunks"]:
        print(chunk.page_content[:100])
        print(chunk.metadata, "score:", score)
        print("---")
    print("conflict:", result["conflict"], "| pair:", result["conflict_pair"])
