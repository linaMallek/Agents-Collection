from langchain_core.tools import create_retriever_tool


def build_similarity_search_tool(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    return create_retriever_tool(
        retriever,
        name='similarity_search',
        description='Search the uploaded document for relevant passages before answering the user.'
    )