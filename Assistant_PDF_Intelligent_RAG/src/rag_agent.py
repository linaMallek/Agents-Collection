from langchain.agents import create_agent
from src.models import model
from src.tool import build_similarity_search_tool
from langgraph.checkpoint.memory import InMemorySaver  

def build_agent(vector_store):
    similarity_search = build_similarity_search_tool(vector_store)
    return create_agent(
        model=model,
        tools=[similarity_search],
        system_prompt='You are a helpful assistant for question answering over the uploaded document. '
        'Always use the similarity_search tool first when the user asks about the document, then answer using the retrieved context.',
        checkpointer=InMemorySaver(),
    )