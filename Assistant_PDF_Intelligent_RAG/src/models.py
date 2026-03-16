from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

model= init_chat_model('gpt-4.1-mini', temperature=0.7)