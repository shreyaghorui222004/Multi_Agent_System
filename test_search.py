from dotenv import load_dotenv
from tools import web_search

load_dotenv()

result = web_search.invoke({
    "query": "RAG techniques in 2026"
})

print(result)