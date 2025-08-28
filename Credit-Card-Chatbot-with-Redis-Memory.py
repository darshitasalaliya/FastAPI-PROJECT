from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Load environment variables
load_dotenv()

app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str
    session_id: str   # track each user's conversation

# Setup Gemini Chatbot LLM
chatbot = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3  # stable responses
)

# System role for Credit Card Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a professional customer support agent for a credit card company. "
     "Answer questions about billing, payments, limits, rewards, and fraud prevention. "
     "Always be polite and concise, and never reveal sensitive account details."),
    ("human", "{question}")
])


# Create Redis-based memory for each session
def get_chain(session_id: str):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Use Redis to persist chat history per session_id
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=redis_url
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history,
        input_key="question"
    )

    chain = LLMChain(
        llm=chatbot,
        prompt=prompt,
        memory=memory
    )
    return chain


@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        chain = get_chain(request.session_id)
        response = chain.invoke({"question": request.query})
        return {"answer": response["text"]}
    except Exception as e:
        return {"error": str(e)}
