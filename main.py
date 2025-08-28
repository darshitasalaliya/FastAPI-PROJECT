from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Chatbot Model
from langchain_google_genai import ChatGoogleGenerativeAI

# LLM 
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

app = FastAPI()

# Request schema for chatbot
class QueryRequest(BaseModel):
    query: str

# Setup Gemini Chatbot LLM
chatbot = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # or "gemini-1.5-pro"
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful AI assistant. Answer the question clearly:\n{question}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly travel guide."),
    ("human", "Suggest 3 must-visit places in {city}.")
])

@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        chain = prompt | chatbot
        response = chain.invoke({"question": request.query})
        return {"answer": response.content}
    except Exception as e:
        return {"error": str(e)}


# Request schema for llm
class TextRequest(BaseModel):
    text: str

# Initialize LLM (not chat)
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

@app.post("/summarize")
def summarize_text(request: TextRequest):
    prompt = f"Summarize the following text in 3-4 sentences:\n\n{request.text}"
    response = llm.invoke(prompt)
    return {"summary": response}