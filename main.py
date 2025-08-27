from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load .env file where OPENAI_API_KEY is stored
load_dotenv()

app = FastAPI()

# Define request schema
class QueryRequest(BaseModel):
    query: str

# Setup LangChain LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",   # you can change to gpt-4
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a simple prompt template
prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question:\n{question}")

@app.post("/ask")
async def ask(request: QueryRequest):
    chain = prompt | llm
    response = chain.invoke({"question": request.query})
    return {"answer": response.content}
