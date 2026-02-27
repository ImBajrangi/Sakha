from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.optimizer import QueryOptimizer
from core.chatbot import FastChatbot
import uvicorn
import config

app = FastAPI(title="Sakha Vrindopnishad AI API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

optimizer = QueryOptimizer()
chatbot = FastChatbot()

class SearchQuery(BaseModel):
    query: str

class OptimizationResponse(BaseModel):
    original: str
    optimized: str
    intent: str
    confidence: str
    seo_keywords: list[str]

class ChatQuery(BaseModel):
    query: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    response: str
    navigate: str | None = None

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_query(request: SearchQuery):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = optimizer.optimize_query(request.query)
    seo = optimizer.generate_seo_keywords(request.query)
    
    return {
        **result,
        "seo_keywords": seo
    }

@app.post("/ask", response_model=ChatResponse)
async def ask_assistant(request: ChatQuery):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = chatbot.get_response(request.query, history=request.history)
    return result

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Sakha Optimizer"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
