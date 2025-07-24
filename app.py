from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MODEL_PATH = "models/quest_generator"
GROQ_MODEL = "llama3-70b-8192" if os.environ.get("GROQ_API_KEY") else None

# Load model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    local_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    print("✅ Fine-tuned model loaded")
except:
    print("⚠️ Fine-tuned model not found, using base model")
    local_pipeline = None

# Groq setup if available
groq_chain = None
if GROQ_MODEL:
    try:
        prompt_template = ChatPromptTemplate.from_template(
            "Generate fantasy quest based on: {prompt}. "
            "Include mystical elements, challenges, and rewards. "
            "Use medieval fantasy language."
        )
        groq_chain = prompt_template | ChatGroq(temperature=0.7, model_name=GROQ_MODEL)
        print(f"✅ Groq connected ({GROQ_MODEL})")
    except:
        print("⚠️ Groq initialization failed")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 200

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_quest(request: GenerateRequest):
    start_time = time.time()
    
    # Try Groq first if available
    if groq_chain:
        try:
            result = await groq_chain.ainvoke({"prompt": request.prompt})
            return {
                "quest": result.content,
                "model": GROQ_MODEL,
                "time": f"{(time.time() - start_time):.2f}s"
            }
        except Exception as e:
            print(f"Groq error: {e}")
    
    # Fallback to local model
    if local_pipeline:
        try:
            result = local_pipeline(
                request.prompt,
                max_length=request.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8
            )
            return {
                "quest": result[0]['generated_text'],
                "model": "local/fine-tuned",
                "time": f"{(time.time() - start_time):.2f}s"
            }
        except Exception as e:
            print(f"Model error: {e}")
    
    raise HTTPException(status_code=500, detail="No available generation methods")

@app.get("/status")
async def status():
    return {
        "status": "operational",
        "models": {
            "local": "loaded" if local_pipeline else "unavailable",
            "groq": GROQ_MODEL or "unavailable"
        },
        "version": "1.0",
        "uptime": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089)