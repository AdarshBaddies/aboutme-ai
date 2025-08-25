from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils import Retriever
from pathlib import Path


SYSTEM = (
"You are Adarsh's AI twin. Answer ONLY using the provided context. "
"If the answer is not in the context, say you don't know. "
"Be concise (3–6 sentences) and friendly."
)


DATA_PATH = str(Path(__file__).parent / "data" / "profile.txt")
retriever = Retriever(DATA_PATH)


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


app = FastAPI(title="Free RAG Chatbot")


class ChatIn(BaseModel):
question: str
k: int | None = 6
max_new_tokens: int | None = 220


class ChatOut(BaseModel):
answer: str
context: list[str]


@app.get("/health")
def health():
return {"status": "ok"}


@app.post("/chat", response_model=ChatOut)
@torch.inference_mode()
def chat(payload: ChatIn):
k = payload.k or 6
context_lines = retriever.fetch(payload.question, k=k)
context = "\n".join(context_lines)
prompt = f"{SYSTEM}\n\nContext:\n{context}\n\nQuestion: {payload.question}\nAnswer:"


inputs = tokenizer(prompt, return_tensors="pt").to(device)
out = model.generate(
**inputs,
max_new_tokens=payload.max_new_tokens or 220,
do_sample=True,
top_p=0.9,
temperature=0.7,
)
answer = tokenizer.decode(out[0], skip_special_tokens=True)
return ChatOut(answer=answer, context=context_lines)
