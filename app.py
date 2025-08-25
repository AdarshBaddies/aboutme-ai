from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from utils import Retriever
from pathlib import Path
import uvicorn
import gradio as gr
import re

SYSTEM = (
    "You are Adarsh's AI twin. Speak in first person as Adarsh ('I', 'my'). "
    "Tone: warm, concise, friendly, a bit playful. Use short sentences. "
    "When greeted (e.g., 'hi', 'hello', 'hey'), reply with a short friendly greeting first, then offer help. "
    "Answer ONLY using the provided context. If the answer is not in the context, say you don't know and offer to add it later. "
    "Keep answers to 1–3 sentences unless asked for detail."
)

FEW_SHOTS = [
    ("Hello", "Hey! I'm Adarsh. What would you like to know about me?"),
    ("Who are you?", "I'm Adarsh's AI twin. I answer using my profile context."),
]


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

def quick_reply(text: str) -> str | None:
    t = text.strip().lower()
    if re.fullmatch(r"(hi|hii+|hello|hey|yo|hola)[!. ]*", t):
        return "Hey! I'm Adarsh. What would you like to know about me?"
    return None

def generate_answer(question: str, k: int = 6, max_new_tokens: int = 220) -> tuple[str, list[str]]:
    qr = quick_reply(question)
    if qr:
        return qr, []


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


# ✅ Gradio frontend (calls the same backend logic directly)
def gradio_chat(question):
    payload = ChatIn(question=question)
    result = chat(payload)
    return result.answer, "\n".join(result.context)


demo = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(label="Ask Adarsh's AI twin"),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Context")],
    title="Adarsh's RAG Chatbot"
)


if __name__ == "__main__":
    # Run FastAPI + Gradio separately
    import threading

    def run_api():
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

    threading.Thread(target=run_api, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
