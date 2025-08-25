from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, re
import gradio as gr
from utils import Retriever
from pathlib import Path

# ---------------- SYSTEM + FEWSHOTS ----------------
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

# ---------------- SETUP ----------------
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

# ---------------- HELPER ----------------
def quick_reply(text: str) -> str | None:
    t = text.strip().lower()
    if re.fullmatch(r"(hi|hii+|hello|hey|yo|hola)[!. ]*", t):
        return "Hey! I'm Adarsh. What would you like to know about me?"
    return None

def generate_answer(question: str, k: int = 6, max_new_tokens: int = 220) -> tuple[str, list[str]]:
    qr = quick_reply(question)
    if qr:
        return qr, []

    # retrieve
    context_lines = retriever.fetch(question, k=k)
    context = "\n".join(context_lines)

    # prompt
    examples = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in FEW_SHOTS])
    prompt = (
        f"{SYSTEM}\n\n"
        f"{examples}\n\n"
        f"Context:\n{context}\n\n"
        f"Q: {question}\nA:"
    )

    # generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
    )
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    return answer, context_lines

# ---------------- FASTAPI ROUTE ----------------
@app.post("/chat", response_model=ChatOut)
@torch.inference_mode()
def chat(payload: ChatIn):
    ans, ctx = generate_answer(payload.question, payload.k, payload.max_new_tokens)
    return ChatOut(answer=ans, context=ctx)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- GRADIO UI ----------------
def gradio_fn(message: str):
    ans, _ = generate_answer(message)
    return ans

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Adarsh's AI Twin")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me anything about Adarsh...")
    clear = gr.Button("Clear")

    def respond(message, history):
        ans, _ = generate_answer(message)
        history = history + [(message, ans)]
        return history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch Gradio along with FastAPI
import threading
def launch_gradio():
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

threading.Thread(target=launch_gradio, daemon=True).start()
