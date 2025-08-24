# Free RAG Chatbot (FastAPI on Hugging Face Spaces)


This is a free, CPU-only personal chatbot that answers questions using your profile text. No fine-tuning.


## Local development
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
