from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pathlib import Path


class Retriever:
def __init__(self, data_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
self.model = SentenceTransformer(model_name)
self.sentences = self._load_lines(data_path)
self.index, self.vecs = self._build_index(self.sentences)


def _load_lines(self, data_path: str):
text = Path(data_path).read_text(encoding="utf-8")
# Keep non-empty, strip whitespace
return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _build_index(self, sentences):
vecs = self.model.encode(sentences, normalize_embeddings=True)
vecs = np.asarray(vecs, dtype="float32")
dim = vecs.shape[1]
index = faiss.IndexFlatIP(dim) # cosine because normalized
index.add(vecs)
return index, vecs


def fetch(self, query: str, k: int = 6):
q = self.model.encode([query], normalize_embeddings=True)
q = np.asarray(q, dtype="float32")
scores, idxs = self.index.search(q, k)
idxs = idxs[0].tolist()
return [self.sentences[i] for i in idxs if i != -1]
