from pypdf import PdfReader
from ollama import embed
import faiss
import numpy as np
import json
from pathlib import Path

full_text = ""
documents_dir = Path("documents")
for pdf_file in documents_dir.glob("*.pdf"):
    book = PdfReader("documents/2308.04079v1.pdf")
    for page in book.pages:
        full_text += page.extract_text()

#fixed chunking implementing first
chunks = []

words = full_text.split()

chunk_size = 250
overlap = 50
for start in range(0,len(words),chunk_size-overlap):
    end = start + chunk_size
    chunks.append(words[start:end])

embed_string = []
count= 0
for chunk in chunks:
   itr =  " ".join(chunk)
   embed_string.append(itr)
embeddings = []
for e in embed_string:
    response = embed(
    model="nomic-embed-text",
    input=e
)
    embeddings.append(response["embeddings"][0])
embedded_array = np.array(embeddings, dtype = np.float32)
dimension = embedded_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedded_array)
filename = "data/chunks.json"
with open(filename,'w') as f:
    json.dump(embed_string,f,indent = 4)
faiss.write_index(index,"data/faiss.index")

    
    

    
