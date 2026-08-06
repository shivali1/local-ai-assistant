from pypdf import PdfReader
from ollama import embed
import faiss
import numpy as np
import json
from pathlib import Path

all_words = []

# Load existing chunks
chunk_file = Path("data/chunks.json")
if chunk_file.exists():
    with open(chunk_file, "r") as f:
        chunks = json.load(f)
else:
    chunks = []

new_chunks = []

chunk_size = 250
overlap = 50

documents_dir = Path("documents")
indexed_file_path = Path("data/indexed_files.json")

if indexed_file_path.exists():
    with open(indexed_file_path, "r") as f:
        indexed_files = json.load(f)
else:
    indexed_files = []

new_files = []

for pdf_file in documents_dir.glob("*.pdf"):

    if pdf_file.name in indexed_files:
        print(f"Skipping {pdf_file.name}")
        continue

    print(f"Indexing {pdf_file.name}")
    new_files.append(pdf_file.name)

    all_words = []

    book = PdfReader(pdf_file)

    for page_number, page in enumerate(book.pages, start=1):

        text = page.extract_text()

        if text:

            words = text.split()

            for word in words:

                all_words.append({
                    "word": word,
                    "file": pdf_file.name,
                    "page": page_number
                })

    for start in range(0, len(all_words), chunk_size - overlap):

        end = start + chunk_size

        chunk_words = all_words[start:end]

        if not chunk_words:
            continue

        files = {word["file"] for word in chunk_words}
        assert len(files) == 1

        text = " ".join(word["word"] for word in chunk_words)

        file_name = chunk_words[0]["file"]
        page_start = chunk_words[0]["page"]
        page_end = chunk_words[-1]["page"]

        new_chunks.append({
            "text": text,
            "file": file_name,
            "page_start": page_start,
            "page_end": page_end
        })

if len(new_chunks) == 0:
    print("No new PDFs found.")
    exit()

embeddings = []

for chunk in new_chunks:

    response = embed(
        model="nomic-embed-text",
        input=chunk["text"]
    )

    embeddings.append(response["embeddings"][0])

embedded_array = np.array(embeddings, dtype=np.float32)

faiss.normalize_L2(embedded_array)
dimension = embedded_array.shape[1]
index_path = Path("data/faiss.index")

if index_path.exists():
    index = faiss.read_index(str(index_path))
else:
    index = faiss.IndexFlatIP(dimension)

index.add(embedded_array)
chunks.extend(new_chunks)

with open(chunk_file, "w") as f:
    json.dump(chunks, f, indent=4)

faiss.write_index(index, str(index_path))

indexed_files.extend(new_files)

with open(indexed_file_path, "w") as f:
    json.dump(indexed_files, f, indent=4)

print(f"\nIndexed {len(new_files)} new PDF(s)")
print(f"Added {len(new_chunks)} new chunks")