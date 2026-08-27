# doc = {
#     "title": "Sample Document",
#     "content": "This is a sample document for testing purposes.",
# }

# doc["chunk_length"] = len(doc["content"]) 

# print(doc)

chunks = [
    {"title": "Sample Document 1", "content": "This is the first chunk of the document."},
    {"title": "Sample Document 2", "content": "This is the second chunk of the document."},
    {"title": "Sample Document 3", "content": "This is the third chunk of the document."}
]

for doc in chunks:
    doc["chunk_length"] = len(doc["content"])
    print(doc)

with open("./article/demo.txt", "r", encoding="utf-8") as f:
    text = f.read()


# Function that takes a long string and splits it every 500 characters
def chunk_text(text, chunk_size=500):
    chunks_list = []
    for i in range(0, len(text), chunk_size):
        # Take a slice of string from index 'i' to 'i + chunk_size'
        chunk = text[i : i + chunk_size]
        chunks_list.append(chunk)
    return chunks_list

# Test your chunker on the loaded file
file_chunks = chunk_text(text, chunk_size=500)

print(f"Total chunks created from file: {len(file_chunks)}")
print("First chunk preview:")
print(file_chunks[0])

raw_chunks = chunk_text(text, chunk_size=500)

# 2. Transform raw string chunks into structured dictionaries
documents = []

for index, chunk in enumerate(raw_chunks):
    doc_dict = {
        "id": f"chunk_{index}",
        "text": chunk,
        "source": "demo.txt",
        "length": len(chunk)
    }
    documents.append(doc_dict)

# 3. Verify your structured data
print(f"Total structured documents: {len(documents)}")
print("\nFirst document structure:")
print(documents[0])