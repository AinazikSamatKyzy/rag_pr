from neo4j import GraphDatabase

# Connection Setup
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password123")
driver = GraphDatabase.driver(URI, auth=AUTH)

# Sample chunk and extracted triplets (Output from FAISS step + Ollama extraction)
sample_data = {
    "chunk_id": "chunk_11",
    "text": "Pando is scaling operations in North America, Europe, and India with marquee customer wins.",
    "triplets": [
        {
            "subject": "Pando",
            "subject_type": "Company",
            "predicate": "OPERATES_IN",
            "object": "North America",
            "object_type": "Region",
        },
        {
            "subject": "Pando",
            "subject_type": "Company",
            "predicate": "OPERATES_IN",
            "object": "Europe",
            "object_type": "Region",
        },
        {
            "subject": "Pando",
            "subject_type": "Company",
            "predicate": "OPERATES_IN",
            "object": "India",
            "object_type": "Region",
        },
    ],
}


def ingest_chunk_and_triplets(tx, data):
    # 1. Create or match the raw text Chunk node
    chunk_query = """
    MERGE (c:Chunk {id: $chunk_id})
    SET c.text = $text
    """
    tx.run(chunk_query, chunk_id=data["chunk_id"], text=data["text"])

    # 2. Iterate through triplets to create entities, relationships, and source links
    for triplet in data["triplets"]:
        # Dynamic Cypher query using labels and parameterized values
        triplet_query = f"""
        // Find the source text chunk
        MATCH (c:Chunk {{id: $chunk_id}})

        // Upsert Subject Node
        MERGE (s:{triplet['subject_type']} {{name: $subject}})

        // Upsert Object Node
        MERGE (o:{triplet['object_type']} {{name: $object}})

        // Create entity-to-entity relationship
        MERGE (s)-[:{triplet['predicate']}]->(o)

        // Step 3: Link Chunk node to the Subject entity for Provenance
        MERGE (c)-[:MENTIONS]->(s)
        """

        tx.run(
            triplet_query,
            chunk_id=data["chunk_id"],
            subject=triplet["subject"],
            object=triplet["object"],
        )


# Execute Ingestion Transaction
with driver.session() as session:
    session.execute_write(ingest_chunk_and_triplets, sample_data)
    print("Chunk and entity triplets successfully ingested into Neo4j!")

driver.close()