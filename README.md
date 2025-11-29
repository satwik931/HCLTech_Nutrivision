# GenAI-Powered Telecom Ticket Analysis System

## Overview

This project builds an AI-based assistant for telecom support ticket resolution using a RAG (Retrieval-Augmented Generation) architecture. The system compares a new customer complaint with past resolved issues and generates the top 3 recommended solutions with suitability scores.

It improves:
- Resolution time
- Accuracy of troubleshooting
- Agent productivity
- Customer satisfaction

## Key Features

- Embedding-based similarity search
- Vector database for fast retrieval
- Retrieval-Augmented Generation
- Local LLM inference (Mistral)
- Top-3 recommended solutions with suitability percentage

## System Architecture (High-Level)

1. Generate embeddings for each ticket using an embedding model
2. Store embeddings in a vector database
3. Retrieve similar past tickets using nearest-neighbor search
4. Feed retrieved results into the RAG pipeline
5. LLM generates recommended solutions + suitability score
6. Display final recommendations to the user

## Dataset

Used a cleaned telecom ticket dataset consisting of:
- Customer issue description
- Troubleshooting steps
- Final solutions

### Preprocessing steps:
- Remove noise
- Normalize text
- Combine fields
- Clean formatting



## Step-by-Step Pipeline Explanation

### 1. Generate Embeddings

We converted each support ticket into dense vectors using:
```python
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
```

This helps the AI understand text meaning.

### 2. Build Vector Database

We stored vector embeddings in FAISS:
```python
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings)
```

FAISS allows lightning-fast similarity search.

### 3. Retrieve Similar Past Tickets

When the user enters a new query:
```python
search_faiss(query, k=3)
```

We fetch top-K closest past tickets based on embeddings.

### 4. Apply RAG (Retrieve-Augment-Generate)

We passed retrieved solutions + tickets into an LLM prompt:
```python
context += f"Past Ticket...\nResolved Solution..."
```

This ensures the model uses real telecom knowledge.

### 5. LLM Generates Final Solutions

We used a local LLM:
```python
llm = Llama(model_path="mistral.gguf")
```

The LLM returns:
- 3 best solutions
- Suitability percentage

Example format:
```
1. Restart router (92%)
2. Check WAN settings (85%)
3. Replace power adapter (78%)
```

### 6. Display Output to User

Final results are printed:
- Similar past tickets
- Recommended solutions
- Suitability scores

**Example Query:**
```
"Internet disconnects frequently and router keeps restarting"
```

**Example Result:**
```
1. Update router firmware (91%)
2. Replace adapter (85%)
3. Technician visit (78%)
```

## Example End-to-End Flow

1. Input Ticket
2. Embedding Generation
3. Vector Search
4. Context Injection using RAG
5. LLM Solution Generation
6. Final Output to User

## Folder Structure
```
.
├── data/
│   └── telecom_tickets_cleaned.csv
├── embeddings/
├── model/
├── code/
└── README.md
```

## Challenges Solved

- **Limited real telecom data** - embedding + vector search solves retrieval
- **Knowledge grounding** - solved using RAG
- **Avoid hallucination** - only real ticket solutions are used

## Future Enhancements

- Deploy UI using Streamlit
- Add multi-class ticket categorization
- Add feedback loop for model improvement
- Cloud deployment (Azure / GCP)

## Conclusion

This system builds a practical GenAI-powered assistant for telecom ticket resolution using real industry workflows. It retrieves past issues, compares similarity, produces actionable solutions, and improves support efficiency.