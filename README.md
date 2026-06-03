# Multimodal RAG Pipeline with Dynamic Cold Start and RAGAS Evaluation

## Project Overview

This project is a Multimodal Retrieval-Augmented Generation (RAG) Pipeline that supports:

- Text Retrieval
- Image Retrieval
- Audio Retrieval
- Video Retrieval
- Dynamic Wikipedia Cold Start
- Hybrid Search (Vector + BM25)
- Confidence-Based Retrieval Filtering
- RAGAS-Based Evaluation

The system retrieves relevant multimodal content using semantic embeddings and keyword search, generates responses based on user intent, and evaluates final outputs using RAGAS metrics.

---

# System Architecture

![image_alt](https://github.com/AnuradhaNama/RAG-MULTI-MODEL-/blob/main/final%20ragg%20t%20image.png)

# Project Structure

![image_alt](https://github.com/AnuradhaNama/RAG-MULTI-MODEL-/blob/ef9f635938050e641b45eeba3958ce35b35bd872/project%20structure%20ragg%20t%20image.png)

# Technologies Used

## Backend

- FastAPI
- Python

## Embeddings

- Gemini Embedding 2

## LLM

- Gemini 2.5 Flash

## Vector Database

- ChromaDB

## Keyword Search

- BM25

## Cold Start Source

- Wikipedia
- Wikimedia Commons

## Evaluation

- RAGAS

## Frontend

- HTML
- CSS
- JavaScript

---

# Intent Detection

Intent detection identifies the type of information requested by the user.

Supported intents:

```text
text
image
audio
video
```

Examples:

```text
tell me about tiger
→ text

show tiger images
→ image

give me tiger audio
→ audio

show tiger video
→ video
```

Intent detection determines:

- retrieval strategy
- modality filtering
- response generation
- evaluation workflow

---

# Embedding Workflow

## Text

Text chunks are embedded using:

```text
Gemini Embedding 2
```

Example:

```text
Tiger is the largest cat species...
```

↓

```text
Vector Embedding
```

---

## Media

Media embeddings are created directly from:

```text
Image Files

Audio Files

Video Files
```

Raw media files are used for semantic retrieval.

Metadata is NOT used for embeddings.

---

# Metadata Generation

Media metadata is generated using Gemini.

Generated metadata includes:

### Images

```text
Caption
Description
Objects
Scene
Concept
```

### Audio

```text
Sound Type
Speaker Information
Topic
Events
Audio Summary
```

### Video

```text
Scene Description
Objects
Actions
Events
Summary
```

Metadata is used only for:

- BM25 Search
- Evaluation
- Frontend Display

Metadata is NOT used for semantic embeddings.

---

# Hybrid Search Workflow

The retrieval pipeline combines:

```text
Semantic Search

+

Keyword Search
```

---

## Vector Search

Query embedding is searched in ChromaDB.

Similarity:

```text
Cosine Similarity
```

Formula:

```text
Similarity = 1 - Distance
```

Higher similarity means stronger semantic match.

---

## BM25 Search

BM25 performs keyword matching using:

- text chunks
- captions
- descriptions
- page titles
- source topics

---

# Reciprocal Rank Fusion (RRF)

Vector results and BM25 results are merged using:

```text
RRF Score

=

1 / (k + rank)
```

Where:

```text
k = 60
```

Example:

```text
Vector Rank = 1

BM25 Rank = 2

RRF

=

1/61 + 1/62

≈ 0.0325
```

---

# Entity Boosting

Important query entities are extracted.

Example:

```text
give me tiger audio
```

Entity:

```text
tiger
```

If retrieved metadata contains:

```text
tiger
```

additional score boost is applied.

Formula:

```text
Boost

=

Entity Match Count × 0.08
```

---

# Confidence Normalization

The final retrieval confidence is calculated using:

```text
Confidence

=

0.45 × Vector Score

+

0.25 × RRF Score

+

0.20 × Entity Match

+

0.10 × Modality Match
```

Output Range:

```text
0.0 → 1.0
```

Examples:

```text
0.90 = Highly Relevant

0.70 = Strong Match

0.50 = Moderate Match

0.20 = Weak Match
```

---

# Threshold Logic

Cold start decisions are based on confidence scores.

Current thresholds:

```text
TEXT_CONFIDENCE_THRESHOLD  = 0.25

IMAGE_CONFIDENCE_THRESHOLD = 0.30

AUDIO_CONFIDENCE_THRESHOLD = 0.30

VIDEO_CONFIDENCE_THRESHOLD = 0.30

MIXED_CONFIDENCE_THRESHOLD = 0.30
```

Decision:

```text
Best Confidence

>

Threshold
```

↓

```text
Use Existing Data
```

Otherwise:

```text
Cold Start Triggered
```

---

# Dynamic Cold Start

If confidence is below threshold:

```text
Wikipedia Search
```

↓

```text
Best Page Selection
```

↓

```text
Text Extraction
```

↓

```text
Media Discovery
```

↓

```text
Embedding Generation
```

↓

```text
Store in ChromaDB
```

↓

```text
Store in BM25
```

↓

```text
Re-run Retrieval
```

---

# Duplicate Prevention

Duplicate content is avoided using:

### Text Hash

```text
SHA256(text)
```

### File Hash

```text
SHA256(file)
```

### Media URL Tracking

Previously ingested URLs are skipped.

---

# Response Routing

## Text Intent

Returns:

```text
Text Answer

+

Related Images

+

Related Audio

+

Related Video
```

Limits:

```text
Images = 2

Audio = 1

Video = 1
```

---

## Image Intent

Returns:

```text
Images Only
```

---

## Audio Intent

Returns:

```text
Audio Only
```

---

## Video Intent

Returns:

```text
Video Only
```

---

# Frontend Workflow

Frontend sends:

```json
{
  "query": "tell me about tiger"
}
```

Backend returns:

```json
{
  "type": "text",
  "answer": "...",
  "media": [...]
}
```

Frontend dynamically renders:

- Text
- Images
- Audio Players
- Video Players

based on response type.

---

# Evaluation Framework

Evaluation runs AFTER response generation.

Pipeline:

```text
Retrieval

↓

Response Generation

↓

Frontend Response

↓

Evaluation
```

Evaluation never affects:

- vector retrieval
- BM25 retrieval
- RRF ranking
- confidence scores
- thresholds
- cold start

---

# Text Evaluation (RAGAS)

Text responses use:

## Answer Relevancy

Measures:

```text
Query

vs

Generated Answer
```

---

## Faithfulness

Measures:

```text
Generated Answer

vs

Retrieved Context
```

Detects hallucination.

---

## Final Text Score

```text
Final Score

=

0.45 × Answer Relevancy

+

0.40 × Faithfulness

+

0.15 × Modality Score
```

---

# Media Evaluation (RAGAS)

Media evaluation uses:

```text
RAGAS Multimodal Relevance
```

Inputs:

```text
User Query

+

Caption

+

Description

+

Page Title

+

Source Topic
```

Example:

```text
Query:
show tiger image

Caption:
Tiger in forest

Description:
Large Bengal tiger standing in jungle
```

RAGAS evaluates metadata relevance.

---

## Media Final Score

```text
Final Score

=

0.80 × Multimodal Relevance

+

0.20 × Modality Score
```

---

# Modality Score

Checks:

```text
Requested Modality

vs

Returned Modality
```

Examples:

```text
Image Query → Image Response

Audio Query → Audio Response

Video Query → Video Response

Text Query → Text Response
```

Scoring:

```text
Correct = 1.0

Incorrect = 0.0
```

---

# Example Evaluation Log

Text:

```text
Evaluation metrics | metrics={
'ragas_answer_relevancy': 0.84,
'ragas_faithfulness': 1.0,
'modality_score': 1.0,
'final_correctness_score': 0.93,
'retrieved_count': 16,
'response_type': 'text'
}
```

Media:

```text
Evaluation metrics | metrics={
'ragas_multimodal_relevance': 0.87,
'modality_score': 1.0,
'final_correctness_score': 0.90,
'retrieved_count': 1,
'response_type': 'audio'
}
```

---

# Logging

All pipeline events are stored in:

```text
logs/pipeline.log
```

Examples:

```text
Intent detected

Vector search completed

BM25 search completed

Cold start triggered

Response sent

Evaluation completed
```

---

# Environment Variables

```env
GEMINI_API_KEY=

CHROMA_PATH=./data/chroma_db

MEDIA_DIR=./data/media

EMBEDDING_MODEL=models/gemini-embedding-2

LLM_MODEL=gemini-2.5-flash

TEXT_CONFIDENCE_THRESHOLD=0.25

IMAGE_CONFIDENCE_THRESHOLD=0.30

AUDIO_CONFIDENCE_THRESHOLD=0.30

VIDEO_CONFIDENCE_THRESHOLD=0.30

MIXED_CONFIDENCE_THRESHOLD=0.30

RAGAS_DO_NOT_TRACK=true
```

---

# Run Backend

```bash
python -m uvicorn app.main:app --reload
```

---

# Run Frontend

Open:

```text
frontend/index.html
```

or use:

```bash
python -m http.server 5500
```

---

# Summary

This Multimodal RAG Pipeline supports:

- Text Retrieval
- Image Retrieval
- Audio Retrieval
- Video Retrieval
- Hybrid Search
- Dynamic Cold Start
- ChromaDB
- BM25
- Confidence Normalization
- Entity Boosting
- RAGAS Evaluation
- Metadata-Based Multimodal Evaluation
- Frontend Media Rendering
- Duplicate Prevention
- Wikipedia Knowledge Expansion

The system combines semantic retrieval, keyword retrieval, dynamic knowledge ingestion, and evaluation to provide accurate multimodal responses.
