## 1. Project Overview

This project is a Multimodal RAG pipeline that can retrieve and answer using multiple data types:

- Text
- Images
- Audio
- Video

The system first searches the local database. If relevant data is missing or weak, it performs a Wikipedia cold start, stores new data, and reruns retrieval.

Example:
User: tell me about Taj Mahal
System returns:
- text answer
- related images
- audio if available
- video if available

User: give me lion video
System returns:
- only lion video if relevant
- otherwise "No relevant video found"


## 2. Project Structure

![image alt](https://github.com/NARASAPURAPUTEJASRI/multi-model-rag/blob/main/Rag_project_structure.jpeg)

## 3. File Explanation                                                                                         
**main.py** - Main backend controller. Handles query flow, retrieval, cold start, evaluation logs,and response.

**config.py** - Stores API key, model names, paths, and modality thresholds.

**embeddings.py** - Creates text embeddings and raw media embeddings using Gemini Embedding 2.

**vector_store.py** - Stores and searches embeddings in ChromaDB using cosine similarity.

**bm25_store.py** - Builds BM25 keyword index from text, captions, and descriptions.

**hybrid.py** - Combines vector search and BM25 results using RRF.

**reranker.py** - Adds entity-aware boosting after RRF.

**ingestion.py** - Embeds and stores text/media into ChromaDB and BM25.

**cold_start.py** - Searches Wikipedia, downloads text/media, embeds, stores, and updates BM25.

**dedupe_store.py** - Prevents duplicate text/media storage using URL/hash tracking.

**intent.py** - Detects whether query asks for text, image, audio, video, or mixed response.

**response_router.py** - Converts final results into frontend response format.

**evaluation.py** - Calculates retrieval and answer quality metrics after response generation.

**logger.py** - Logs pipeline events into terminal and logs/pipeline.log.

**script.js** - Sends frontend query to backend and displays text/media.

**index.html** - Frontend page structure.

**style.css** - Frontend UI styling.          


## 4. Tech Stack

## Backend

Python

FastAPI

Uvicorn

ChromaDB

BM25Okapi

Wikipedia API

Requests

FFmpeg

Gemini API

## Frontend

HTML

CSS

JavaScript

## AI Models

Gemini Embedding 2

Gemini 2.5 Flash


## 5. What Backend Handles

## Backend handles:

Query receiving

Intent detection

Query embedding

Vector search

BM25 search

RRF merging

Entity reranking

Final relevance gate

Cold start

Wikipedia scraping

Media download

Deduplication

Response routing

Evaluation logging


## 6. What LLM Handles

LLM is used for:

Generating text answers from retrieved context

Describing image/audio/video for BM25 keyword search

Evaluating final answer quality in evaluation.py

LLM is not used for the main retrieval decision.


## 7. What Embedding Handles

Gemini Embedding 2 creates embeddings for:

text → text embedding
image → raw image embedding
audio → raw audio embedding
video → raw video embedding

All embeddings are stored in the same ChromaDB collection.

## Important rule:
Captions/descriptions are NOT used as semantic media embeddings.
They are used only for BM25 keyword search and display support.


## 8. Retrieval Scoring Workflow

## Step 1: Vector Similarity Search

Query is converted into one embedding vector.

query = "give me lion video"-> Gemini Embedding 2 -> query vector

ChromaDB compares query vector with stored vectors using cosine distance.

Similarity is calculated as:

vector_score = 1 - cosine_distance

Higher score means more semantically similar.

## Step 2: BM25 Keyword Score

BM25 searches keyword text:

text chunks

captions

descriptions

page title

source topic

BM25 formula idea:

BM25 = TF × IDF × length normalization

Where:

TF = how often query word appears

IDF = how rare/important word is

length normalization = avoids favoring very long documents

BM25 helps exact keyword matching.

## Step 3: RRF Merge

Vector search and BM25 return separate ranked lists.

RRF combines them using:

RRF score = 1 / (k + rank)

In this project:

k = 60

rank 1 = 1 / 61 = 0.01639

rank 2 = 1 / 62 = 0.01612

If an item appears in both vector and BM25 results, its RRF score increases.

## Step 4: Entity Boosting

The system extracts important words from the query.

## Example:

give me lion video

↓

entity = lion

If retrieved result contains lion in caption, description, page title, or source topic:

entity_boost = match_count × 0.08

## Final score becomes:

final_score = RRF score + entity_boost

## Example:

RRF score = 0.032

entity_boost = 0.08

final_score = 0.112

## 9. Cold Start Threshold Logic

Cold start does not depend only on vector similarity.

It depends on final score after:

Vector search

+ BM25 search
 
+ RRF merge
 
+ Entity reranking
  
+ Intent filtering
  
+ Final relevance gate

Thresholds:

TEXT_FINAL_THRESHOLD  = 0.07

IMAGE_FINAL_THRESHOLD = 0.08

AUDIO_FINAL_THRESHOLD = 0.08

VIDEO_FINAL_THRESHOLD = 0.08

MIXED_FINAL_THRESHOLD = 0.08

If:

final_score < threshold

or required modality is missing, cold start triggers.

## Example:

User: give me lion video

Final score = 0.016

Video threshold = 0.08

↓

Cold start triggered

## 10. Evaluation Metrics

The evaluation system measures how well the multimodal RAG pipeline performs after the final response is generated.

Evaluation is completely separate from retrieval and cold start logic.

Pipeline flow:

Retrieval

→ Reranking

→ Relevance Filtering

→ Response Generation

→ Evaluation

Evaluation does not affect:

vector retrieval

BM25 retrieval

RRF merging

entity boosting

threshold filtering

cold start triggering

Cold start decisions use only retrieval scores and thresholds.

Evaluation metrics are logged only in backend logs and are not displayed in the frontend UI.

Example backend log:

Evaluation metrics | metrics={

  'context_precision': 0.86,
  
  'modality_correctness': 1.0,
  
  'answer_relevancy': 0.91,
  
  'final_correctness_score': 0.91,
  
  'retrieved_count': 7,
  
  'response_type': 'text'
  
  }
  
Context Precision

Context Precision measures how many retrieved results are relevant to the user query.

The system:

extracts important query entities

compares them with retrieved results

calculates the relevance ratio

Formula:

context_precision =

relevant_retrieved_results / total_retrieved_results

Example:

retrieved_results = 10

relevant_results = 8

context_precision = 0.8

This metric evaluates retrieval quality.

Modality Correctness

Modality Correctness checks whether the response type matches the detected user intent.

Examples:

image query → only image results

audio query → only audio result

video query → only video result

text query → text answer with related media

Scoring:

Correct modality response = 1.0

Wrong modality response = 0.0

This metric evaluates response routing correctness.

Answer Relevancy

Answer Relevancy checks whether the generated answer correctly addresses the user query.

This metric is evaluated using Gemini 2.5 Flash as an LLM judge.

The evaluation model receives:

user query

generated answer

and returns a score between:

0.0 → irrelevant answer

0.5 → partially relevant

1.0 → highly relevant answer

This metric is used only for text-based responses.

Media-only responses do not use answer relevancy evaluation because they do not generate LLM text answers.

Final Correctness Score

The final correctness score combines:

retrieval quality

modality correctness

answer quality

For text queries:

final_correctness_score =

(0.4 × context_precision)

+ (0.3 × modality_correctness)
+ 
+ (0.3 × answer_relevancy)

For image/audio/video queries:

final_correctness_score =

(0.6 × context_precision)

+ (0.4 × modality_correctness)

LLM answer relevancy is not used for media-only responses.

Evaluation Design Principles

The evaluation system follows these principles:

evaluation is independent from retrieval

evaluation never controls cold start

evaluation never changes retrieval ranking

evaluation runs only after final response generation

evaluation metrics are backend-only

frontend never displays evaluation metrics

media-only queries do not generate LLM text answers

text queries evaluate both retrieval quality and answer quality

## 11. Complete Architecture

![image alt](https://github.com/NARASAPURAPUTEJASRI/multi-model-rag/blob/main/Rag_pipline_Image.jpeg)

## 12. Frontend Behavior

tell me about Taj Mahal

→ text + 2 images + 1 audio + 1 video if available

give me Taj Mahal images

→ 2 images

give me Taj Mahal audio

→ only audio

give me Taj Mahal video

→ only video

## 13. How to Run

## Step 1: Install backend dependencies

cd backend

pip install -r requirements.txt

## Step 2: Create .env

Inside backend/.env:

GEMINI_API_KEY=your_api_key_here

CHROMA_PATH=./data/chroma_db

MEDIA_DIR=./data/media

SIMILARITY_THRESHOLD=0.25

## Step 3: Start backend

cd backend

python -m uvicorn app.main:app --reload --port 8000

Backend docs:

http://127.0.0.1:8000/docs

## Step 4: Start frontend

Open another terminal:

cd frontend

python -m http.server 5500

Open:

http://127.0.0.1:5500

## 14. Important Debug Command

Check BM25 pickle

cd backend

python debug_bm25_pickle.py

## 15. Summary

The system avoids returning irrelevant media by using final relevance validation. If local data is weak or missing, it dynamically fetches and stores new data from Wikipedia.




















