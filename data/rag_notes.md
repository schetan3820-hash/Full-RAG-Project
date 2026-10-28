# RAG Study Notes (sample document)

This is a sample document so that the project works out of the box. Replace it with your own
PDFs, notes, or books by putting them in the `data/` folder.

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is a technique where a language model answers a question
using documents retrieved from an external knowledge base. Instead of relying only on what the
model learned during training, the system first searches a document collection for relevant
passages and then gives those passages to the model as context. This reduces hallucination and
lets the model use private or recent information without retraining.

A RAG system has two phases. The indexing phase runs offline: documents are loaded, split into
chunks, converted into embeddings, and stored in a vector index. The query phase runs for every
question: the question is embedded, similar chunks are retrieved, and a large language model
generates an answer grounded in those chunks.

## Chunking

Chunking splits long documents into smaller pieces. Chunks that are too large dilute the
relevant information and waste context window tokens. Chunks that are too small lose context.
A common starting point is 500 to 1000 characters with an overlap of 10 to 20 percent. The
overlap ensures that a sentence cut at a chunk boundary still appears completely in at least
one chunk. Recursive character splitting tries paragraph breaks first, then line breaks, then
sentences, and finally words, so that chunks break at natural boundaries.

## Embeddings

An embedding model converts text into a dense vector. Texts with similar meaning have vectors
that are close together, usually measured with cosine similarity. BGE-small-en-v1.5 is a small
open-source embedding model that produces 384-dimensional vectors. BGE models expect a short
instruction to be prepended to queries but not to documents. Normalizing vectors to unit length
makes the dot product equal to cosine similarity.

## Hybrid search and BM25

Dense vector search captures meaning but often misses exact keywords such as product codes,
names, or acronyms. BM25 is a classic keyword ranking function based on term frequency, inverse
document frequency, and document length normalization. Hybrid search runs both dense search
and BM25 and merges the results. Reciprocal Rank Fusion (RRF) merges ranked lists by summing
1 / (k + rank) for each document, where k is usually 60. RRF only uses ranks, so it does not
need the two score scales to be normalized.

## Reranking

A bi-encoder embeds the query and each document separately, which is fast but approximate.
A cross-encoder reads the query and a document together and outputs a relevance score, which is
slower but much more accurate. A typical pipeline retrieves 20 to 50 candidates with hybrid
search and then uses a cross-encoder such as ms-marco-MiniLM-L-6-v2 to rerank them and keep the
top 3 to 5. Reranking is often the single biggest quality improvement in a RAG system.

## Conversational RAG

In a chat, follow-up questions such as "what about its limitations?" cannot be searched
directly because they depend on earlier turns. Query rewriting (also called query condensing)
uses the language model to turn the follow-up and the chat history into a standalone search
query before retrieval.

## Evaluating RAG

RAG evaluation measures both retrieval and generation. Hit rate checks whether a relevant
document appears in the top-k results. Mean Reciprocal Rank (MRR) averages 1 / rank of the first
relevant result. With an LLM acting as a judge, four common metrics are: correctness (answer
versus reference answer), relevance (answer versus question), groundedness (answer versus
retrieved documents, which detects hallucination), and retrieval relevance (retrieved documents
versus question). If retrieval relevance is low, improve chunking, embeddings, or search. If
retrieval is good but groundedness is low, improve the prompt or the model.

## Vectorless RAG

Vectorless approaches such as PageIndex build a hierarchical tree of a document, similar to a
table of contents, and let a language model reason over the tree to pick relevant sections.
This works well for long, structured documents like financial reports but costs an extra
language model call per query.
