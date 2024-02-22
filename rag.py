from pathlib import Path
import re
from openai import OpenAI
from logger import logger
from typing import Tuple, TypedDict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client import models
from typing import Optional, Tuple

COLLECTION_NAME = "embeddings"


class DocumentMetadata(TypedDict):
    video_id: str


class Document(TypedDict):
    page_content: str
    metadata: DocumentMetadata


def get_db(*args, **kwargs) -> QdrantClient:
    client = QdrantClient("localhost", port=6333)
    setup(client, *args, **kwargs)
    return client


def get_model_client() -> OpenAI:
    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        # required but ignored
        api_key="mistral",
    )
    return client


def setup(db: QdrantClient, vector_size: int):
    try:
        db.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        db.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.video_id",
            field_schema="keyword",
        )
    except UnexpectedResponse:
        pass


def clear(db: QdrantClient):
    db.delete_collection(COLLECTION_NAME)


def embed(
    db: QdrantClient,
    embeddings: SentenceTransformer,
    chunks: list[Document],
):
    embeddings = embeddings.encode(chunks)
    db.upload_records(
        collection_name="embeddings",
        records=[
            models.Record(id=idx, vector=emb.tolist(), payload=doc)
            for idx, (emb, doc) in enumerate(zip(embeddings, chunks))
        ],
    )


def search(
    db: QdrantClient,
    embeddings: SentenceTransformer,
    query: str,
    video_id: Optional[str] = None,
    limit: int = 4,
) -> list[Document]:
    query_filter = (
        models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.video_id", match=models.MatchValue(value=video_id)
                )
            ]
        )
        if video_id
        else None
    )
    hits = db.search(
        collection_name="embeddings",
        query_vector=embeddings.encode(query).tolist(),
        limit=limit,
        query_filter=query_filter,
    )
    documents = [Document(**hit.payload) for hit in hits]
    return documents


def split_subtitles(
    subtitles_path: Path,
    metadata: DocumentMetadata,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Document]:
    text = subtitles_path.read_text()
    cleaned_text = re.sub(
        r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n", "", text
    )
    cleaned_text = re.sub(r"&nbsp;", " ", cleaned_text)
    cleaned_text = re.sub(r"\n+", "", cleaned_text).strip()
    sentences = cleaned_text.split(".")
    logger.info(f"sentences {len(cleaned_text)}")
    documents = []
    document = Document(page_content="", metadata=metadata)
    for sentence in sentences:
        chunk_done = len(document["page_content"]) >= chunk_size
        if chunk_done:
            logger.info(f"new chunk = {len(document['page_content'])}")
            overlap_size = min(overlap, len(document["page_content"]))
            documents.append(document)
            document = Document(
                page_content=document["page_content"][-overlap_size:], metadata=metadata
            )
        document["page_content"] += sentence
    return documents


def get_answer(
    db: QdrantClient,
    embeddings: SentenceTransformer,
    model_client: OpenAI,
    question: str,
    video_id: str,
) -> Tuple[str, list[Document]]:
    prompt = Path("prompts/qa.prompt").read_text()
    documents = search(db, embeddings, question, video_id)
    context = "\n".join([document["page_content"] for document in documents])
    prompt = prompt.format(question=question, context=context)
    logger.info(prompt)
    chat_completion = model_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            }
        ],
        model="mistral",
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content, documents
