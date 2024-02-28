from argparse import ArgumentParser
from pathlib import Path

from sentence_transformers import SentenceTransformer

from crud import get_db, get_video_from_db, save_video_to_db
from logger import logger
from rag import DocumentMetadata, embed, get_answer
from rag import get_db as get_vector_db
from rag import get_model_client, split_subtitles
from yt import (
    download_subtitles_from_video_url,
    get_id_from_video_url,
    get_info_from_video_url,
)

SUBTITLES_DIR = Path(".subtitles")
SUBTITLES_DIR.mkdir(exist_ok=True, parents=True)


def main(video_url: str):
    embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    db = get_db()
    vector_db = get_vector_db(vector_size=embeddings.get_sentence_embedding_dimension())
    logger.info("Connected to DB!")
    model_client = get_model_client()
    video_id = get_id_from_video_url(video_url)
    logger.info(f"[{video_id}] Processing ... ")
    if not get_video_from_db(video_id, db):
        logger.info(f"[{video_id}] Downloading ... ")
        video_info = get_info_from_video_url(video_url)
        subtitles_path = download_subtitles_from_video_url(video_url, SUBTITLES_DIR)
        logger.info(f"[{video_id}] Splitting ... ")
        chunks = split_subtitles(
            subtitles_path, metadata=DocumentMetadata(video_id=video_id)
        )
        logger.info(f"[{video_id}] Embeddings ... ")
        embed(vector_db, embeddings, chunks)
        save_video_to_db({"id": video_id, **video_info}, db)
        logger.info(f"[{video_id}] Updating DB ... ")
    logger.info(f"[{video_id}] Done!")
    while question := input(">"):
        answer, sources = get_answer(
            vector_db, embeddings, model_client, question, video_id
        )
        print(answer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", type=str, help="video url", required=True)
    args = parser.parse_args()
    video_url = args.i
    # video_url = "https://www.youtube.com/watch\?v\=D5u7trVY5Ho"
    main(video_url)
