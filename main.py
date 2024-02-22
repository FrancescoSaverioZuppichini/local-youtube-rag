import sqlite3
from sqlite3 import Connection
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from youtube_dl import YoutubeDL
from logger import logger
from typing import Optional, TypedDict
from rag import get_answer, get_db as get_vector_db, get_model_client, split_subtitles
from rag import clear as vector_db_clear
from rag import setup as vector_db_setup
from rag import embed, search
from sentence_transformers import SentenceTransformer


class Video(TypedDict):
    id: str
    title: str


SUBTITLES_DIR = Path(".subtitles")


def setup() -> Connection:
    SUBTITLES_DIR.mkdir(exist_ok=True, parents=True)
    conn = sqlite3.connect("main.db")

    conn.execute(
        """
    create table if not exists videos (
        id TEXT primary key,
        title TEXT
    )
    """
    )

    conn.commit()
    return conn


def get_video_from_db(video_id: str, conn: Connection) -> Optional[Video]:
    curr = conn.execute("SELECT * FROM videos WHERE id=?", (video_id,))
    row = curr.fetchone()
    if row:
        columns = [column[0] for column in curr.description]
        return dict(zip(columns, row))
    else:
        return None


def save_video_to_db(video: Video):
    conn.execute(
        "INSERT OR IGNORE INTO videos (id, title) VALUES (?, ?)",
        (video["id"], video["title"]),
    )
    conn.commit()


def get_id_from_video_url(video_url: str) -> str:
    query_string = urlparse(video_url).query
    video_id = parse_qs(query_string).get("v", [None])[0]
    if video_id is None:
        raise ValueError("Invalid YouTube URL or video ID not found.")
    return video_id


def get_info_from_video_url(video_url: str) -> dict:
    with YoutubeDL() as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        video_title = info_dict.get("title", "No Title")
    return {"title": video_title}


def download_subtitles_from_video_url(video_url: str) -> Path:
    video_id = get_id_from_video_url(video_url)
    output_dir = SUBTITLES_DIR / video_id
    output_dir.mkdir(exist_ok=True)
    outtmpl = str(output_dir / "subtitles")

    ydl_opts = {
        "writesubtitles": True,
        "subtitlesformat": "vtt",
        "subtitleslangs": ["en"],
        "outtmpl": outtmpl,
        "skip_download": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    output_path = output_dir / "subtitles.en.vtt"

    if not output_path.exists():
        raise KeyError("Not Subtitles")
    return output_path


video_url = "https://www.youtube.com/watch?v=etIL-f8NfIc"
embeddings = SentenceTransformer("all-MiniLM-L6-v2")
conn = setup()
vector_db = get_vector_db(vector_size=embeddings.get_sentence_embedding_dimension())
logger.info("Connected to DB!")
model_client = get_model_client()
video_id = get_id_from_video_url(video_url)
logger.info(f"[{video_id}] Processing ... ")
# if not get_video_from_db(video_id, conn):
#     logger.info(f"[{video_id}] Downloading ... ")
#     video_info = get_info_from_video_url(video_url)
#     subtitles_path = download_subtitles_from_video_url(video_url)
#     save_video_to_db({"id": video_id, **video_info})
#     embed(vector_db, embeddings, ["hey"], {"video_id": "123"})
# video_id = "etIL-f8NfIc"
subtitles_path = Path(".subtitles/etIL-f8NfIc/subtitles.en.vtt")
# chunks = split_subtitles(subtitles_path, {"video_id": video_id})
# vector_db_clear(vector_db)
# vector_db_setup(vector_db, vector_size=embeddings.get_sentence_embedding_dimension())
# embed(vector_db, embeddings, chunks )
# print(chunks)
# res = search(vector_db, embeddings, "Jesse Alexander", video_id=video_id)
# print(res)
answer, sources = get_answer(
    vector_db, embeddings, model_client, "Who is Jesse Alexander", video_id
)
print(answer)
