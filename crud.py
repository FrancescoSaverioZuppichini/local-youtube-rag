import sqlite3
from sqlite3 import Connection
from typing import Optional, TypedDict

from schemas import Video


def get_db() -> Connection:
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


def save_video_to_db(video: Video, conn: Connection):
    conn.execute(
        "INSERT OR IGNORE INTO videos (id, title) VALUES (?, ?)",
        (video["id"], video["title"]),
    )
    conn.commit()
