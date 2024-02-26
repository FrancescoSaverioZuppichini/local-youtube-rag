from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_dl import YoutubeDL


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


def download_subtitles_from_video_url(video_url: str, out_dir: Path) -> Path:
    video_id = get_id_from_video_url(video_url)
    output_dir = out_dir / video_id
    output_dir.mkdir(exist_ok=True)
    outtmpl = str(output_dir / "subtitles")

    ydl_opts = {
        "writesubtitles": True,
        "subtitlesformat": "vtt",
        "subtitleslangs": ["en"],
        "writeautomaticsub": True,
        "outtmpl": outtmpl,
        "skip_download": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    output_path = output_dir / "subtitles.en.vtt"

    if not output_path.exists():
        raise KeyError("Not Subtitles")
    return output_path
