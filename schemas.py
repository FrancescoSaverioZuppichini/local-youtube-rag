from typing import TypedDict


class DocumentMetadata(TypedDict):
    video_id: str


class Document(TypedDict):
    page_content: str
    metadata: DocumentMetadata


class Video(TypedDict):
    id: str
    title: str
