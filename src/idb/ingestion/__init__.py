"""文件摄入模块"""
from idb.ingestion.parser import parse_file
from idb.ingestion.chunker import chunk_document
from idb.ingestion.embedder import embed_chunks

__all__ = ["parse_file", "chunk_document", "embed_chunks"]
