
import re
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader, UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader)


def sanitize_text(text: str) -> str:
    text = re.sub(r"\b(role|assistant|system|user)\b\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<.*?>", "", text)
    return text


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        if n - start <= max_chars:
            chunks.append(text[start:])
            break
        end = start + max_chars
        cut = text.rfind("\n", start, end)
        if cut != -1 and cut > start:
            chunks.append(text[start:cut + 1])
            start = cut + 1
            continue
        cut = text.rfind("。", start, end)
        if cut != -1 and cut > start:
            chunks.append(text[start:cut + 1])
            start = cut + 1
            continue
        chunks.append(text[start:end])
        start = end
    return chunks


def load_and_chunk_per_file(base_dir: str, max_chars: int = 800):
    loader_map = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
    }

    base_path = Path(base_dir)
    file_dict = {}

    for file_path in base_path.rglob("*"):
        ext = file_path.suffix.lower()
        if ext not in loader_map:
            continue

        try:
            loader = loader_map[ext](str(file_path))
            raw_docs = loader.load()
        except Exception as e:
            print(f"[Warning] 読み込み失敗: {file_path} → {e}")
            continue

        concatenated = []
        for doc in raw_docs:
            clean = sanitize_text(doc.page_content)
            concatenated.append(clean)
        raw_text = "\n".join(concatenated)

        chunks = chunk_text(raw_text, max_chars=max_chars)
        file_dict[str(file_path)] = (raw_text, chunks)

    return file_dict


if __name__ == "__main__":
    base_dir = "./knowledge_base"
    per_file_data = load_and_chunk_per_file(base_dir, max_chars=800)

    rows = []
    for file_path, (raw_text, chunks) in per_file_data.items():
        raw_len = len(raw_text)
        chunk_lens = [len(c) for c in chunks]
        total_chunks_len = sum(chunk_lens)
        num_chunks = len(chunks)
        diff = total_chunks_len - raw_len

        rows.append({
            "file": file_path,
            "raw_length": raw_len,
            "num_chunks": num_chunks,
            "sum_chunk_lengths": total_chunks_len,
            "length_diff": diff
        })

    df_summary = pd.DataFrame(rows)
    print("=== ファイル別チャンク化サマリ ===")
    print(df_summary)

    print("\n> length_diff が 0 以外のファイル")
    print(df_summary[df_summary["length_diff"] != 0])
