import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, cast

import chainlit as cl
import tiktoken
from bs4 import BeautifulSoup
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader, TextLoader, UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader)
from langchain_community.vectorstores import FAISS
from langchain_core.stores import BaseStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from rank_bm25 import BM25Okapi

from translator_cache import cache_translate

client = OpenAI()


settings = {
    "model": "gpt-4o",
    "temperature": 0.5,
}


file_content = ""
parent_child_retriever: Optional[ParentDocumentRetriever] = None
last_user_message = ""
bm25_index = None
bm25_children = []
INDEX_PATH = "faiss_index"
FAISS_DIR = "faiss_child"
PARENTS_JSONL = "parents.jsonl"
loader_executor = ThreadPoolExecutor(max_workers=4)
emb_query = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=16)

# Remove HTML tags and conversation labels (e.g., "role: user") 
def sanitize_text(text: str) -> str:
    text = re.sub(r'(?im)^(role|assistant|system|user)\s*:\s*', '', text)
    text = BeautifulSoup(text, "lxml").get_text(" ")
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\.\-/:]+|[\u3040-\u30ff\u4e00-\u9fff]+", text.lower())

# Lightweight language detection (Japanese vs. English) 
# based on Unicode ranges
def detect_language(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        if (0x3040 <= cp <= 0x30FF) or (0x4E00 <= cp <= 0x9FFF) or (0x3000 <= cp <= 0x303F):
            return "ja"
    return "en"


enc = tiktoken.get_encoding("cl100k_base")


def tok_len(text: str) -> int:
    return len(enc.encode(text))


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120,
    separators=["\n\n", "\n", "。", "．", ".", " "],
    length_function=tok_len,
)


parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1600,
    chunk_overlap=160,
    separators=["\n\n", "\n", "。", "．", ".", " "],
    length_function=tok_len,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=60,
    separators=["\n\n", "\n", "。", "．", ".", " "],
    length_function=tok_len,
)


def _load_file_chunks(LoaderClass, file_path: Path) -> list[Document]:
    docs: list[Document] = []
    loader = LoaderClass(str(file_path))
    for raw in loader.load():
        clean_text = sanitize_text(raw.page_content)
        for idx, chunk in enumerate(splitter.split_text(clean_text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path.name,
                        "chunk_index": idx,
                        "lang": detect_language(chunk),
                        **raw.metadata},
                )
            )
    return docs


def load_all_documents_as_chunks(base_dir: str) -> list[Document]:
    all_docs = []
    base_path = Path(base_dir)
    loader_map = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".xls": UnstructuredExcelLoader,
        ".xlsx": UnstructuredExcelLoader,
    }
    futures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for file_path in base_path.rglob("*"):
            ext = file_path.suffix.lower()
            if ext not in loader_map:
                continue
            LoaderClass = loader_map[ext]
            futures.append(
                executor.submit(_load_file_chunks, LoaderClass, file_path)
            )
        for future in futures:
            docs = future.result()
            all_docs.extend(docs)
    return all_docs


def vectorize_documents(documents: list[Document]) -> FAISS:
    return FAISS.from_documents(documents, emb_query)


@cache_translate
def translate_ja_to_en(text: str) -> str:
    print("[Debug] translate_ja_to_en input:", text)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a translation assistant."},
            {
                "role": "user",
                "content": (
                    "Translate the following Japanese into English without answering the question "
                    "or adding any extra explanation. Provide only a literal translation:\n\n"
                    + text
                ),
            },
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    print("[Debug] translate_ja_to_en output:", content)
    return content.strip()


@cache_translate
def translate_en_to_ja(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a translation assistant. Translate English to Japanese."},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


def build_parent_child_retriever(documents: list[Document]) -> ParentDocumentRetriever:
    child_vs = FAISS.from_texts(["__seed__"], emb_query, metadatas=[{"seed": True}])

    parent_store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=child_vs,
        docstore=parent_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(documents)

    return retriever


def save_faiss():
    if parent_child_retriever is None:
        return
    vs = cast(FAISS, parent_child_retriever.vectorstore)
    vs.save_local(FAISS_DIR)


def load_faiss() -> FAISS:
    return FAISS.load_local(FAISS_DIR, emb_query, allow_dangerous_deserialization=True)


def save_parent_docstore(store: BaseStore[str, Document], path: str = PARENTS_JSONL) -> None:
    keys = list(store.yield_keys())
    with open(path, "w", encoding="utf-8") as f:
        for k in keys:
            v = store.mget([k])[0]
            if v is None:
                continue
            if hasattr(v, "page_content"):
                content = v.page_content
                meta = getattr(v, "metadata", {})
            else:
                content = str(v)
                meta = {}
            rec = {"doc_id": k, "content": content, "metadata": meta}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_parent_docstore(path: str = PARENTS_JSONL) -> InMemoryStore:
    store = InMemoryStore()
    p = Path(path)
    if not p.exists():
        return store
    pairs = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            doc = Document(page_content=rec["content"], metadata=rec.get("metadata", {}))
            pairs.append((rec["doc_id"], doc))
    if pairs:
        store.mset(pairs)
    return store


def build_bm25_from_child_vs():
    global bm25_index, bm25_children
    if parent_child_retriever is None:
        raise RuntimeError("Retriever not initialized")
    vs = cast(FAISS, parent_child_retriever.vectorstore)
    vs_any = cast(Any, vs)
    store = getattr(vs_any.docstore, "_dict", {})
    ids = list(vs_any.index_to_docstore_id.values())
    bm25_children = []
    for sid in ids:
        d = store.get(sid)
        if not d:
            continue
        if d.metadata.get("seed"):
            continue
        bm25_children.append(d)
    tokenized = [tokenize(d.page_content) for d in bm25_children]
    bm25_index = BM25Okapi(tokenized)


def bm25_child_search(query: str, topn: int = 24):
    if not bm25_index or not bm25_children:
        return []
    scores = bm25_index.get_scores(tokenize(query))
    import numpy as np
    idxs = np.argsort(scores)[::-1][:topn]
    return [bm25_children[i] for i in idxs]


# Hybrid retrieval: FAISS (vector search with MMR) + BM25 interleaving
# Optionally applies translation boost if results are biased to one language
def bilingual_search(query: str, k: int = 4, fetch_k: int = 24) -> list:  # noqa: C901
    if not isinstance(query, str):
        raise TypeError("クエリは文字列である必要があります。")
    if parent_child_retriever is None:
        raise RuntimeError("Retriever がまだ初期化されていません。")
    retriever = parent_child_retriever          # type: ignore[assignment]
    assert retriever is not None
    vs = cast(FAISS, retriever.vectorstore)
    query_lang = detect_language(query)

    qv = emb_query.embed_query(query)
    base_hits = vs.max_marginal_relevance_search_by_vector(
        qv, k=min(k*3, 12), fetch_k=fetch_k, lambda_mult=0.5
    )
    bm25_hits = bm25_child_search(query, topn=fetch_k)

    def fetch_parents(child_docs):
        pid_order, seen_id = [], set()
        for cd in child_docs:
            if cd.metadata.get("seed"):
                continue
            pid = cd.metadata.get("doc_id")
            if not pid or pid in seen_id:
                continue
            seen_id.add(pid)
            pid_order.append(pid)
        parents = retriever.docstore.mget(pid_order)
        out = []
        for pid, pd in zip(pid_order, parents):
            if pd is None:
                continue
            if hasattr(pd, "page_content"):
                meta = {**getattr(pd, "metadata", {}), "doc_id": pid}
                content = pd.page_content
            else:
                meta = {"doc_id": pid}
                content = str(pd)
            if meta.get("lang") not in ("ja", "en"):
                meta["lang"] = detect_language(content)
            out.append(Document(page_content=content, metadata=meta))
        return out

    base_parents = fetch_parents(base_hits)
    langs = [d.metadata.get("lang") for d in base_parents]
    ja_ratio = (langs.count("ja") / len(langs)) if langs else 0
    en_ratio = (langs.count("en") / len(langs)) if langs else 0
    need_boost = (ja_ratio > 0.8 or en_ratio > 0.8)
    merged = []
    seen = set()

    def push_parents(child_docs):
        pid_order, seen_id = [], set()
        for cd in child_docs:
            if cd.metadata.get("seed"):
                continue
            pid = cd.metadata.get("doc_id")
            if not pid or pid in seen_id:
                continue
            seen_id.add(pid)
            pid_order.append(pid)

        parents = retriever.docstore.mget(pid_order)
        for pid, pd in zip(pid_order, parents):
            if pd is None:
                continue
            if hasattr(pd, "page_content"):
                parent_doc = pd
                parent_doc.metadata = {**getattr(pd, "metadata", {}), "doc_id": pid}
            else:
                parent_doc = Document(page_content=str(pd), metadata={"doc_id": pid})

            key = parent_doc.metadata.get("doc_id") or (
                parent_doc.metadata.get("source"),
                parent_doc.metadata.get("chunk_index"),
            )
            if key not in seen:
                seen.add(key)
                merged.append(parent_doc)
                if len(merged) >= k:
                    return

    def interleave(a, b):
        out = []
        i = j = 0
        while i < len(a) or j < len(b):
            if i < len(a):
                out.append(a[i])
                i += 1
            if j < len(b):
                out.append(b[j])
                j += 1
        return out

    for cd in interleave(base_hits, bm25_hits):
        if len(merged) >= k:
            break
        push_parents([cd])

    if need_boost and len(merged) < k:
        try:
            if query_lang == "ja":
                t = translate_ja_to_en(query)
            else:
                t = translate_en_to_ja(query)
            if t:
                qv2 = emb_query.embed_query(t)
                boost_hits = vs.max_marginal_relevance_search_by_vector(
                    qv2, k=min(k*3, 12), fetch_k=fetch_k, lambda_mult=0.5
                )
                push_parents(boost_hits)
        except Exception as e:
            print("[Warn] 翻訳フォールバック失敗:", e)

    return merged[:k]


def create_prompt(user_message, relevant_texts):
    system_message = (
        "専門的な知識をもつQAアシスタントです。"
        "以下の資料断片をもとに、正確かつ明瞭に回答してください。"
    )
    assistant_message = "\n".join(relevant_texts)
    return [
        {"role": "system", "content": system_message},
        {"role": "assistant", "content": assistant_message},
        {"role": "user", "content": user_message.strip()},
    ]


async def call_openai_api(user_message, relevant_texts):
    try:
        messages = create_prompt(user_message, relevant_texts)
        response = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=settings["temperature"],
        )
        return response.choices[0].message.content, relevant_texts
    except Exception as e:
        return f"Error: {str(e)}", []


def format_sources(references):
    return "\n".join([f"参照元 {i+1}:\n{ref}" for i, ref in enumerate(references)])


async def send_feedback_options():
    feedback_options = [
        cl.Action(name="like", label="👍 良い", payload={"feedback": "liked"}),
        cl.Action(name="dislike", label="👎 悪い", payload={"feedback": "disliked"}),
    ]
    await cl.Message(content="この回答は役に立ちましたか？", actions=feedback_options).send()


@cl.on_app_startup
async def startup_event():
    global parent_child_retriever

    base_dir = "./knowledge_base"
    print(f"[Startup] Knowledge Base を {base_dir} から読み込み/復元…")

    try:
        child_vs = load_faiss()
        parent_store = load_parent_docstore()
        if child_vs is not None and len(list(parent_store.yield_keys())) > 0:
            parent_child_retriever = ParentDocumentRetriever(
                vectorstore=child_vs,
                docstore=parent_store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            build_bm25_from_child_vs()
            print([d.metadata.get("doc_id") for d in bilingual_search("テスト", k=3)])
            print([d.metadata.get("doc_id") for d in bilingual_search("test query", k=3)])
            print("[Startup] 既存インデックスから復元しました。")
            return
        else:
            print("[Startup] 復元対象が不十分（空）。再構築に切替。")
    except Exception as e:
        print("[Startup] 復元失敗。再構築に切替:", e)

    docs = await asyncio.to_thread(load_all_documents_as_chunks, base_dir)
    if not docs:
        print("[Error] ドキュメントが見つかりません。")
        return

    print(f"[Startup] ドキュメント総数: {len(docs)}。Parent-Child Retriever 構築中…")
    parent_child_retriever = await asyncio.to_thread(build_parent_child_retriever, docs)
    build_bm25_from_child_vs()

    try:
        save_faiss()
        save_parent_docstore(parent_child_retriever.docstore)
        print("[Startup] 新規構築を保存しました。")
    except Exception as e:
        print("[Startup] 保存時に警告:", e)

    print("[Startup] 準備完了。")


@cl.on_message
async def main(message: cl.Message):
    global parent_child_retriever, last_user_message

    user_message = message.content.strip()
    last_user_message = user_message

    if parent_child_retriever is None:
        await cl.Message(content="Knowledge Base の読み込み中です。しばらくお待ちください。").send()
        return

    try:
        docs = bilingual_search(user_message, k=3)
    except Exception as e:
        await cl.Message(content=f"検索中にエラーが発生しました: {e}").send()
        return

    relevant_texts = [doc.page_content for doc in docs]

    answer, _ = await call_openai_api(user_message, relevant_texts)

    await cl.Message(content=f"回答:\n{answer}").send()

    src_preview_lines = []
    for i, doc in enumerate(docs):
        preview = doc.page_content[:120].replace("\n", " ")
        src_preview_lines.append(f"[{i+1}] ({doc.metadata.get('source')}) {preview}...")

    await cl.Message(
        content="参照元（親ドキュメント）:\n" + "\n".join(src_preview_lines)
    ).send()

    await send_feedback_options()


@cl.action_callback("source_0")
async def show_source_0(action: cl.Action):
    await cl.Message(content=f"参照元 1 の内容:\n{action.payload}").send()


@cl.action_callback("source_1")
async def show_source_1(action: cl.Action):
    await cl.Message(content=f"参照元 2 の内容:\n{action.payload}").send()


@cl.action_callback("source_2")
async def show_source_2(action: cl.Action):
    await cl.Message(content=f"参照元 3 の内容:\n{action.payload}").send()


@cl.action_callback("like")
async def like_feedback(action: cl.Action):
    feedback = action.payload["feedback"]
    await cl.Message(content=f"フィードバック ({feedback}) ありがとう！").send()


# DEMO: For simplicity, we do not call any API or save feedback.
# Only acknowledge receipt of the feedback.
@cl.action_callback("dislike")
async def dislike_feedback(action: cl.Action):
    feedback = action.payload["feedback"]
    await cl.Message(content=f"フィードバック（{feedback}）ありがとうございます。今後の改善に活用させていただきます。").send()
