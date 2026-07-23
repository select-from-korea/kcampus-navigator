"""
loader.py — docs/ 문서를 청크·임베딩해 벡터 인덱스로 만드는 문서 로더

무엇을 하는가
  docs/*.md (frontmatter 포함) 를 읽어 본문을 청크로 만들고,
  VectorStore 에 임베딩해 data/processed/vectors.npz 로 저장합니다.
  당일 09:00 에 한 번 실행해두면 이후 재시작·다른 노트북에서 재임베딩이
  필요 없습니다 (vector_store.py 의 영속화 설계와 짝을 이룹니다).

왜 마크다운 frontmatter 인가
  데이터·기획 담당이 docs/ 에 넣는 규정 문서는 아래 형식입니다.

      ---
      title: 유학(D-2) 시간제취업 허용시간
      source: https://www.hikorea.go.kr/...   (또는 정부 매뉴얼 인용)
      retrieved: 2026-07-24
      topic: 시간제취업
      ---
      본문 (200~600자, 한 파일 = 한 주제) ...

  한 파일이 이미 한 주제·적정 길이로 쪼개져 있으므로 (SSOT §8 RAG 요건)
  파일 하나 = 청크 하나가 기본입니다. 본문이 예외적으로 길면 문단 단위로
  분할합니다. PDF 도 같은 파이프라인으로 흡수합니다 (pdfplumber, 문단+길이 청킹).

메타데이터 (검색 결과 → Source 계약으로 흐름)
  title    : 답변에 출처로 노출 (contract.Source.title)
  url      : source 안의 http(s) 링크를 추출, 없으면 None (contract.Source.url)
  source   : 원문 출처 문자열 전체 (URL 이 아닐 수 있음 — 정부 매뉴얼 인용 등)
  topic    : 주제 태그 (필터·디버깅용)
  retrieved: 수집일
  doc_id   : 파일명 (디버깅·중복 제거용)

사용법
  from src.loader import build_index
  vs = build_index()            # docs/ → vectors.npz

단독 실행
  python src/loader.py          # docs/ 전체를 임베딩해 인덱스 저장
  python src/loader.py --dry    # 임베딩 없이 파싱·청크 결과만 출력 (API 불필요)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE / "docs"

# 같은 폴더 실행(python src/loader.py)과 패키지 임포트 모두 지원
try:
    from .vector_store import VectorStore, DEFAULT_PATH
except ImportError:
    from vector_store import VectorStore, DEFAULT_PATH

_URL_RE = re.compile(r"https?://[^\s)>\]]+")

# 본문이 이보다 길면 문단 단위로 분할합니다. 우리 규정 문서는 200~600자라
# 대부분 단일 청크로 남습니다. PDF·긴 문서 대비용 안전장치입니다.
MAX_CHARS = 1200
OVERLAP = 150


# =================================================================
#  frontmatter 파싱
# =================================================================

def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """`--- ... ---` frontmatter 를 (메타 dict, 본문) 으로 분리합니다.

    frontmatter 가 없으면 ({}, 원문) 을 반환합니다. YAML 라이브러리를 쓰지
    않는 이유: 의존성을 늘리지 않기 위해서입니다. 우리 frontmatter 는
    `key: value` 한 줄짜리 평면 구조라 단순 파서로 충분합니다.
    """
    if not text.startswith("---"):
        return {}, text
    # 첫 줄(---) 이후부터 다음 --- 까지를 frontmatter 로 봅니다.
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        meta[key.strip()] = val.strip()
    return meta, body.strip()


def extract_url(source: str) -> str | None:
    """source 문자열에서 첫 http(s) 링크를 뽑습니다. 없으면 None.

    source 는 URL 일 수도(하이코리아 웹페이지) 인용 문자열일 수도(정부 매뉴얼)
    있습니다. 후자에도 괄호 안에 링크가 들어있는 경우가 많아 정규식으로 찾습니다.
    """
    if not source:
        return None
    m = _URL_RE.search(source)
    return m.group(0).rstrip(".,") if m else None


# =================================================================
#  청킹
# =================================================================

def chunk_text(text: str, max_chars: int = MAX_CHARS,
               overlap: int = OVERLAP) -> list[str]:
    """긴 본문을 문단 경계 우선으로 분할합니다.

    max_chars 이하이면 그대로 한 청크. 우리 규정 문서(200~600자)는 거의
    모두 단일 청크입니다. 문단(빈 줄) 경계를 지키되, 한 문단이 너무 길면
    문장 단위로 자르고 overlap 만큼 겹쳐 맥락 손실을 줄입니다.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                # 한 문단이 max_chars 초과 → 문장 단위로 자르며 overlap 유지
                sents = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s*", p)
                sb = ""
                for s in sents:
                    s = s.strip()
                    if not s:
                        continue
                    if len(sb) + len(s) + 1 <= max_chars:
                        sb = f"{sb} {s}".strip()
                    else:
                        if sb:
                            chunks.append(sb)
                        sb = (sb[-overlap:] + " " + s).strip() if sb else s
                buf = sb
    if buf:
        chunks.append(buf)
    return chunks


# =================================================================
#  문서 로딩
# =================================================================

def _meta_from(meta: dict[str, str], doc_id: str) -> dict[str, Any]:
    source = meta.get("source", "")
    return {
        "title": meta.get("title", doc_id),
        "url": extract_url(source),
        "source": source,
        "topic": meta.get("topic", ""),
        "retrieved": meta.get("retrieved", ""),
        "doc_id": doc_id,
    }


def load_markdown_docs(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict]]:
    """docs/*.md 를 (청크 텍스트, 메타) 리스트로 반환합니다.

    frontmatter 에 title 이 없는 파일(INDEX.md 등 목록/안내 파일)은 검색
    대상이 아니므로 건너뜁니다.
    """
    out: list[tuple[str, dict]] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        if not meta.get("title") or not body.strip():
            # frontmatter 없는 파일(INDEX 등)은 코퍼스에서 제외
            continue
        base_meta = _meta_from(meta, path.name)
        chunks = chunk_text(body)
        for i, ch in enumerate(chunks):
            m = dict(base_meta)
            m["chunk"] = i
            if len(chunks) > 1:
                m["title"] = f"{base_meta['title']} ({i + 1}/{len(chunks)})"
            out.append((ch, m))
    return out


def load_pdf_docs(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict]]:
    """docs/*.pdf 를 페이지 텍스트 → 청크로 반환합니다 (있을 때만).

    우리 코퍼스는 .md 로 정제돼 있지만, 정제 전 원본 PDF 를 바로 넣어도
    파이프라인이 흡수하도록 열어둡니다. pdfplumber 는 requirements 에 포함.
    """
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        return []
    import pdfplumber

    out: list[tuple[str, dict]] = []
    for path in pdfs:
        text_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        full = "\n\n".join(text_parts).strip()
        base_meta = {
            "title": path.stem, "url": None, "source": path.name,
            "topic": "", "retrieved": "", "doc_id": path.name,
        }
        chunks = chunk_text(full)
        for i, ch in enumerate(chunks):
            m = dict(base_meta)
            m["chunk"] = i
            m["title"] = f"{path.stem} ({i + 1}/{len(chunks)})"
            out.append((ch, m))
    return out


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict]]:
    """docs/ 의 md + pdf 를 모두 (텍스트, 메타) 청크로 반환합니다."""
    return load_markdown_docs(docs_dir) + load_pdf_docs(docs_dir)


# =================================================================
#  인덱스 구축
# =================================================================

def build_index(docs_dir: Path = DOCS_DIR,
                out_path: Path = DEFAULT_PATH) -> VectorStore:
    """docs/ 를 임베딩해 vectors.npz 로 저장하고 VectorStore 를 반환합니다."""
    docs = load_documents(docs_dir)
    if not docs:
        raise RuntimeError(
            f"임베딩할 문서가 없습니다: {docs_dir}\n"
            f"docs/ 에 frontmatter 를 갖춘 .md 파일이 있는지 확인하세요."
        )
    texts = [t for t, _ in docs]
    metas = [m for _, m in docs]

    vs = VectorStore()
    vs.add(texts, metas)
    saved = vs.save(out_path)
    print(f"✅ 인덱스 저장: {saved}")
    print(f"   문서 청크 {len(vs)}개 | 차원 {vs.vectors.shape[1]}")
    return vs


# =================================================================
#  단독 실행
# =================================================================

def _dry_run(docs_dir: Path = DOCS_DIR) -> None:
    """임베딩 없이 파싱·청크 결과만 점검합니다 (API 불필요)."""
    docs = load_documents(docs_dir)
    print(f"\n파싱된 청크: {len(docs)}개  (docs/ = {docs_dir})\n")
    topics: dict[str, int] = {}
    no_url = 0
    for text, m in docs:
        topics[m.get("topic", "")] = topics.get(m.get("topic", ""), 0) + 1
        if not m.get("url"):
            no_url += 1
    print(f"{'제목':50s} {'글자수':>6s}  URL")
    print("-" * 78)
    for text, m in docs[:8]:
        u = "○" if m.get("url") else "-"
        print(f"{m['title'][:48]:50s} {len(text):6d}   {u}")
    if len(docs) > 8:
        print(f"... 외 {len(docs) - 8}개")
    print("-" * 78)
    print(f"주제 분포: {dict(sorted(topics.items()))}")
    print(f"URL 없는 청크(정부 매뉴얼 인용 등): {no_url}/{len(docs)}\n")


if __name__ == "__main__":
    if "--dry" in sys.argv:
        _dry_run()
    else:
        build_index()
