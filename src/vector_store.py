"""
vector_store.py — OpenAI 임베딩 + numpy 코사인 유사도 검색 (다국어 브릿지 포함)

왜 chromadb / faiss 를 안 쓰는가
  - chromadb 는 fastapi / uvicorn / kubernetes / opentelemetry 등
    40개 넘는 의존성을 끌고 옵니다 (dry-run 으로 확인).
  - 우리 문서 규모는 수천 청크입니다. 이 정도면 numpy 브루트포스가
    더 빠르고, 의존성이 0이고, 디버깅이 투명합니다.
  - faiss 는 10만 건 이상일 때 의미가 있습니다. 해당 없음.

왜 질의를 한국어로 번역하는가  ★ v2 핵심
  질문은 영어, 문서는 한국어입니다. 측정해보니 정답 문서조차 유사도가
  0.35~0.57 에 머물렀고, 임계값 0.60 에서는 모든 질문이 REFUSE 됐습니다.
  특히 "extend my visa" 와 "체류기간 연장" 처럼 어휘가 전혀 겹치지 않는
  경우에는 정답 문서가 Top-2 에도 들지 못했습니다.
  → 교차언어 검색(cross-lingual retrieval) 성능 저하.
  이를 완화하기 위해 검색 직전에 질의를 한국어로 번역합니다.

핵심 최적화
  인덱싱 시점에 벡터를 L2 정규화해서 저장합니다.
  → 검색 시 코사인 유사도가 단순 내적(dot product)이 되어 훨씬 빠릅니다.

디스크 영속화가 중요한 이유
  당일 09:00에 임베딩하고 .npz 로 저장해두면, 이후 재시작·다른 노트북에서
  재임베딩이 필요 없습니다. API 비용도 시간도 아낍니다.
  → 15:00 프론트 노트북 검증 때 결정적입니다.

사용법
  from src.vector_store import VectorStore

  vs = VectorStore()
  vs.add(["문서1 내용", "문서2 내용"],
         metas=[{"title": "출입국관리법 제23조", "url": "..."}, {...}])
  vs.save()                      # data/processed/vectors.npz

  vs2 = VectorStore.load()
  hits = vs2.search("Can I work part-time on a D-2 visa?", k=5)
  vs2.last_search_query          # 실제로 검색에 쓰인 한국어 질의

단독 실행
  python src/vector_store.py     # 더미 문서로 번역 전후 비교 테스트
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

_client = OpenAI()
_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

DEFAULT_PATH = BASE / "data" / "processed" / "vectors.npz"
BATCH_SIZE = 100          # OpenAI 임베딩 API 권장 배치 크기


# =================================================================
#  임베딩
# =================================================================

def embed(texts: list[str], *, is_query: bool = False) -> np.ndarray:
    """텍스트 리스트를 L2 정규화된 (n, dim) float32 배열로 변환합니다.

    is_query 는 현재 사용하지 않습니다. OpenAI 임베딩은 로컬 e5 계열과 달리
    "query:" / "passage:" 접두사가 필요 없습니다. 나중에 로컬 모델로
    전환할 경우를 대비해 시그니처만 열어둡니다.
    """
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        # 빈 문자열은 API 가 거부합니다. 공백 하나로 치환.
        chunk = [t if t.strip() else " " for t in chunk]
        resp = _client.embeddings.create(model=_MODEL, input=chunk, timeout=60)
        vecs.extend(d.embedding for d in resp.data)

    arr = np.asarray(vecs, dtype=np.float32)
    # L2 정규화 → 이후 코사인 유사도 = 내적
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# =================================================================
#  다국어 브릿지 — 질의를 한국어로 번역
# =================================================================

_TRANSLATE_SYSTEM = """Translate the user's question into natural Korean \
suitable for searching Korean government and university documents.

Rules:
- Output ONLY the Korean translation. No explanation, no quotes.
- Use official Korean terminology, not colloquial words.
  e.g. "extend my visa"  -> "체류기간 연장"
       "part-time job"   -> "시간제 취업"
       "dropout rate"    -> "중도탈락률"
       "health insurance"-> "국민건강보험"
       "alien registration" -> "외국인등록"
- Keep codes and proper nouns as-is: D-2, TOPIK, KAIST.
- If the input is already Korean, return it unchanged.
- IMPORTANT: apply the glossary substitutions above LITERALLY. If a glossary
  term applies, the Korean output MUST contain that exact term.
- Output a noun-phrase style search query, not a polite full sentence.
  Good: "D-2 비자 시간제 취업 허가 조건"
  Bad:  "D-2 비자로 시간제 취업이 가능한가요?"
"""


def translate_to_korean(query: str) -> str:
    """검색용 한국어 질의로 변환합니다. 실패 시 원문을 그대로 반환합니다."""
    if not query or not query.strip():
        return query
    try:
        resp = _client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[
                {"role": "system", "content": _TRANSLATE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=120,
            timeout=10,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or query
    except Exception:
        # 번역 실패해도 검색은 계속되어야 합니다 (원문으로 폴백)
        return query


# =================================================================
#  벡터 스토어
# =================================================================

@dataclass
class Hit:
    score: float
    text: str
    meta: dict[str, Any]

    def __repr__(self) -> str:
        title = self.meta.get("title", "")[:40]
        return f"Hit({self.score:.3f}, {title!r}, {self.text[:50]!r}...)"


class VectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray = np.zeros((0, 1536), dtype=np.float32)
        self.texts: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self.last_search_query: str = ""     # 디버깅·발표용

    # ---------- 인덱싱 ----------

    def add(self, texts: Iterable[str],
            metas: Iterable[dict[str, Any]] | None = None) -> None:
        texts = list(texts)
        if not texts:
            return
        metas = list(metas) if metas is not None else [{} for _ in texts]
        if len(metas) != len(texts):
            raise ValueError(
                f"texts({len(texts)}) 와 metas({len(metas)}) 길이가 다릅니다"
            )

        new_vecs = embed(texts)
        if self.vectors.shape[0] == 0:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])
        self.texts.extend(texts)
        self.metas.extend(metas)

    # ---------- 검색 ----------

    def search(self, query: str, k: int = 5,
               translate: bool = True) -> list[Hit]:
        """코사인 유사도 상위 k개를 점수 내림차순으로 반환합니다.

        translate=True 이면 질의를 한국어로 번역한 뒤 검색합니다.
        문서가 한국어이므로 교차언어 유사도 저하를 크게 줄여줍니다.
        실제 사용된 질의는 self.last_search_query 에 남습니다.
        """
        if self.vectors.shape[0] == 0:
            self.last_search_query = query
            return []

        search_query = translate_to_korean(query) if translate else query
        self.last_search_query = search_query

        qv = embed([search_query], is_query=True)[0]     # (dim,)
        scores = self.vectors @ qv                       # 정규화됨 → 내적 = 코사인

        k = min(k, len(scores))
        # argpartition 으로 상위 k개만 추린 뒤 정렬 (전체 정렬보다 빠름)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return [
            Hit(float(scores[i]), self.texts[i], self.metas[i])
            for i in idx
        ]

    # ---------- 영속화 ----------

    def save(self, path: Path | str = DEFAULT_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors)
        # 텍스트·메타는 JSON 으로 별도 저장 (사람이 읽고 디버깅 가능)
        side = path.with_suffix(".json")
        side.write_text(
            json.dumps(
                {"model": _MODEL, "texts": self.texts, "metas": self.metas},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "VectorStore":
        path = Path(path)
        side = path.with_suffix(".json")
        if not path.exists() or not side.exists():
            raise FileNotFoundError(
                f"인덱스가 없습니다: {path}\n"
                f"먼저 add() 후 save() 를 실행하세요."
            )
        vs = cls()
        vs.vectors = np.load(path)["vectors"]
        data = json.loads(side.read_text(encoding="utf-8"))
        vs.texts = data["texts"]
        vs.metas = data["metas"]

        saved_model = data.get("model")
        if saved_model != _MODEL:
            print(f"⚠️  인덱스는 {saved_model} 로 만들어졌는데 현재 설정은 "
                  f"{_MODEL} 입니다. 재임베딩이 필요할 수 있습니다.")
        return vs

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        dim = self.vectors.shape[1] if len(self) else 0
        return f"VectorStore({len(self)} chunks, dim={dim})"


# =================================================================
#  단독 실행 — 번역 전후 비교 테스트
# =================================================================

_DUMMY_DOCS = [
    ("유학(D-2) 체류자격을 가진 사람이 시간제 취업을 하려면 사전에 "
     "체류자격 외 활동허가를 받아야 한다. 주당 허용 시간은 학위과정과 "
     "한국어능력(TOPIK) 등급에 따라 달라진다.",
     {"title": "출입국관리법 시행령 제23조", "url": "https://www.hikorea.go.kr/"}),

    ("학사과정 입학을 위해서는 일반적으로 TOPIK 3급 이상 또는 "
     "대학이 인정하는 영어 성적이 필요하다. 대학별로 요건이 다르므로 "
     "지원 전 해당 대학 국제처에 확인해야 한다.",
     {"title": "외국인 유학생 입학 안내", "url": "https://www.studyinkorea.go.kr/"}),

    ("외국인 등록은 입국일로부터 90일 이내에 관할 출입국·외국인청에 "
     "신청해야 한다. 미신청 시 범칙금이 부과될 수 있다.",
     {"title": "외국인등록 안내", "url": "https://www.hikorea.go.kr/"}),

    ("유학생은 입국 후 6개월이 지나면 국민건강보험에 당연 가입된다. "
     "보험료는 매월 고지되며 미납 시 체류기간 연장이 제한될 수 있다.",
     {"title": "유학생 건강보험 안내", "url": "https://www.nhis.or.kr/"}),

    ("체류기간 연장허가는 만료일 4개월 전부터 만료일까지 신청할 수 있다. "
     "성적 미달 또는 출석률 저조 시 연장이 거부될 수 있다.",
     {"title": "체류기간 연장 안내", "url": "https://www.hikorea.go.kr/"}),
]

# (질문, 기대 문서 제목)  — 마지막은 문서에 없는 질문
_TEST_QUERIES = [
    ("Can I work part-time on a D-2 visa?", "출입국관리법 시행령 제23조"),
    ("What TOPIK level do I need for a bachelor's program?", "외국인 유학생 입학 안내"),
    ("How do I extend my student visa?", "체류기간 연장 안내"),
    ("Do I need health insurance as an international student?", "유학생 건강보험 안내"),
    ("When must I register as a foreign resident?", "외국인등록 안내"),
    ("Which university has the most Chinese students?", None),   # 답 없음이 정답
]


if __name__ == "__main__":
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    print(f"\n임베딩: {_MODEL} | 번역: {_LLM_MODEL} | 임계값: {threshold}\n")

    vs = VectorStore()
    vs.add([t for t, _ in _DUMMY_DOCS], [m for _, m in _DUMMY_DOCS])
    saved = vs.save(BASE / "data" / "processed" / "_test_vectors.npz")
    vs2 = VectorStore.load(saved)
    print(f"인덱싱/로드 완료: {vs2}\n")

    raw_scores, ko_scores = [], []
    raw_hit, ko_hit = 0, 0
    answerable = 0

    print("=" * 96)
    for q, expect in _TEST_QUERIES:
        raw = vs2.search(q, k=1, translate=False)[0]
        raw_q = vs2.last_search_query
        ko = vs2.search(q, k=1, translate=True)[0]
        ko_q = vs2.last_search_query
        delta = ko.score - raw.score

        print(f"\nQ  {q}")
        print(f"   번역 → {ko_q}")
        print(f"   번역X : {raw.score:.3f}  {raw.meta['title']}")
        print(f"   번역O : {ko.score:.3f}  {ko.meta['title']}   ({delta:+.3f})")

        if expect is None:
            verdict = "REFUSE(정답)" if ko.score < threshold else "⚠️ 답변함(오답)"
            print(f"   판정  : {verdict}")
        else:
            answerable += 1
            raw_scores.append(raw.score)
            ko_scores.append(ko.score)
            raw_hit += (raw.meta["title"] == expect)
            ko_hit += (ko.meta["title"] == expect)
            mark = "OK" if ko.score >= threshold else "REFUSE"
            correct = "O" if ko.meta["title"] == expect else "X"
            print(f"   판정  : [{mark}] 문서정확도 {correct}")

    print("\n" + "=" * 96)
    print("  ※ 아래 숫자를 기록해두세요. 발표 슬라이드 재료입니다.")
    print("=" * 96)
    print(f"{'':22s} {'평균 Top-1 점수':>16s} {'Top-1 문서 정확도':>18s}")
    print(f"{'영어 질의 그대로':22s} {np.mean(raw_scores):16.3f} "
          f"{raw_hit}/{answerable:<18d}")
    print(f"{'한국어 번역 후':22s} {np.mean(ko_scores):16.3f} "
          f"{ko_hit}/{answerable:<18d}")
    print(f"{'개선폭':22s} {np.mean(ko_scores)-np.mean(raw_scores):+16.3f} "
          f"{ko_hit - raw_hit:+d}")
    print("=" * 96 + "\n")

    saved.unlink(missing_ok=True)
    saved.with_suffix(".json").unlink(missing_ok=True)