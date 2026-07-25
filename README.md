# kcampus-navigator

[English](./README.en.md) · **한국어**

한국 유학을 고려하는 외국인 학생이 **영어로 질문하면**, 숫자 질문은 **SQL**로 공공데이터를 조회하고 규정 질문은 **RAG**로 정부 문서를 검색해 **출처와 함께** 답합니다. 벡터 검색이 구조적으로 할 수 없는 집계·순위·비교는 라우터가 SQL로 보내고, 근거가 임계값에 못 미치면 **답을 생성하지 않고 거부**합니다. 비자·체류는 틀린 답 하나가 사람을 위험에 빠뜨리는 도메인이기 때문입니다.

Team 12 `SELECT * FROM Korea` 의 2026 BIGDATA-USC Conference Hackathon 프로젝트 — 발표 제품명 **Kcampus Navigator**.

## 무엇이 다른가

일반 LLM은 **학습 컷오프에 얼어붙은 지식**으로 출처 없이 자신 있게 답합니다. 네 가지로 차별화합니다.

- **거부(abstention)** — 근거가 임계값(0.42) 미만이면 문장을 만들지 않고, 가장 가까운 공식 주제와 담당 창구(하이코리아 ☎1345)를 안내합니다.
- **기준일(as-of)** — 모든 grounded 답변에 근거 문서의 수집일을 붙입니다. 일반 AI는 자기 지식이 최신인지 알 수 없습니다.
- **개인화** — 프로필을 주면 근거 문서의 **조건별 규정 중 그 학생에게 해당하는 가지**를 고릅니다. 값은 문서에서 '선택'할 뿐 지어내지 않습니다.

  | 프로필 | `How many hours can I work part-time on a D-2 visa?` |
  |---|---|
  | 없음 | 규정의 **모든 가지** — 학부 25h / 석·박사 30h / 우수자 30·35h / 어학요건 미달 10·15h |
  | 서울시립대 · 석사 · TOPIK 4 | **"주 30시간"** 한 줄 |

- **큐레이션 레이어 2종** — 공공데이터에도 정부 문서에도 없는 정보를 직접 만들었습니다. **장학금**(대학 홈페이지 공지가 유일 출처)과 **캠퍼스 생활 팁**(정부 문서가 다루지 않음). LLM 생성이 아니라 손으로 쓴 데이터라 할루시네이션이 0이고, 모르는 학교면 **지어내지 않고** 정부 문서 경로로 넘깁니다.

## 실행

```bash
python -m venv .venv && .venv\Scripts\activate     # macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                               # OPENAI_API_KEY 채우기

streamlit run app.py                               # 데모 UI (8501)
```

> 빌드된 `vectors.npz`·`kcampus.db` 가 저장소에 포함돼 있어 인덱스/DB 빌드는 생략할 수 있습니다.
> 다시 만들려면 `python src/build_db.py && python src/loader.py`.

`.env` 주요 항목: `OPENAI_API_KEY` · `LLM_MODEL=gpt-4o-mini` · `EMBEDDING_MODEL=text-embedding-3-small` · `CONFIDENCE_THRESHOLD=0.42`(미만이면 거부) · `FUSION=score`

**스택** — Python 3.12 · OpenAI API(LLM·임베딩) · numpy 벡터 스토어(chromadb/faiss 미사용) · rank-bm25 + kiwipiepy(한국어 형태소 BM25) · SQLite · Streamlit · pandas/matplotlib(EDA)

## 인터페이스

프론트는 함수 하나만 호출합니다. 계약은 `contract.py` 에 고정돼 있습니다.

```python
from src.pipeline import answer_question

answer_question("Can I work part-time on a D-2 visa?", lang="en", profile=None)
```

| 파라미터 | 설명 |
|---|---|
| `question` | 사용자 질문 (영어, 또는 `lang`) |
| `lang` | 답변 언어. 기본 `"en"` (`ko`·`zh` 지원) |
| `profile` | 선택 · `{visa, program, school, major, topik, nationality, grad_date, region}` 중 채운 것만. 하위호환 |

질문은 **큐레이션 레이어 두 개**(장학금 → 선배 라운지)를 먼저 지나고, 안 걸리면 **라우터**가 분류합니다. `refused` 는 라우터가 아니라 **검색 단계**에서 신뢰도가 임계값 미만일 때 결정됩니다.

| route | 언제 | 처리 |
|---|---|---|
| `rag` (My School) | 장학금 질문 + 프로필에 큐레이션된 학교 | `docs/scholarships.json` 을 과정·TOPIK 요건과 대조 → '지금 해당 / 요건 미달 / 정보 부족'. 학교 공지 링크·기준일 첨부. 모르는 학교면 통과 |
| `local` | 캠퍼스 생활·문화·행정 팁 | `docs/local_tips.json` 키워드 매칭(라우터 前) + RAG가 거부하려는 순간 의미 매칭으로 구제 |
| `sql` | 개수·순위·비교·집계 | Text-to-SQL → 표 + 막대차트 |
| `rag` | 규정·절차·자격 | 질의 한국어 번역 → 하이브리드 검색(BM25+Dense) → 출처 인용 |
| `hybrid` | 통계 + 규정 동시 | SQL + RAG |
| `refused` | 근거가 임계값 미만 | **생성 안 함** + 스마트 거부(가까운 공식 주제·창구 안내) |

### 응답 스키마 (`Answer`)

| 필드 | 설명 |
|---|---|
| `route` | `sql` \| `rag` \| `hybrid` \| `refused` \| `local` |
| `answer_text` | 최종 답변(영어). `refused` 면 `""` |
| `table_markdown` | SQL 결과 표. 없으면 `""` |
| `chart` | `{kind, x_label, y_label, labels, values}`. 없으면 `kind="none"` |
| `sources` | `{title, snippet, url, score}` 목록 |
| `confidence` | 검색 신뢰도 0~1 (Dense 최대 코사인) |
| `refused_reason` | `refused` 일 때만 채움 |

<details>
<summary>응답 예시 3종 (sql · refused · My School 장학금)</summary>

**`sql`** — `answer_question("Which universities in Seoul have the most international students?")`

```json
{
  "route": "sql",
  "answer_text": "The universities in Seoul with the most international students are Hanyang University with 7,437 students, ...",
  "table_markdown": "| univ_name | univ_name_en | students |\n|---|---|---|\n| 한양대학교 | HANYANG UNIVERSITY | 7437 |",
  "chart": { "kind": "bar", "x_label": "univ_name", "y_label": "students",
             "labels": ["한양대학교", "중앙대학교", "경희대학교"], "values": [7437, 4864, 4858] },
  "sources": [{ "title": "대학별 외국인 유학생 현황 (2025)", "snippet": "SQL: SELECT ...",
                "url": "https://www.data.go.kr/", "score": 1.0 }],
  "confidence": 1.0, "refused_reason": ""
}
```

> 유학생 수 = 등록 유학생 전체(어학·교환 포함) headcount — 전국 197,163명과 동일 기준.

**`refused`** — `answer_question("How do I get Korean citizenship?")`

```json
{
  "route": "refused", "answer_text": "", "confidence": 0.366,
  "refused_reason": "We don't have a verified official source for this (best match 0.366 < threshold 0.42), so we won't guess — a wrong visa or immigration answer can put you at real risk. For your specific situation, contact HiKorea (☎ 1345, hikorea.go.kr) ..."
}
```

**My School 장학금** — `profile={"school": "University of Seoul (서울시립대학교)", "program": "Master's", "topik": "4"}`

```
### Scholarships at University of Seoul (서울시립대학교)
Matched to your profile — Master's · TOPIK 4.

**You are eligible to apply for:**
- Korean Proficiency (TOPIK) Scholarship — partial tuition*
  ✅ you hold TOPIK 4 ≥ 4

**Depends on something we don't know about you:**
- Academic Excellence Scholarship — partial tuition*
  ℹ️ depends on your GPA — needs 3.5+ last semester, which your profile doesn't include

🗓 Curated from the university's own scholarship notice, as of 2026-07-25 ...
```

> TOPIK 2급 프로필이면 같은 항목이 `⚠️ needs TOPIK 4 — you have TOPIK 2` 로 바뀝니다. 판정은 JSON 의 조건(`levels`·`topik_min`·`gpa_min`)과 프로필 비교일 뿐 — LLM 이 문장을 만들지 않습니다.

</details>

## 구조

```
kcampus-navigator/
├── contract.py            # 프론트↔백엔드 계약 (Answer 스키마)
├── app.py                 # Streamlit 데모 UI (대조 뷰 · 페르소나 · 선배 카드)
├── src/
│   ├── pipeline.py        # 전체 조립: answer_question() 진입점
│   ├── router.py          # sql / rag / hybrid 분류 (LLM + 키워드 폴백)
│   ├── retriever.py       # 하이브리드 검색(BM25+Dense) + Abstention
│   ├── vector_store.py    # OpenAI 임베딩 + numpy 코사인 + 한국어 질의 번역
│   ├── sql_chain.py       # Text-to-SQL (한국어 값 용어집 + self-repair)
│   ├── scholarships.py    # My School: 학교별 장학금 개인화 (라우터 前)
│   ├── local.py           # 선배 라운지: 생활·문화 팁 매칭 (라우터 前)
│   ├── loader.py          # docs/*.md → 청크 → 임베딩 → vectors.npz
│   └── build_db.py        # 공공데이터 CSV → SQLite
├── docs/                  # 규정 코퍼스 46개 + local_tips.json + scholarships.json + 발표 그래픽(svg/png)
├── data/                  # raw CSV · processed(vectors.npz, kcampus.db)
├── notebooks/eda.ipynb    # 7 Steps EDA (결측 MAR · 국적 분포)
└── eval/                  # 평가셋 30문항 · 재보정 하니스 · 발표 프리플라이트
```

## 검증 명령

```bash
python eval/demo_check.py                  # 발표 프리플라이트: 데모 7장면 route 검증 (--dry 는 무API)
python src/router.py eval/questions.csv    # 라우터 분류 정확도 (29/30)
python eval/run_eval.py                    # 검색 재보정 (브릿지·전략·임계값 스윕)
python eval/sql_eval.py                    # SQL 정답 정확도 (--dry 는 무API 스키마 가드)
python src/scholarships.py                 # 장학금 분기 스모크 (무API)
python src/local.py                        # 선배 라운지 키워드 스모크 (무API)
```

**자체 측정 (2026-07-24 코퍼스, 원분수)** — 라우터 29/30 · Recall@3 13/16 · 거부 판별 21/22 @0.42 · EN→KO 브릿지 0.45→0.54 · SQL 정답 5/5

> SQL 채점은 시스템이 자기 답을 채점하지 않습니다 — 원본 CSV 에서 **pandas 로 정답을 따로 계산한 오라클**과 대조합니다 (`eval/sql_eval.py`). 분모가 5라는 것도 그대로 밝힙니다.

## 데이터 출처

규정 문서 46개는 전부 한국 정부 공식 출처입니다 — 하이코리아, 법무부 출입국·외국인정책본부 자격별 안내매뉴얼, Study in Korea(국립국제교육원), 국민건강보험공단. 대학 통계는 공공데이터(data.go.kr / 대학알리미, 2025). 학교별 장학금은 각 대학 공지에서 직접 큐레이션했으며(`docs/scholarships.json`, 서울시립대·숙명여대 2교), 금액·구간은 학기마다 바뀌므로 답변에 큐레이션 기준일과 국제교류처 확인 안내를 함께 노출합니다.
