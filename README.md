# kcampus-navigator

[English](./README.en.md) · **한국어**

한국 유학을 고려하는 외국인 학생이 **영어로 질문하면**, 숫자 질문은 **SQL**로 공공데이터를 조회하고 규정 질문은 **RAG**로 정부 문서를 검색해 **출처와 함께** 답하는 의사결정 지원 시스템입니다. 벡터 검색이 구조적으로 할 수 없는 집계·순위·비교는 라우터가 SQL 경로로 보내 처리하며, 근거가 임계값에 못 미치면 **답변을 생성하지 않고 거부** 합니다. 비자·체류 규정은 틀린 답 하나가 사람을 위험에 빠뜨릴 수 있는 도메인이기 때문입니다. `SELECT * FROM Korea` 팀의 2026 BIGDATA-USC Conference Hackathon 프로젝트입니다.

## 기술 스택

- Python 3.12 (3.13/3.14 미지원)
- OpenAI API — LLM `gpt-4o-mini`, 임베딩 `text-embedding-3-small`
- numpy — 벡터 스토어(코사인 유사도, L2 정규화 후 내적). chromadb/faiss 미사용
- rank-bm25 + kiwipiepy — 한국어 형태소 기반 BM25 어휘 검색
- SQLite — 대학 통계 Text-to-SQL 대상 (`kcampus.db`)
- pandas, matplotlib — 데이터 처리·EDA
- pdfplumber — 원본 PDF 로딩
- Streamlit — 데모 UI

## 실행 방법

`.env` 파일을 생성해 다음 항목을 채웁니다. (`.env.example` 복사)

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CONFIDENCE_THRESHOLD=0.42     # 이 값 미만이면 답변 거부
FUSION=score                  # 검색 융합 방식 (score | rrf)
W_DENSE=1.0
W_BM25=1.0
```

의존성을 내려받고, 인덱스와 DB를 빌드한 뒤 실행합니다.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS: source .venv/bin/activate)
pip install -r requirements.txt

python src/build_db.py            # data/raw CSV → data/processed/kcampus.db (SQLite)
python src/loader.py              # docs/*.md → data/processed/vectors.npz (검색 인덱스)

streamlit run app.py              # 데모 UI (기본 포트 8501)
```

> 빌드된 `vectors.npz`·`kcampus.db` 가 저장소에 포함돼 있으면 `build_db`·`loader` 단계는 건너뛸 수 있습니다.

## 인터페이스 개요

프론트엔드는 단 하나의 함수만 호출합니다. 인터페이스 계약은 `contract.py` 에 고정돼 있습니다.

```python
from src.pipeline import answer_question

answer = answer_question("Can I work part-time on a D-2 visa?", lang="en")
```

### `answer_question(question, lang="en", profile=None) -> Answer`

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `question` | string | 필수 | 사용자 질문. 영어(또는 `lang`) |
| `lang` | string | 선택 | 답변 언어. 기본 `"en"` (`ko`, `zh` 지원) |
| `profile` | dict | 선택 | `{visa, program, topik, nationality, grad_date, region}` — 규정 답변을 그 학생 기준으로 맞춤화. 하위호환 |

질문은 먼저 **선배 라운지** 레이어가 캠퍼스 생활·문화·행정 팁을 가로채고(매칭 시 `local`), 아니면 **라우터**가 아래 경로 중 하나로 분류합니다. `refused` 는 라우터가 아니라 **검색 단계**에서 신뢰도가 임계값 미만일 때 결정됩니다.

| route | 언제 | 처리 |
|---|---|---|
| `local` | 캠퍼스 생활·문화·행정 꿀팁 (정부 문서로는 답할 수 없음) | 2단계 매칭 → '선배' 페르소나 답변 (할루시네이션 0). ① 키워드: 라우터 前 `docs/local_tips.json` 트리거 매칭 ② 의미: RAG 가 거부하려는 순간 임베딩으로 가장 가까운 팁 구제 |
| `sql` | 개수·순위·비교·집계 | Text-to-SQL → 표 + 막대차트 |
| `rag` | 규정·절차·자격 | 질의 한국어 번역 → 하이브리드 검색(BM25+Dense) → 출처 인용 답변 |
| `hybrid` | 통계 + 규정 동시 | SQL + RAG 동시 |
| `refused` | 근거가 임계값 미만 | **답변 생성 안 함** + 스마트 거부(가까운 공식 주제·창구 안내) |

### 신뢰 기능 — 왜 일반 AI(ChatGPT/Claude)가 아니라

비자·체류처럼 예민한 도메인에서 일반 LLM은 **학습 컷오프에 얼어붙은 지식**으로 출처 없이 자신 있게 답합니다(틀려도). 세 장치로 차별화합니다.

- **대조(contrastive) 데모** — `ungrounded_answer()`: 문서 컨텍스트 없이 LLM 에 그대로 물은 '근거 0' 답변. 데모 UI 의 `🆚 Compare` 토글로 우리 답(인용/거부)과 **나란히** 보여 grounding 의 가치를 제품이 스스로 증명합니다.
- **최신화(freshness)** — 모든 grounded 답변 끝에 근거 문서의 **수집 기준일(as-of)** 과 "규정은 바뀔 수 있으니 하이코리아 ☎1345 로 확인" 안내를 붙입니다. 일반 AI 는 '이게 최신인지' 를 구조적으로 알 수 없습니다.
- **스마트 거부(smart abstention)** — 근거가 없으면 막다른 "답 없음" 이 아니라, 가장 가까운 **공식 주제**와 **담당 창구(하이코리아·국제교류처)** 를 안내합니다.
- **개인화(personalization)** — `answer_question(q, profile=...)`: 비자·과정·TOPIK·국적·졸업예정일을 주면, 근거 문서의 **조건별 규정 중 그 학생에게 해당하는 가지**(예: 석사·TOPIK4 → 주 30시간)를 골라 답합니다. 값은 문서에서 '선택' 할 뿐 지어내지 않습니다. 일반 AI 는 당신의 개인 상황을 모릅니다. UI 의 `🧑‍🎓 My profile` 로 입력.

### 응답 스키마 (`Answer`)

| 필드 | 타입 | 설명 |
|---|---|---|
| `route` | string | `sql` \| `rag` \| `hybrid` \| `refused` \| `local` |
| `answer_text` | string | 최종 답변(영어). `refused` 면 `""` |
| `table_markdown` | string | SQL 결과 표. 없으면 `""` |
| `chart` | object | `{kind, x_label, y_label, labels, values}`. 없으면 `kind="none"` |
| `sources` | array | `{title, snippet, url, score}` 목록. 없으면 `[]` |
| `confidence` | float | 검색 신뢰도 0~1 (Dense 최대 코사인) |
| `refused_reason` | string | `refused` 일 때만 채움 |

### 예시 — 정성형 (`rag`)

요청: `answer_question("Can I work part-time on a D-2 visa? How many hours?")`

```json
{
  "route": "rag",
  "answer_text": "Yes. D-2 holders may work part-time with prior permission. If you meet the Korean-language requirement, undergraduates may work up to 25 hours per week and graduate students up to 30 ...",
  "table_markdown": "",
  "chart": { "kind": "none", "x_label": "", "y_label": "", "labels": [], "values": [] },
  "sources": [
    {
      "title": "유학(D-2) 시간제취업 허용시간 (한국어능력·학위과정별)",
      "snippet": "유학(D-2) 체류자격 소지자의 시간제취업 허용시간은 ...",
      "url": "법무부 출입국·외국인정책본부 「체류민원 자격별 안내 매뉴얼」",
      "score": 0.558
    }
  ],
  "confidence": 0.558,
  "refused_reason": ""
}
```

### 예시 — 정량형 (`sql`)

요청: `answer_question("Which universities in Seoul have the most international students?")`

```json
{
  "route": "sql",
  "answer_text": "Yonsei University hosts the most international students in Seoul (4,740), followed by Korea University (4,471) and Chung-Ang University (4,257).",
  "table_markdown": "| univ_name | univ_name_en | total |\n|---|---|---|\n| 연세대학교 | YONSEI UNIVERSITY | 4740 |\n| 고려대학교 | KOREA UNIVERSITY | 4471 |",
  "chart": {
    "kind": "bar", "x_label": "univ_name", "y_label": "total",
    "labels": ["연세대학교", "고려대학교", "중앙대학교"],
    "values": [4740, 4471, 4257]
  },
  "sources": [{ "title": "대학별 외국인 유학생 현황 · 대학 기본정보 (2025)", "snippet": "SQL: SELECT ...", "url": "https://www.data.go.kr/", "score": 1.0 }],
  "confidence": 1.0,
  "refused_reason": ""
}
```

### 예시 — 선배 라운지 (`local`)

요청: `answer_question("Should I marry a Korean to get a visa?")` — 회색지대 질문은 키워드로 잡아 위험을 경고하고 합법 경로로 안내합니다 (차갑게 거부하지 않고, 불법도 조언하지 않음).

```json
{
  "route": "local",
  "answer_text": "😅 Sorry hoobae — Sunbae is your mentor, not your wedding planner ... A 'marriage of convenience' is an actual crime in Korea. The real way to stay is D-2 → D-10 (job-seeking) → E-7 (work) ...",
  "table_markdown": "",
  "chart": { "kind": "none", "x_label": "", "y_label": "", "labels": [], "values": [] },
  "sources": [{ "title": "🎓 K-Campus Sunbae Lounge · campus-life tip (not an official regulation)", "snippet": "Should I marry a Korean to get a visa?", "url": null, "score": 1.0 }],
  "confidence": 1.0,
  "refused_reason": ""
}
```

### 예시 — 거부 (`refused`)

요청: `answer_question("How do I get Korean citizenship?")` — 코퍼스(D-2 유학생 중심)에 근거 문서가 없어 거부하고, 담당 창구를 안내합니다.

```json
{
  "route": "refused",
  "answer_text": "",
  "table_markdown": "",
  "chart": { "kind": "none", "x_label": "", "y_label": "", "labels": [], "values": [] },
  "sources": [],
  "confidence": 0.366,
  "refused_reason": "We don't have a verified official source for this (best match 0.366 < threshold 0.42), so we won't guess — a wrong visa or immigration answer can put you at real risk. For your specific situation, contact HiKorea (☎ 1345, hikorea.go.kr) or your international office."
}
```

## 프로젝트 구조

```
kcampus-navigator/
├── contract.py            # 프론트↔백엔드 인터페이스 계약 (Answer 스키마)
├── mock.py                # 프론트 개발용 목업
├── app.py                 # Streamlit 데모 UI
├── src/
│   ├── router.py          # 질문 분류: sql / rag / hybrid (LLM + 키워드 폴백)
│   ├── vector_store.py    # OpenAI 임베딩 + numpy 코사인 검색 + 한국어 질의 번역
│   ├── retriever.py       # 하이브리드 검색(BM25+Dense) + Abstention
│   ├── loader.py          # docs/*.md → 청크 → 임베딩 → vectors.npz
│   ├── build_db.py        # 공공데이터 CSV → SQLite(kcampus.db)
│   ├── sql_chain.py       # Text-to-SQL (값 한국어 용어집 + 실패 시 self-repair)
│   ├── local.py           # 선배 라운지: 로컬 생활·문화 팁 매칭 (라우터 前 실행)
│   └── pipeline.py        # 전체 조립: answer_question() 진입점
├── docs/
│   ├── local_tips.json    # 선배 라운지 큐레이션 팁 21개 (규정 아님, 생활/문화/행정)
│   └── *.md               # RAG 코퍼스: 정부 규정 문서 46개 + 발표 자료
├── data/
│   ├── raw/               # 원본 공공데이터 CSV
│   └── processed/         # vectors.npz(검색 인덱스), kcampus.db(SQLite)
├── notebooks/eda.ipynb    # 7 Steps EDA (결측 MAR·국적 다양성)
└── eval/                  # 평가셋 30문항 + 재보정 하니스(run_eval.py)
```

## 검증·재보정 명령

```bash
python src/pipeline.py                     # 경로별 스모크 테스트
python src/local.py                        # 선배 라운지 키워드 스모크 (무API)
python src/local.py --semantic             # 선배 라운지 의미 매칭 스모크 (API 필요)
python src/router.py eval/questions.csv    # 라우터 분류 정확도 (29/30)
python eval/run_eval.py                    # 검색 재보정 (브릿지·전략·임계값 스윕)
```

## 데이터 출처

모든 규정 문서는 한국 정부 공식 출처입니다 — 하이코리아(hikorea.go.kr), 법무부 출입국·외국인정책본부 자격별 안내매뉴얼, Study in Korea(국립국제교육원), 국민건강보험공단. 대학 통계는 공공데이터(data.go.kr / 대학알리미 계열, 2025)를 사용합니다.
