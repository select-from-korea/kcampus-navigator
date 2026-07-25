# kcampus-navigator

[English](./README.en.md) · **한국어**

한국 유학을 고려하는 외국인 학생이 **영어로 질문하면**, 숫자 질문은 **SQL**로 공공데이터를 조회하고 규정 질문은 **RAG**로 정부 문서를 검색해 **출처와 함께** 답하는 의사결정 지원 시스템입니다. 벡터 검색이 구조적으로 할 수 없는 집계·순위·비교는 라우터가 SQL 경로로 보내 처리하며, 근거가 임계값에 못 미치면 **답변을 생성하지 않고 거부** 합니다. 비자·체류 규정은 틀린 답 하나가 사람을 위험에 빠뜨릴 수 있는 도메인이기 때문입니다. Team 12 `SELECT * FROM Korea` 의 2026 BIGDATA-USC Conference Hackathon 프로젝트이며, 발표 제품명은 **Kcampus Navigator** 입니다.

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

> 빌드된 `vectors.npz`·`kcampus.db` 가 저장소에 포함돼 있어 `build_db`·`loader` 단계는 생략할 수 있습니다.

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
| `profile` | dict | 선택 | `{visa, program, school, major, topik, nationality, grad_date, region}` — 규정·장학금 답변을 그 학생 기준으로 맞춤화. 하위호환 |

질문은 먼저 **큐레이션 레이어 두 개**(장학금 → 선배 라운지)가 공공데이터로는 답할 수 없는 질문을 가로채고, 아니면 **라우터**가 아래 경로 중 하나로 분류합니다. `refused` 는 라우터가 아니라 **검색 단계**에서 신뢰도가 임계값 미만일 때 결정됩니다.

| route | 언제 | 처리 |
|---|---|---|
| `rag` (My School) | 장학금 질문 + 프로필에 큐레이션된 학교 | `docs/scholarships.json` 에서 그 학교 장학금을 뽑아 **과정·TOPIK 요건과 대조** → '지금 해당 / 요건 미달 / 정보 부족' 으로 분류해 제시. LLM 생성 없음(할루시네이션 0), 학교 공지 링크·기준일 첨부. 모르는 학교면 통과시켜 정부 문서(GKS)로 |
| `local` | 캠퍼스 생활·문화·행정 꿀팁 (정부 문서로는 답할 수 없음) | 2단계 매칭 → '선배' 페르소나 답변 (할루시네이션 0). ① 키워드: 라우터 前 `docs/local_tips.json` 트리거 매칭 ② 의미: RAG 가 거부하려는 순간 임베딩으로 가장 가까운 팁 구제 |
| `sql` | 개수·순위·비교·집계 | Text-to-SQL → 표 + 막대차트 |
| `rag` | 규정·절차·자격 | 질의 한국어 번역 → 하이브리드 검색(BM25+Dense) → 출처 인용 답변 |
| `hybrid` | 통계 + 규정 동시 | SQL + RAG 동시 |
| `refused` | 근거가 임계값 미만 | **답변 생성 안 함** + 스마트 거부(가까운 공식 주제·창구 안내) |

### 신뢰 기능 — 왜 일반 AI(ChatGPT/Claude)가 아니라

비자·체류처럼 예민한 도메인에서 일반 LLM은 **학습 컷오프에 얼어붙은 지식**으로 출처 없이 자신 있게 답합니다(틀려도). 세 장치로 차별화합니다.

- **대조(contrastive) 데모** — `ungrounded_answer()`: 문서 컨텍스트 없이 LLM 에 그대로 물은 '근거 0' 답변. 데모 UI 의 `🆚 Compare` 토글로 우리 답(인용/거부)과 **나란히** 보여 grounding 의 가치를 제품이 스스로 증명합니다. (라우팅·거부 없는 순수 벡터 검색 대조군 `baseline_answer()` 도 코드에 남아 있으나, 5분 발표에 들어가지 않아 데모 UI 에서는 뺐습니다.)
- **최신화(freshness)** — 모든 grounded 답변 끝에 근거 문서의 **수집 기준일(as-of)** 과 "규정은 바뀔 수 있으니 하이코리아 ☎1345 로 확인" 안내를 붙입니다. 일반 AI 는 '이게 최신인지' 를 구조적으로 알 수 없습니다.
- **스마트 거부(smart abstention)** — 근거가 없으면 막다른 "답 없음" 이 아니라, 가장 가까운 **공식 주제**와 **담당 창구(하이코리아·국제교류처)** 를 안내합니다.
- **개인화(personalization)** — `answer_question(q, profile=...)`: 비자·과정·학교·TOPIK·국적·졸업예정일을 주면, 근거 문서의 **조건별 규정 중 그 학생에게 해당하는 가지**를 골라 답합니다. 값은 문서에서 '선택' 할 뿐 지어내지 않습니다. UI 의 `🧑‍🎓 My profile` 로 입력하거나 **데모 페르소나 드롭다운**으로 한 번에 전환합니다.

  같은 질문 `How many hours can I work part-time on a D-2 visa?` 이 프로필 유무로 갈립니다 (실측):

  | 프로필 | 답 |
  |---|---|
  | 없음 | 규정의 **모든 가지** 나열 — 학부 25h / 석·박사 30h / 우수자 30·35h / 어학요건 미달 10·15h. 맞지만 자기 답이 뭔지는 알 수 없습니다 |
  | **Roger** (서울시립대 · 석사 · TOPIK 4) | **"주 30시간"** 한 줄 — *그에게* 해당하는 가지만 |

  같은 질문, 같은 문서입니다. 달라진 건 프로필뿐이고, 값은 문서에서 고를 뿐 지어내지 않습니다.
- **My School 장학금** — 장학금은 정부 규정 코퍼스에도 공공데이터 통계에도 없습니다(대학이 각자 홈페이지 공지로만 냅니다). 그래서 학교별 구조화 데이터를 직접 만들었습니다(`docs/scholarships.json`). 프로필의 학교·과정·TOPIK 과 대조해 **"지금 지원 가능 / TOPIK 4급 필요(현재 2급) / 성적에 따라 달라짐"** 으로 갈라 보여주고, 커버하지 않는 학교는 **지어내지 않고** 국가장학금(GKS) 문서 경로로 넘깁니다. 현재 커버리지는 **서울시립대·숙명여대 2교** (확장은 JSON 항목 추가만, 코드 수정 없음).

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
  "answer_text": "The universities in Seoul with the most international students are Hanyang University with 7,437 students, Chung-Ang University with 4,864 students, and Kyung Hee University with 4,858 students.",
  "table_markdown": "| univ_name | univ_name_en | students |\n|---|---|---|\n| 한양대학교 | HANYANG UNIVERSITY | 7437 |\n| 중앙대학교 | CHUNG-ANG UNIVERSITY | 4864 |\n| 경희대학교 | KYUNG HEE UNIVERSITY | 4858 |",
  "chart": {
    "kind": "bar", "x_label": "univ_name", "y_label": "students",
    "labels": ["한양대학교", "중앙대학교", "경희대학교"],
    "values": [7437, 4864, 4858]
  },
  "sources": [{ "title": "대학별 외국인 유학생 현황 · 대학 기본정보 (2025)", "snippet": "SQL: SELECT ...", "url": "https://www.data.go.kr/", "score": 1.0 }],
  "confidence": 1.0,
  "refused_reason": ""
}
```

> 유학생 수 = 등록 유학생 전체(어학·교환 포함) headcount 기준 — 전국 197,163명과 동일 기준입니다.

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

### 예시 — My School 장학금 (`rag`, 큐레이션)

요청: `answer_question("What scholarships can I get at my school?", profile={"school": "University of Seoul (서울시립대학교)", "program": "Master's", "topik": "4"})`

```
### Scholarships at University of Seoul (서울시립대학교)
Matched to your profile — Master's · TOPIK 4.

**You are eligible to apply for:**
- International Student Admission Scholarship (외국인 신입생 입학장학금) — 30–100% of tuition*
- Korean Proficiency (TOPIK) Scholarship (한국어능력 우수 장학금) — partial tuition*
  ✅ you hold TOPIK 4 ≥ 4
- Graduate Research / Teaching Assistantship (RA·TA) — stipend + tuition support*

**Depends on something we don't know about you:**
- Academic Excellence Scholarship (성적우수 장학금) — partial tuition*
  ℹ️ depends on your GPA — needs 3.5+ last semester, which your profile doesn't include

🗓 Curated from the university's own scholarship notice, as of 2026-07-25 ...
```

> TOPIK 2급 프로필로 같은 질문을 하면 TOPIK 장학금이 `⚠️ needs TOPIK 4 — you have TOPIK 2` 로 바뀝니다. 판정은 `docs/scholarships.json` 의 조건(`levels`·`topik_min`·`gpa_min`)과 프로필을 비교할 뿐 — LLM 이 문장을 만들지 않습니다.

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
├── app.py                 # Streamlit 데모 UI (대조 뷰 · 페르소나 전환 · 선배 라운지 카드)
├── src/
│   ├── router.py          # 질문 분류: sql / rag / hybrid (LLM + 키워드 폴백)
│   ├── vector_store.py    # OpenAI 임베딩 + numpy 코사인 검색 + 한국어 질의 번역
│   ├── retriever.py       # 하이브리드 검색(BM25+Dense) + Abstention
│   ├── loader.py          # docs/*.md → 청크 → 임베딩 → vectors.npz
│   ├── build_db.py        # 공공데이터 CSV → SQLite(kcampus.db)
│   ├── sql_chain.py       # Text-to-SQL (값 한국어 용어집 + 실패 시 self-repair)
│   ├── local.py           # 선배 라운지: 로컬 생활·문화 팁 매칭 (라우터 前 실행)
│   ├── scholarships.py    # My School: 학교별 장학금 개인화 (라우터 前 실행)
│   └── pipeline.py        # 전체 조립: answer_question() 진입점
├── docs/
│   ├── local_tips.json    # 선배 라운지 큐레이션 팁 21개 (규정 아님, 생활/문화/행정)
│   ├── scholarships.json  # 학교별 장학금 큐레이션 (서울시립대·숙명여대)
│   ├── 05_architecture_diagram.svg/.png   # 발표 덱 "How It Works" 교체용 (Slides 삽입은 PNG)
│   └── *.md               # RAG 코퍼스: 정부 규정 문서 46개 + 발표 자료
├── data/
│   ├── raw/               # 원본 공공데이터 CSV
│   └── processed/         # vectors.npz(검색 인덱스), kcampus.db(SQLite)
├── notebooks/eda.ipynb    # 7 Steps EDA (결측 MAR·국적 다양성)
├── .streamlit/config.toml # 데모 UI 테마
└── eval/                  # 평가셋 30문항 + 재보정 하니스(run_eval.py) + SQL 정확도(sql_eval.py)
```

## 검증·재보정 명령

```bash
python src/pipeline.py                     # 경로별 스모크 테스트
python src/local.py                        # 선배 라운지 키워드 스모크 (무API)
python src/local.py --semantic             # 선배 라운지 의미 매칭 스모크 (API 필요)
python src/scholarships.py                 # My School 장학금 분기 스모크 (무API)
python src/router.py eval/questions.csv    # 라우터 분류 정확도 (29/30)
python eval/run_eval.py                    # 검색 재보정 (브릿지·전략·임계값 스윕)
python eval/sql_eval.py                     # SQL 정답 정확도 (--dry 는 무API 스키마 가드)
python eval/demo_check.py                  # 발표 프리플라이트: 데모 7장면 route 검증 (--dry 는 무API)
```

## 데이터 출처

모든 규정 문서는 한국 정부 공식 출처입니다 — 하이코리아(hikorea.go.kr), 법무부 출입국·외국인정책본부 자격별 안내매뉴얼, Study in Korea(국립국제교육원), 국민건강보험공단. 대학 통계는 공공데이터(data.go.kr / 대학알리미 계열, 2025)를 사용합니다. 학교별 장학금은 각 대학 장학 공지에서 직접 큐레이션했으며(`docs/scholarships.json`), 금액·구간은 학기마다 대학이 정하므로 답변에 큐레이션 기준일과 국제교류처 확인 안내를 함께 노출합니다.
