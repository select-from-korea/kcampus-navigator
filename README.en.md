# kcampus-navigator

**English** · [한국어](./README.md)

A decision-support system for international students considering study in Korea. **Ask in English**: numeric questions are answered by **SQL** over public datasets, and rules questions are answered by **RAG** over government documents — always **with a source**. Aggregation, ranking, and comparison — which vector search structurally cannot do — are routed to SQL. And when no evidence clears the confidence threshold, the system **refuses instead of generating an answer (abstention)**, because visa and immigration rules are a domain where one wrong answer can put a person at real risk. A 2026 BIGDATA-USC Conference Hackathon project by team `SELECT * FROM Korea`.

## Tech stack

- Python 3.12 (3.13/3.14 not supported)
- OpenAI API — LLM `gpt-4o-mini`, embeddings `text-embedding-3-small`
- numpy — vector store (cosine similarity via dot product after L2 normalization; no chromadb/faiss)
- rank-bm25 + kiwipiepy — Korean morpheme-based BM25 lexical search
- SQLite — Text-to-SQL target for university statistics (`kcampus.db`)
- pandas, matplotlib — data processing / EDA
- pdfplumber — original PDF loading (optional)
- Streamlit — demo UI

## Getting started

Create a `.env` file with the following (copy `.env.example`):

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CONFIDENCE_THRESHOLD=0.42     # below this, the answer is refused
FUSION=score                  # retrieval fusion (score | rrf)
W_DENSE=1.0
W_BM25=1.0
LOCAL_SEM_THRESHOLD=0.50      # Sunbae Lounge semantic-rescue threshold
```

Install dependencies, build the index and DB, then run:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS: source .venv/bin/activate)
pip install -r requirements.txt

python src/build_db.py            # data/raw CSV → data/processed/kcampus.db (SQLite)
python src/loader.py              # docs/*.md → data/processed/vectors.npz (search index)

streamlit run app.py              # demo UI (default port 8501)
```

> `app.py` imports the real pipeline: `from src.pipeline import answer_question, baseline_answer, ungrounded_answer`. To preview the UI without a backend, swap the first import for `from mock import answer_question` (same signature).
> If the built `vectors.npz` and `kcampus.db` are committed, you can skip the `build_db` / `loader` steps.

## Interface overview

The frontend calls a single function. The interface contract is fixed in `contract.py`.

```python
from src.pipeline import answer_question

answer = answer_question("Can I work part-time on a D-2 visa?", lang="en")
```

### `answer_question(question, lang="en", profile=None) -> Answer`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `question` | string | yes | User question, in English (or `lang`) |
| `lang` | string | no | Answer language. Default `"en"` (`ko`, `zh` supported) |
| `profile` | dict | no | Optional `{visa, program, topik, nationality, grad_date, region}` — personalizes rules answers to this student. Backward compatible. |

Each question first passes through the **Sunbae Lounge** layer, which intercepts campus-life / culture / bureaucracy questions (route `local`); otherwise the **router** classifies it into one of the routes below. `refused` is decided not by the router but at the **retrieval stage**, when confidence falls below the threshold.

| route | When | How |
|---|---|---|
| `local` | Campus-life / culture / bureaucracy tips (things no government doc covers) | 2-tier match → "Sunbae" (senior) persona answer (zero hallucination). ① keyword: `docs/local_tips.json` triggers, before the router ② semantic: at the moment RAG would refuse, embeddings rescue the closest tip |
| `sql` | Counts, rankings, comparisons, aggregation | Text-to-SQL → table + bar chart |
| `rag` | Rules, procedures, eligibility | Query → Korean translation → hybrid search (BM25 + Dense) → cited answer |
| `hybrid` | Statistics + rules together | SQL + RAG together |
| `refused` | Evidence below threshold | **No answer is generated** + smart refusal (nearest official topic + where to ask) |

### Trust features — why not just a generic AI (ChatGPT / Claude)

In a sensitive domain like visas, a generic LLM answers confidently from **knowledge frozen at its training cutoff**, with no source — even when it's wrong. Four mechanisms set this system apart.

- **Contrastive demo** — `ungrounded_answer()`: the LLM asked the raw question with **no document context** (a "no-source" answer). The `🆚 Compare` toggle shows it **side by side** with our answer (cited or refused), so the product proves the value of grounding itself. (Measured: "How many hours on a D-2?" → generic AI says "20 hours" [wrong] vs ours "25/30 hours" [Ministry of Justice manual].)
- **Freshness** — every grounded answer ends with the source's **as-of date** and a note to "confirm the current rule at HiKorea ☎1345, because rules change." A generic AI structurally cannot tell you whether its answer is current.
- **Smart abstention** — when there's no evidence, instead of a dead-end "no answer" it points to the closest **official topic** and the **right office** (HiKorea / your international office).
- **Personalization** — `answer_question(q, profile=...)`: given visa, degree program, TOPIK level, nationality, or graduation date, it selects the **branch of the rule that applies to this student** (e.g. Master's + TOPIK 4 → 30 hours/week) from the source. It *selects* values from the document; it never invents them. A generic AI doesn't know your situation. Enter it via `🧑‍🎓 My profile` in the UI.

### Response schema (`Answer`)

| Field | Type | Description |
|---|---|---|
| `route` | string | `sql` \| `rag` \| `hybrid` \| `refused` \| `local` |
| `answer_text` | string | Final answer (English). `""` when `refused` |
| `table_markdown` | string | SQL result table. `""` if none |
| `chart` | object | `{kind, x_label, y_label, labels, values}`. `kind="none"` if none |
| `sources` | array | List of `{title, snippet, url, score}`. `[]` if none |
| `confidence` | float | Retrieval confidence 0–1 (max Dense cosine) |
| `refused_reason` | string | Filled only when `refused` |

### Example — rules (`rag`)

Request: `answer_question("Can I work part-time on a D-2 visa? How many hours?")`

```json
{
  "route": "rag",
  "answer_text": "Yes. D-2 holders may work part-time with prior permission. If you meet the Korean-language requirement, undergraduates may work up to 25 hours per week and graduate students up to 30 ...\n\n🗓 Grounded in official sources as of 2026-07-24. Immigration rules can change — confirm the current rule at HiKorea (☎ 1345, hikorea.go.kr) ...",
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

### Example — statistics (`sql`)

Request: `answer_question("Which universities in Seoul have the most international students?")`

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

### Example — Sunbae Lounge (`local`)

Request: `answer_question("Should I marry a Korean to get a visa?")` — a grayzone question is caught by keyword, warned, and redirected to the legal path (never refused coldly, never given illegal advice).

```json
{
  "route": "local",
  "answer_text": "😅 Sorry hoobae ('junior') — Sunbae is your mentor, not your wedding planner ... A 'marriage of convenience' is an actual crime in Korea. The real way to stay long-term is D-2 → D-10 (job-seeking) → E-7 (work) ...",
  "table_markdown": "",
  "chart": { "kind": "none", "x_label": "", "y_label": "", "labels": [], "values": [] },
  "sources": [{ "title": "🎓 K-Campus Sunbae Lounge · campus-life tip (not an official regulation)", "snippet": "Should I marry a Korean to get a visa?", "url": null, "score": 1.0 }],
  "confidence": 1.0,
  "refused_reason": ""
}
```

### Example — refusal (`refused`)

Request: `answer_question("How do I get Korean citizenship?")` — no document in the corpus (which is D-2 student-focused) clears the threshold, so it refuses and points you to the right channel.

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

## Project structure

```
kcampus-navigator/
├── contract.py            # frontend↔backend interface contract (Answer schema)
├── mock.py                # mock for frontend dev (swap for src.pipeline on the day)
├── app.py                 # Streamlit demo UI (Compare view + My profile + Sunbae card)
├── src/
│   ├── router.py          # question classification: sql / rag / hybrid (LLM + keyword fallback)
│   ├── vector_store.py    # OpenAI embeddings + numpy cosine search + Korean query translation
│   ├── retriever.py       # hybrid search (BM25+Dense) + abstention
│   ├── loader.py          # docs/*.md → chunks → embeddings → vectors.npz
│   ├── build_db.py        # public-data CSV → SQLite (kcampus.db)
│   ├── sql_chain.py       # Text-to-SQL (Korean value glossary + self-repair on failure)
│   ├── local.py           # Sunbae Lounge: local life/culture tip matching (runs before router)
│   └── pipeline.py        # full assembly: answer_question() entry point
├── docs/
│   ├── local_tips.json    # 21 curated Sunbae Lounge tips (not regulations; life/culture/bureaucracy)
│   └── *.md               # RAG corpus: 46 government regulation docs + presentation material
├── data/
│   ├── raw/               # original public-data CSV
│   └── processed/         # vectors.npz (search index), kcampus.db (SQLite)
├── notebooks/eda.ipynb    # 7-step EDA (missingness MAR · nationality diversity)
└── eval/                  # 30-question eval set + recalibration harness (run_eval.py)
```

## Validation / recalibration commands

```bash
python src/pipeline.py                     # smoke test across routes
python src/local.py                        # Sunbae Lounge keyword smoke (no API)
python src/local.py --semantic             # Sunbae Lounge semantic smoke (needs API)
python src/router.py eval/questions.csv    # router classification accuracy (29/30)
python eval/run_eval.py                    # retrieval recalibration (bridge · strategy · threshold sweep)
```

## Data sources

All regulation documents come from official Korean government sources — HiKorea (hikorea.go.kr), the Ministry of Justice Korea Immigration Service "residence/visa manuals by status", Study in Korea (National Institute for International Education), and the National Health Insurance Service. University statistics use public data (data.go.kr / Academyinfo lineage, 2025).
