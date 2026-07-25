# kcampus-navigator

**English** · [한국어](./README.md)

A decision-support system for international students considering study in Korea. **Ask in English**: numeric questions are answered by **SQL** over public datasets, rules questions by **RAG** over government documents — always **with a source**. Aggregation, ranking and comparison, which vector search structurally cannot do, are routed to SQL. And when no evidence clears the confidence threshold, the system **refuses instead of generating an answer**, because visa and immigration rules are a domain where one wrong answer can put a person at real risk.

A 2026 BIGDATA-USC Conference Hackathon project by Team 12 `SELECT * FROM Korea`; presented as **Kcampus Navigator**.

## What makes it different

A generic LLM answers confidently from knowledge frozen at a training cutoff, with no source. Four things separate this from that.

- **Abstention** — if the best evidence is below the threshold (0.42), no sentence is generated; the answer names the closest official topic and the immigration hotline (☎1345) instead.
- **As-of dates** — every grounded answer carries the date its source document was collected. A generic model cannot tell you whether its own knowledge is current.
- **Personalization** — given a profile, the model picks the **branch of the regulation that applies to that student**. Values are *selected* from the document, never invented.

  | Profile | `How many hours can I work part-time on a D-2 visa?` |
  |---|---|
  | none | **every branch** — 25h undergrad / 30h graduate / 30–35h with excellent grades / 10–15h below the language requirement |
  | University of Seoul · Master's · TOPIK 4 | **"30 hours per week"** — one line |

- **Two curated layers** — for the things no public dataset holds: **scholarships** (each university posts its own notice) and **campus life** (no government document covers it). These are hand-written, not generated, so they cannot hallucinate — and for a school we don't cover the system **invents nothing**, it falls through to the government documents.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate     # macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                               # fill in OPENAI_API_KEY

streamlit run app.py                               # demo UI on :8501
```

> Prebuilt `vectors.npz` and `kcampus.db` are committed, so you can skip the build steps.
> To rebuild: `python src/build_db.py && python src/loader.py`.

Key `.env` values: `OPENAI_API_KEY` · `LLM_MODEL=gpt-4o-mini` · `EMBEDDING_MODEL=text-embedding-3-small` · `CONFIDENCE_THRESHOLD=0.42` (below this it refuses) · `FUSION=score`

**Stack** — Python 3.12 · OpenAI API (LLM + embeddings) · numpy vector store (no chromadb/faiss) · rank-bm25 + kiwipiepy (Korean morphological BM25) · SQLite · Streamlit · pandas/matplotlib for EDA

## Interface

The frontend calls exactly one function. The contract is frozen in `contract.py`.

```python
from src.pipeline import answer_question

answer_question("Can I work part-time on a D-2 visa?", lang="en", profile=None)
```

| Parameter | Description |
|---|---|
| `question` | User question, in English (or `lang`) |
| `lang` | Answer language. Default `"en"` (`ko`, `zh` supported) |
| `profile` | Optional · any subset of `{visa, program, school, major, topik, nationality, grad_date, region}`. Backward compatible |

Each question first passes **two curated layers** (scholarships → Sunbae Lounge); if neither matches, the **router** classifies it. `refused` is decided not by the router but at the **retrieval stage**, when confidence falls below the threshold.

| route | When | How |
|---|---|---|
| `rag` (My School) | Scholarship question + a curated school in the profile | Checks `docs/scholarships.json` against the student's degree level and TOPIK → "eligible now / one condition short / depends on your GPA", with the school's notice linked and the curation date stamped. Unknown school → falls through |
| `local` | Campus-life / culture / bureaucracy tips | `docs/local_tips.json` keyword match before the router, plus a semantic rescue at the moment RAG would refuse |
| `sql` | Counts, rankings, comparisons | Text-to-SQL → table + bar chart |
| `rag` | Rules, procedures, eligibility | Query translated to Korean → hybrid retrieval (BM25 + dense) → cited answer |
| `hybrid` | A statistic *and* a rule | SQL + RAG |
| `refused` | Evidence below threshold | **No answer generated** + smart abstention (closest official topic, where to ask) |

### Response schema (`Answer`)

| Field | Description |
|---|---|
| `route` | `sql` \| `rag` \| `hybrid` \| `refused` \| `local` |
| `answer_text` | Final answer (English). `""` when refused |
| `table_markdown` | SQL result table, or `""` |
| `chart` | `{kind, x_label, y_label, labels, values}`; `kind="none"` when absent |
| `sources` | List of `{title, snippet, url, score}` |
| `confidence` | Retrieval confidence 0–1 (max dense cosine) |
| `refused_reason` | Filled only when `refused` |

<details>
<summary>Three response examples (sql · refused · My School scholarships)</summary>

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

> Student counts are total registered international students (including language and exchange) — the same basis as the national figure of 197,163.

**`refused`** — `answer_question("How do I get Korean citizenship?")`

```json
{
  "route": "refused", "answer_text": "", "confidence": 0.366,
  "refused_reason": "We don't have a verified official source for this (best match 0.366 < threshold 0.42), so we won't guess — a wrong visa or immigration answer can put you at real risk. For your specific situation, contact HiKorea (☎ 1345, hikorea.go.kr) ..."
}
```

**My School scholarships** — `profile={"school": "University of Seoul (서울시립대학교)", "program": "Master's", "topik": "4"}`

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

> With TOPIK 2 the same entry flips to `⚠️ needs TOPIK 4 — you have TOPIK 2`. The verdict is a comparison between the profile and the conditions in the JSON (`levels` · `topik_min` · `gpa_min`) — no LLM writes these sentences.

</details>

## Structure

```
kcampus-navigator/
├── contract.py            # frontend↔backend contract (Answer schema)
├── app.py                 # Streamlit demo UI (compare view · persona · Sunbae card)
├── src/
│   ├── pipeline.py        # full assembly: answer_question() entry point
│   ├── router.py          # sql / rag / hybrid classification (LLM + keyword fallback)
│   ├── retriever.py       # hybrid retrieval (BM25 + dense) + abstention
│   ├── vector_store.py    # OpenAI embeddings + numpy cosine + Korean query bridge
│   ├── sql_chain.py       # Text-to-SQL (Korean value glossary + self-repair)
│   ├── scholarships.py    # My School: per-school scholarship personalization
│   ├── local.py           # Sunbae Lounge: campus-life tip matching
│   ├── loader.py          # docs/*.md → chunks → embeddings → vectors.npz
│   └── build_db.py        # public-data CSV → SQLite
├── docs/                  # 46 regulation docs + local_tips.json + scholarships.json + deck graphics (svg/png)
├── data/                  # raw CSV · processed (vectors.npz, kcampus.db)
├── notebooks/eda.ipynb    # 7-step EDA (MAR missingness · nationality mix)
└── eval/                  # 30-question eval set · recalibration harness · demo pre-flight
```

## Verification commands

```bash
python eval/demo_check.py                  # demo pre-flight: all 7 demo scenes route as scripted (--dry = no API)
python src/router.py eval/questions.csv    # router classification accuracy (29/30)
python eval/run_eval.py                    # retrieval recalibration (bridge · strategy · threshold sweep)
python eval/sql_eval.py                    # SQL answer accuracy (--dry = schema guard, no API)
python src/scholarships.py                 # scholarship branching smoke (no API)
python src/local.py                        # Sunbae Lounge keyword smoke (no API)
```

**Our own measurement (2026-07-24 corpus, raw fractions)** — router 29/30 · Recall@3 13/16 · abstention 21/22 @0.42 · EN→KO bridge 0.45→0.54 · SQL answers 5/5

> The SQL score is not self-graded: for each question the expected answer is computed independently in pandas from the raw CSVs and compared (`eval/sql_eval.py`). The denominator is 5, and we say so.

## Data sources

All 46 regulation documents come from official Korean government sources — HiKorea, the Ministry of Justice Korea Immigration Service manuals, Study in Korea (NIIED), and the National Health Insurance Service. University statistics use public data (data.go.kr / Academyinfo, 2025). Per-school scholarships are hand-curated from each university's own notice (`docs/scholarships.json`, two schools: University of Seoul and Sookmyung); amounts and tiers are set per semester by the school, so every answer carries the curation date and a pointer to the international office.
