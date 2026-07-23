# Problem Definition — SELECT * FROM Korea

> 2026 BIGDATA-USC Conference Hackathon · Team `SELECT * FROM Korea`
> One-page problem statement. Full rationale lives in the team's project notes.

## One line

International students deciding **where** to study in Korea have no way to query
Korean public data — it is scattered across 20+ government sites, written only in
Korean, and impossible to aggregate or compare. We turn that data into answers.

## Who

**Inbound** international students considering study in Korea (not Korean students
going abroad). This is deliberate: Korea's public datasets are *about* inbound
students, and the language barrier means competing teams and local students
literally cannot read the source data. That is our built-in differentiation.

## The problem

- **197,163** international students across **224** universities (2025 public data).
- The information they need exists — the government and universities publish it —
  but it is **fragmented, Korean-only, and non-comparable**.
- Two kinds of questions go unanswered today:

| Question type | Example | Why search fails |
|---|---|---|
| Quantitative | "Which universities in Seoul have the most international students?" | Vector search retrieves passages; it cannot **count, rank, or aggregate**. |
| Regulatory | "Can I work part-time on a D-2 visa? How many hours?" | Rules are buried in Korean government PDFs; a wrong answer can get someone deported. |

So students choose based on a friend's advice or a YouTube video.

## Why it is hard (and interesting)

1. **Aggregation ≠ retrieval.** A pure RAG chatbot structurally cannot answer
   "the top 3 universities by X." We show this failure live, then route quantitative
   questions to SQL and regulatory questions to document search.
2. **Cross-lingual gap.** Questions are English, documents are Korean. We measured a
   ~23% drop in retrieval similarity from the language gap alone, so we translate each
   query into a Korean keyword phrase before searching.
3. **Wrong answers are dangerous.** Visa/admission rules must not be hallucinated. If
   no source clears our confidence threshold, the system **refuses** and returns the
   source link instead of generating text.

## What we built (data foundation)

- **Regulatory corpus:** 46 single-topic documents, all from Korean government sources
  (HiKorea, Ministry of Justice residence/visa manuals, Study in Korea, NHIS), covering
  visas, stay, part-time work, D-10/E-7 transition, registration, insurance, scholarships.
- **Statistics:** `kcampus.db` — 224 universities × student counts by nationality, visa
  status, gender, plus field, language, and dormitory attributes.
- **Honest inputs (EDA).** Dorm-fee data is missing **not at random** — the missing rate
  falls from 49% at the smallest universities to 5% at the largest (MAR); naively dropping
  those rows would bias any cost comparison toward large schools. We also derived a
  nationality-diversity index (KAIST draws from 85 countries; some large schools are 93%
  a single nationality) that the raw data does not contain.

## Judging-criteria fit

| Criterion | Our angle |
|---|---|
| Idea / problem definition | Inbound reframing + data no one else can read |
| Usefulness | Answers real decisions: which university, which region, what rules |
| Technical difficulty | Router + Text-to-SQL + hybrid retrieval + cross-lingual bridge |
| Polish | Quantitative evaluation (router 29/30 on a 30-question set) |
| Presentation | Charts + live English demo, including a deliberate refusal |

## Scope (what we are *not* building)

No accounts, no persistence, no chat history, no deployment. Three people, ~8 build
hours. The value is the routing + honest data, not the UI.
