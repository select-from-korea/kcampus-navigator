"""
sql_eval.py — Text-to-SQL 정확도 하니스 (정의 B: students.headcount 전 visa 합)
  독립 오라클(pandas) vs 공개 계약 answer_question().
  정의 B = 시스템과 동일 기준(어학/교환 포함, 전국 197,163 기준).
실행:
  python eval/sql_eval.py            # 전체 (OPENAI_API_KEY + kcampus.db 필요)
  python eval/sql_eval.py --dry      # 오라클 + 스키마 가드만 (무API)
"""
from __future__ import annotations
import re, sqlite3, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
DB = BASE / "data" / "processed" / "kcampus.db"

def _load():
    u = pd.read_csv(RAW / "universities.csv", encoding="utf-8-sig")
    u.columns = [c.strip() for c in u.columns]
    u["univ_key"] = u["univ_key"].astype(str).str.strip()
    s = pd.read_csv(RAW / "students_by_nationality.csv", encoding="utf-8-sig")
    s["univ_key"] = s["univ_key"].astype(str).str.strip()
    return u, s

U, S = _load()
REG = U.drop_duplicates("univ_key").set_index("univ_key")["region"]
S["region"] = S["univ_key"].map(REG)
NAME = U.drop_duplicates("univ_key").set_index("univ_key")

def _named(keys):
    nm = NAME.reindex(keys)
    return nm["univ_name"].tolist() + nm["univ_name_en"].tolist() + list(keys)

# ── 오라클 (정의 B: headcount 합) ──────────────────────────────
def ora_seoul_top():
    seoul = REG[REG == "서울"].index
    d = S[S.univ_key.isin(seoul)].groupby("univ_key").headcount.sum().nlargest(3)
    return {"names": _named(d.index.tolist()), "values": [int(v) for v in d.tolist()]}

def ora_chinese_top():
    d = S[S.nationality == "중국"].groupby("univ_key").headcount.sum().nlargest(3)
    return {"names": _named(d.index.tolist()), "values": [int(v) for v in d.tolist()]}

def ora_national_top():
    d = S.groupby("univ_key").headcount.sum().nlargest(3)
    return {"names": _named(d.index.tolist()), "values": [int(v) for v in d.tolist()]}

def ora_vietnam_count():
    return {"names": [], "values": [int(S[S.nationality == "베트남"].headcount.sum())]}

def ora_seoul_busan():
    d = S[S.region.isin(["서울", "부산"])].groupby("region").headcount.sum()
    return {"names": ["Seoul", "Busan", "서울", "부산"],
            "values": [int(d.get("서울", 0)), int(d.get("부산", 0))]}

# (질문, 오라클, 기대경로, 매칭모드)  mode: name | value | direction
GOLD = [
    ("Which universities in Seoul have the most international students?", ora_seoul_top, "sql", "name"),
    ("Which universities have the most Chinese students?", ora_chinese_top, "sql", "name"),
    ("Compare the number of international students in Seoul vs Busan.", ora_seoul_busan, "sql", "direction"),
    ("How many Vietnamese students study in Korea?", ora_vietnam_count, "sql", "value"),
    ("List the top 3 universities by international student count.", ora_national_top, "sql", "name"),
]

# ── 스키마 가드 ───────────────────────────────────────────────
ADVERTISED = {"dropout": ["dropout", "중도탈락"], "scholarship": ["scholarship", "장학"],
              "dormitory_capacity": ["dorm_capacity"], "region": ["region"], "nationality": ["nationality"]}
def schema_guard():
    print("\n── 스키마 가드 (DB 실제 컬럼) " + "─" * 34)
    if not DB.exists():
        print(f"  ⚠️ DB 없음: {DB}  → `python src/build_db.py`"); return
    con = sqlite3.connect(DB); cols = set()
    for t in ("universities", "students"):
        try: cols |= {r[1].lower() for r in con.execute(f"PRAGMA table_info({t})")}
        except sqlite3.OperationalError: pass
    con.close()
    for c, needles in ADVERTISED.items():
        ok = any(any(n.lower() in col for col in cols) for n in needles)
        print(f"  {'OK ' if ok else '❌ 없음':6s} {c}")
    print("  (dropout/scholarship ❌ 는 정상 — 컬럼 없음. router 프롬프트에서만 제거)")

# ── 매칭 ─────────────────────────────────────────────────────
def _hay(r):
    ch = r.get("chart", {}) or {}
    return " ".join([r.get("answer_text") or "", r.get("table_markdown") or ""]
                    + [str(x) for x in ch.get("labels", [])])
def _nums(t): return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", t or "")}
def _match(r, o, mode):
    hay = _hay(r); low = hay.lower()
    if mode == "name":
        for n in o["names"]:
            if not n: continue
            nl = n.lower()
            if nl in low: return True
            tok = nl.split()[0]
            if len(tok) >= 4 and tok in low: return True
        return False
    if mode == "direction":
        ch = r.get("chart", {}) or {}
        m = {str(l): v for l, v in zip(ch.get("labels", []), ch.get("values", []))}
        se = next((v for l, v in m.items() if "서울" in l or "seoul" in l.lower()), None)
        bu = next((v for l, v in m.items() if "부산" in l or "busan" in l.lower()), None)
        if se is not None and bu is not None: return se > bu
        return ("seoul" in low or "서울" in hay) and ("busan" in low or "부산" in hay)
    ch = r.get("chart", {}) or {}
    got = _nums(hay) | {int(v) for v in ch.get("values", []) if isinstance(v, (int, float))}
    return any(v in got for v in o["values"])

def main(dry):
    print("=" * 78 + "\n  SQL 정확도 하니스 — 정의 B(headcount) · 오라클 vs answer_question()\n" + "=" * 78)
    print("\n── 오라클 정답 미리보기 " + "─" * 45)
    for q, ora, _, _ in GOLD:
        o = ora(); print(f"  · {q[:54]:54s} → {(o['names'][:1] or o['values'][:1])}")
    schema_guard()
    if dry: print("\n[--dry] API 생략.\n"); return
    sys.path.insert(0, str(BASE))
    try: from src.pipeline import answer_question
    except Exception as e: print(f"\n⚠️ import 실패: {e}"); return
    print("\n── 채점 " + "─" * 60)
    ro = an = 0
    for q, ora, exp, mode in GOLD:
        r = answer_question(q); rok = (r["route"] == exp)
        aok = _match(r, ora(), mode) if rok else False
        ro += rok; an += aok
        print(f"  route[{'O' if rok else 'X'}] answer[{'O' if aok else 'X'}] ({r['route']:6s}) {q[:50]}")
    n = len(GOLD)
    print("\n" + "=" * 78 + f"\n  라우팅 정확도 : {ro}/{n}\n  SQL 정답 정확도: {an}/{n}   ← 슬라이드/Q&A에 이 원분수\n" + "=" * 78)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows 콘솔(cp949)에서 '—' 깨짐 방지
    except Exception:
        pass
    main(dry="--dry" in sys.argv)
