\# SELECT \* FROM Korea



한국 유학을 고려하는 외국인 학생을 위한 데이터 기반 의사결정 지원 시스템.

2026 BIGDATA-USC Conference Hackathon (7/25, SGM 101)



\## 무엇이 다른가



\- 정량 질문은 \*\*SQL\*\*로 실제 공공데이터를 조회합니다. 벡터 검색은 집계 연산을 못 합니다.

\- 정성 질문은 \*\*RAG\*\*로 비자·학사 규정을 검색해 출처와 함께 답변합니다.

\- 근거가 부족하면 \*\*답변을 거부합니다\*\* (Abstention). 비자 정보는 틀리면 사람이 다칩니다.



\## 시작하기



```

git clone https://github.com/ssub17/kcampus-navigator.git

cd kcampus-navigator



python -m venv .venv

.venv\\Scripts\\activate          # Windows

source .venv/bin/activate       # macOS



pip install -r requirements.txt



copy .env.example .env          # Windows

cp .env.example .env            # macOS

\# .env 에 키 입력 (팀 채팅방 참조)



streamlit run app.py

```



\## 파일 소유권 (충돌 방지)



| 담당 | 파일 |

|---|---|

| 백엔드 | `contract.py` `mock.py` `src/router.py` `src/sql\_chain.py` `src/retriever.py` `src/vector\_store.py` `src/pipeline.py` |

| 프론트 | `app.py` `src/charts.py` |

| 데이터 | `data/` `notebooks/` `eval/` `docs/` |



\## 협업 규칙



1\. \*\*자기 담당 파일만 수정합니다.\*\* 남의 파일이 고쳐져야 하면 직접 고치지 말고 채팅방에 말하세요.

2\. 푸시 전 항상 `git pull --rebase`

3\. `main` 직접 푸시 (PR 없음). 12시간 안에 리뷰 대기는 사치입니다.

4\. `contract.py` 는 확정본입니다. 변경하려면 반드시 먼저 공지하세요.

5\. `.env` 는 절대 커밋 금지. 커밋되면 OpenAI가 키를 자동 폐기합니다.

