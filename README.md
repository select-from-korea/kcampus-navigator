# SELECT \* FROM Korea



한국 유학을 고려하는 외국인 학생을 위한 데이터 기반 의사결정 지원 시스템.

2026 BIGDATA-USC Conference Hackathon (7/25, SGM 101)



## 무엇이 다른가



- 정량 질문은 \*\*SQL\*\*로 실제 공공데이터를 조회합니다. 벡터 검색은 집계 연산을 못 합니다.

- 정성 질문은 \*\*RAG\*\*로 비자·학사 규정을 검색해 출처와 함께 답변합니다.

- 근거가 부족하면 \*\*답변을 거부합니다\*\* (Abstention). 비자 정보는 틀리면 사람이 다칩니다.



## 시작하기



```

git clone https://github.com/ssub17/kcampus-navigator.git

cd kcampus-navigator



python -m venv .venv

.venv\\Scripts\\activate          # Windows

source .venv/bin/activate       # macOS



pip install -r requirements.txt



copy .env.example .env          # Windows

cp .env.example .env            # macOS

# .env 에 키 입력



streamlit run app.py

```

