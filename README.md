# Words Translator: 설교문 번역기

## 세팅
1. `openai` API 키를 발급받아 `.env` 파일에 저장합니다.
```bash
# .env
OPENAI_API_KEY=sk-...
```

2. `requirements.txt`에 있는 패키지를 설치합니다.
```bash
pip install -r requirements.txt
```

## 사용법
1. [네이버 클로바 노트](https://clova.ai/note)를 이용하여 설교문을 녹음하여 텍스트로 변환합니다.
2. 녹음 텍스트를 복사하여 `samples/1_raw.txt`와 같은 파일에 저장합니다.
3. 해당 파일을 확인하여 "------"로 문단을 구분하여 `samples/2_break.txt`와 같은 파일에 저장합니다.
4. `words.py`를 실행하여 맞춤법 검사를 진행합니다
```bash
python words.py --mode proofread --input samples/2_break.txt --output samples/3_proofread.txt

```
5. 맞춤법 검사 결과를 확인하여 오류가 있는 부분을 수정합니다.
6. `words.py`를 실행하여 번역을 진행합니다
```bash
python words.py --mode translate --input samples/3_proofread.txt --output samples/4_translated.txt
```

## 번역본
[새누리 교회 설교 번역](https://drive.google.com/file/d/11GujPdp8BoR66tFNpmRstnF6haqJiTWb/view?usp=sharing)
