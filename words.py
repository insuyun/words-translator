import re
import argparse
import openai
import tiktoken
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv

GPT_TOKEN_LIMIT = 8192
Mode = Literal["proofread", "translate"]

LANGUAGES = {
    "english": {"name": "영어", "bible": "NIV", "church": "Saenuri Church", "output": "translate_english.txt"},
    "vietnamese": {"name": "베트남어", "bible": "RVV", "church": "Nhà thờ Saenuri", "output": "translate_vietnamese.txt"},
    "uyghur": {"name": "위구르어", "bible": None, "church": "سائېنۇرى چېركاۋ", "output": "translate_uyghur.txt"},
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["proofread", "translate"], required=True)
    return parser.parse_args()

def count_tokens(text: str) -> int:
    # GPT-3.5-turbo 인코딩 사용
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def _check_token_limit(paragraphs: list[str]):
    for paragraph in paragraphs:
        tokens = count_tokens(paragraph)
        if tokens > GPT_TOKEN_LIMIT:
            raise ValueError(f"Total tokens exceed the limit: {tokens}", paragraph)


def _make_prompt(paragraph: str, mode: Mode, language: str = "english") -> str:
    if mode == "proofread":
        return f"""문장을 검수하고 오류를 수정하여 반환해주세요. 성경 문장은 개역개정판을 기준으로 합니다. 문장의 내용은 바꾸지 말고 내용만 수정해주세요. 절대 내용을 축약하거나 늘리지 말아주세요. 입력드리는 내용을 하나의 문단으로 바꿔주세요. 즉 줄바꿈을 다 없애주세요. 결과만 반환하고 설명은 하지 않습니다.

        ----------
        {paragraph}
        """
    elif mode == "translate":
        lang_config = LANGUAGES[language]
        bible_instruction = f" 성경 문장은 {lang_config['bible']}를 기준으로 합니다." if lang_config["bible"] else ""
        church_instruction = f" 새누리 교회처럼 보이는 단어는 {lang_config['church']}로 번역해주세요."
        return f"""문장을 {lang_config['name']}로 번역해주세요.{bible_instruction} 문장의 내용은 바꾸지 말고 내용만 번역해주세요. 절대 내용을 축약하거나 늘리지말아주세요. 입력드리는 내용을 하나의 문단으로 처리해주세요. 즉 줄바꿈 없이 하나의 문단으로 처리해주세요. 결과만 반환하고 설명은 하지 않습니다.{church_instruction}

        ----------
        {paragraph}
        """

def _make_system_prompt(mode: Mode, language: str = "english") -> str:
    if mode == "proofread":
        return "당신은 문장을 검수하는 프로그램입니다. 문장을 검수하고 오류를 수정하여 반환해주세요. 절대 질문하지 말고 결과만 반환하세요."
    elif mode == "translate":
        lang_config = LANGUAGES[language]
        return f"당신은 문장을 {lang_config['name']}로 번역하는 프로그램입니다. 문장을 {lang_config['name']}로 번역하여 반환해주세요. 절대 질문하지 말고 결과만 반환하세요."

def _to_paragraphs(mode: Mode, text: str) -> list[str]:
    if mode == "proofread":
        return re.split(r"-{5,}", text)
    elif mode == "translate":
        return re.split(r"\n\n", text)

def _process(client: openai.OpenAI, mode: Mode, language: str, input_path: str, output_path: str):
    if Path(output_path).exists():
        print(f"[!] Output file already exists, skipping: {output_path}")
        return

    input_text = Path(input_path).read_text()
    paragraphs = _to_paragraphs(mode, input_text)

    _check_token_limit(paragraphs)
    output = ""
    previous_response_id = None

    for paragraph in paragraphs:
        prompt = _make_prompt(paragraph, mode, language)
        if paragraph.strip() == "":
            continue

        result = None
        for attempt in range(3):
            response = client.responses.create(
                model="gpt-5.4-nano",
                instructions=_make_system_prompt(mode, language),
                input=prompt,
                max_output_tokens=16384,
                store=True,
                **({"previous_response_id": previous_response_id} if previous_response_id else {}),
            )

            if response.status == "completed":
                result = response.output_text
                previous_response_id = response.id
                break
            print(f"[!] Retry {attempt + 1}/3 (status={response.status})")

        if result is None:
            result = paragraph.strip()
            raise

        print(f"FROM: {paragraph}")
        print(f"TO: {result}\n\n")
        output += result + "\n\n"

    Path(output_path).write_text(output)
    print(f"[+] Written: {output_path}")

def main():
    args = parse_args()
    client = openai.OpenAI()

    if args.mode == "proofread":
        _process(client, "proofread", "english", "raw.txt", "proofread.txt")
    elif args.mode == "translate":
        for language, config in LANGUAGES.items():
            print(f"\n{'='*60}")
            print(f"[*] Translating to {config['name']} ({language})...")
            print(f"{'='*60}\n")
            _process(client, "translate", language, "proofread.txt", config["output"])

if __name__ == "__main__":
    load_dotenv()
    main()
