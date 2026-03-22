import re
import argparse
import openai
import tiktoken
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv

GPT_TOKEN_LIMIT = 8192
Mode = Literal["proofread", "translate"]


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["proofread", "translate"], required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
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


def _make_prompt(paragraph: str, mode: Mode) -> str:
    if mode == "proofread":
        return f"""문장을 검수하고 오류를 수정하여 반환해주세요. 성경 문장은 개역개정판을 기준으로 합니다. 문장의 내용은 바꾸지 말고 내용만 수정해주세요. 절대 내용을 축약하거나 늘리지 말아주세요. 입력드리는 내용을 하나의 문단으로 바꿔주세요. 즉 줄바꿈을 다 없애주세요. 결과만 반환하고 설명은 하지 않습니다.

        ----------
        {paragraph}
        """
    elif mode == "translate":
        return f"""문장을 영어로 번역해주세요. 성경 문장은 NIV를 기준으로 합니다. 문장의 내용은 바꾸지 말고 내용만 번역해주세요. 절대 내용을 축약하거나 늘리지말아주세요. 입력드리는 내용을 하나의 문단으로 처리해주세요. 즉 줄바꿈 없이 하나의 문단으로 처리해주세요. 결과만 반환하고 설명은 하지 않습니다. 새누리 교회처럼 보이는 단어는 Saenuri Church로 번역해주세요..

        ----------
        {paragraph}
        """

def _make_system_prompt(mode: Mode) -> str:
    if mode == "proofread":
        return "당신은 문장을 검수하는 프로그램입니다. 문장을 검수하고 오류를 수정하여 반환해주세요."
    elif mode == "translate":
        return "당신은 문장을 번역하는 프로그램입니다. 문장을 번역하여 반환해주세요."

def _to_paragraphs(mode: Mode, text: str) -> list[str]:
    if mode == "proofread":
        return re.split(r"-{5,}", text)
    elif mode == "translate":
        return re.split(r"\n\n", text)

def main():
    args = parse_args()

    if Path(args.output).exists():
        print(f"[!] Output file already exists: {args.output}")
        return

    client = openai.OpenAI()
    input_text = Path(args.input).read_text()
    paragraphs = _to_paragraphs(args.mode, input_text)

    _check_token_limit(paragraphs)
    output = ""
    previous_response_id = None

    for paragraph in paragraphs:
        prompt = _make_prompt(paragraph, args.mode)
        if paragraph.strip() == "":
            continue

        result = None
        for attempt in range(3):
            response = client.responses.create(
                model="gpt-4o",
                instructions=_make_system_prompt(args.mode),
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

    Path(args.output).write_text(output)

if __name__ == "__main__":
    load_dotenv()
    main()
