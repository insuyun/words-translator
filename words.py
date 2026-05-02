import re
import argparse
import subprocess
import datetime
import openai
import tiktoken
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv

GPT_TOKEN_LIMIT = 8192
Mode = Literal["proofread", "translate"]
AI_NOTE = "NOTE: This is AI-generated, so there may be some inaccuracies. Thank you for your understanding."

LANGUAGES = {
    "english": {
        "code": "en",
        "name": "영어",
        "bible": "NIV",
        "church": "Saenuri Church",
        "font": None,
        "pandoc_extra": ["-V", "lang=en"],
    },
    "vietnamese": {
        "code": "vi",
        "name": "베트남어",
        "bible": "RVV",
        "church": "Nhà thờ Saenuri",
        "font": "DejaVu Serif",
        "pandoc_extra": ["-V", "lang=vi"],
    },
    "uyghur": {
        "code": "ug",
        "name": "위구르어",
        "bible": None,
        "church": "سائېنۇرى چېركاۋ",
        "font": "Noto Naskh Arabic",
        "pandoc_extra": ["-V", "dir=rtl", "-V", "lang=ug"],
    },
}

SCRIPTURE_HINT = (
    "다음은 성경 본문 표기입니다. '대상'(역대상), '대하'(역대하), '삼상'(사무엘상), '삼하'(사무엘하), "
    "'왕상'(열왕기상), '왕하'(열왕기하), '시'(시편), '잠'(잠언), '전'(전도서), '사'(이사야), "
    "'렘'(예레미야), '겔'(에스겔), '단'(다니엘), '마'(마태복음), '막'(마가복음), '눅'(누가복음), "
    "'요'(요한복음), '행'(사도행전), '롬'(로마서), '고전'(고린도전서), '고후'(고린도후서), "
    "'갈'(갈라디아서), '엡'(에베소서), '빌'(빌립보서), '골'(골로새서), '히'(히브리서), '계'(요한계시록) "
    "등 한국어 성경 책 이름의 줄임말을 정확히 풀어 해당 언어의 표준 성경 책 이름으로 번역하세요. "
    "장:절 표기는 그대로 유지합니다."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["proofread", "translate"], required=True)
    parser.add_argument("--prompt", type=str, default="", help="추가 프롬프트 지시사항")
    parser.add_argument("--language", type=str, choices=list(LANGUAGES.keys()), default=None,
                        help="translate 모드에서 특정 언어 하나만 번역 (생략 시 전체 언어)")
    parser.add_argument("--glossary", type=str, default=None,
                        help="용어집 markdown 파일 경로 (기본: ./glossary.md, 없으면 무시)")
    return parser.parse_args()


DEFAULT_GLOSSARY_PATH = Path("glossary.md")


def _load_glossary(path: Path | None) -> str:
    if path is None:
        return DEFAULT_GLOSSARY_PATH.read_text() if DEFAULT_GLOSSARY_PATH.exists() else ""
    if not path.exists():
        raise FileNotFoundError(f"용어집 파일을 찾을 수 없습니다: {path}")
    return path.read_text()

def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def _check_token_limit(paragraphs: list[str]):
    for paragraph in paragraphs:
        tokens = count_tokens(paragraph)
        if tokens > GPT_TOKEN_LIMIT:
            raise ValueError(f"Total tokens exceed the limit: {tokens}", paragraph)


def _today_str() -> str:
    return datetime.date.today().strftime("%Y.%m.%d")


def _parse_raw(text: str) -> tuple[str, str, str]:
    title_match = re.search(r"^제목:[ \t]*(.+)$", text, re.MULTILINE)
    scripture_match = re.search(r"^본문:[ \t]*(.+)$", text, re.MULTILINE)
    sep_match = re.search(r"^---+\s*$", text, re.MULTILINE)

    missing = []
    if not title_match or not title_match.group(1).strip():
        missing.append("제목")
    if not scripture_match or not scripture_match.group(1).strip():
        missing.append("본문")
    if not sep_match:
        missing.append("--- 구분선")
    if missing:
        raise ValueError(f"입력 파일에 누락된 필드: {', '.join(missing)}")

    title = title_match.group(1).strip()
    scripture = scripture_match.group(1).strip()
    body = text[sep_match.end():].strip()

    if not body:
        raise ValueError("입력 파일에 누락된 필드: 설교 본문")

    return title, scripture, body


def _format_with_header(title: str, scripture: str, sermon: str) -> str:
    return f"제목: {title}\n본문: {scripture}\n---\n{sermon}\n"


def _make_prompt(paragraph: str, mode: Mode, language: str = "english", extra_prompt: str = "") -> str:
    extra = f" {extra_prompt}" if extra_prompt else ""
    if mode == "proofread":
        return f"""문장을 검수하고 오류를 수정하여 반환해주세요. 성경 문장은 개역개정판을 기준으로 합니다. 문장의 내용은 바꾸지 말고 내용만 수정해주세요. 절대 내용을 축약하거나 늘리지 말아주세요. 입력드리는 내용을 하나의 문단으로 바꿔주세요. 즉 줄바꿈을 다 없애주세요. 결과만 반환하고 설명은 하지 않습니다.{extra}

        ----------
        {paragraph}
        """
    elif mode == "translate":
        lang_config = LANGUAGES[language]
        bible_instruction = f" 성경 문장은 {lang_config['bible']}를 기준으로 합니다." if lang_config["bible"] else ""
        church_instruction = f" 새누리 교회처럼 보이는 단어는 {lang_config['church']}로 번역해주세요."
        return f"""문장을 {lang_config['name']}로 번역해주세요.{bible_instruction} 문장의 내용은 바꾸지 말고 내용만 번역해주세요. 절대 내용을 축약하거나 늘리지말아주세요. 입력드리는 내용을 하나의 문단으로 처리해주세요. 즉 줄바꿈 없이 하나의 문단으로 처리해주세요. 결과만 반환하고 설명은 하지 않습니다.{church_instruction}{extra}

        ----------
        {paragraph}
        """

def _glossary_block(glossary: str) -> str:
    if not glossary.strip():
        return ""
    return (
        "\n\n다음은 이 교회 고유의 용어집입니다. 검수/번역 시 반드시 이 용어집을 따르세요. "
        "사역자 이름 및 고유 용어는 임의로 수정/축약하지 말고 용어집 지침대로 처리하세요.\n"
        "----------\n"
        f"{glossary.strip()}\n"
        "----------"
    )


def _make_system_prompt(mode: Mode, language: str = "english", glossary: str = "") -> str:
    if mode == "proofread":
        base = "당신은 문장을 검수하는 프로그램입니다. 문장을 검수하고 오류를 수정하여 반환해주세요. 절대 질문하지 말고 결과만 반환하세요."
    elif mode == "translate":
        lang_config = LANGUAGES[language]
        base = f"당신은 문장을 {lang_config['name']}로 번역하는 프로그램입니다. 문장을 {lang_config['name']}로 번역하여 반환해주세요. 절대 질문하지 말고 결과만 반환하세요."
    return base + _glossary_block(glossary)

def _to_paragraphs(mode: Mode, text: str) -> list[str]:
    if mode == "proofread":
        return re.split(r"-{5,}", text)
    elif mode == "translate":
        return re.split(r"\n\n", text)


def _call_gpt(client: openai.OpenAI, mode: Mode, language: str, prompt: str, previous_response_id: str | None, glossary: str = "") -> tuple[str, str]:
    for attempt in range(3):
        response = client.responses.create(
            model="gpt-5.4-nano",
            instructions=_make_system_prompt(mode, language, glossary),
            input=prompt,
            max_output_tokens=16384,
            store=True,
            **({"previous_response_id": previous_response_id} if previous_response_id else {}),
        )
        if response.status == "completed":
            return response.output_text, response.id
        print(f"[!] Retry {attempt + 1}/3 (status={response.status})")
    raise RuntimeError(f"GPT call failed after 3 retries (mode={mode}, language={language})")


def _process_body(client: openai.OpenAI, mode: Mode, language: str, body: str, extra_prompt: str = "", glossary: str = "") -> str:
    paragraphs = _to_paragraphs(mode, body)
    _check_token_limit(paragraphs)

    output_paragraphs = []
    previous_response_id = None

    for paragraph in paragraphs:
        if paragraph.strip() == "":
            continue
        prompt = _make_prompt(paragraph, mode, language, extra_prompt)
        result, previous_response_id = _call_gpt(client, mode, language, prompt, previous_response_id, glossary)
        print(f"FROM: {paragraph}")
        print(f"TO: {result}\n\n")
        output_paragraphs.append(result)

    return "\n\n".join(output_paragraphs)


def _translate_text(client: openai.OpenAI, language: str, text: str, extra_prompt: str = "", glossary: str = "") -> str:
    prompt = _make_prompt(text, "translate", language, extra_prompt)
    result, _ = _call_gpt(client, "translate", language, prompt, None, glossary)
    return result.strip()


def _run_pandoc(md_path: Path, pdf_path: Path, language: str):
    config = LANGUAGES[language]
    cmd = ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=xelatex",
           "-V", "geometry:margin=2cm"]
    if config.get("font"):
        cmd += ["-V", f"mainfont={config['font']}"]
    cmd += config.get("pandoc_extra", [])
    print(f"[*] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def proofread_main(client: openai.OpenAI, extra_prompt: str = "", glossary: str = ""):
    output_path = Path("proofread.txt")
    if output_path.exists():
        print(f"[!] Output file already exists, skipping: {output_path}")
        return

    title, scripture, sermon = _parse_raw(Path("raw.txt").read_text())
    proofread_sermon = _process_body(client, "proofread", "english", sermon, extra_prompt, glossary)
    output_path.write_text(_format_with_header(title, scripture, proofread_sermon))
    print(f"[+] Written: {output_path}")


def translate_main(client: openai.OpenAI, extra_prompt: str = "", only_language: str | None = None, glossary: str = ""):
    title, scripture, sermon = _parse_raw(Path("proofread.txt").read_text())
    today = _today_str()
    out_dir = Path(today)
    out_dir.mkdir(parents=True, exist_ok=True)

    languages = {only_language: LANGUAGES[only_language]} if only_language else LANGUAGES
    for language, config in languages.items():
        print(f"\n{'='*60}")
        print(f"[*] Translating to {config['name']} ({language})...")
        print(f"{'='*60}\n")

        pdf_path = out_dir / f"{today}-{config['code']}.pdf"
        if pdf_path.exists():
            print(f"[!] PDF already exists, skipping: {pdf_path}")
            continue

        translated_title = _translate_text(client, language, title, extra_prompt, glossary)
        translated_scripture = _translate_text(client, language, scripture, SCRIPTURE_HINT, glossary)
        translated_sermon = _process_body(client, "translate", language, sermon, extra_prompt, glossary)

        md_content = (
            f"# {translated_title} ({translated_scripture})\n\n"
            f"*{today}*\n\n"
            f"> {AI_NOTE}\n\n"
            f"{translated_sermon}\n"
        )
        md_path = out_dir / f"{today}-{config['code']}.md"
        md_path.write_text(md_content)
        print(f"[+] Written: {md_path}")

        _run_pandoc(md_path, pdf_path, language)
        print(f"[+] Written: {pdf_path}")


def main():
    args = parse_args()
    client = openai.OpenAI()

    glossary = _load_glossary(Path(args.glossary) if args.glossary else None)
    if glossary:
        print(f"[*] 용어집 로드: {len(glossary)}자")

    if args.mode == "proofread":
        proofread_main(client, args.prompt, glossary)
    elif args.mode == "translate":
        translate_main(client, args.prompt, args.language, glossary)

if __name__ == "__main__":
    load_dotenv()
    main()
