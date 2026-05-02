from words import _make_prompt, _make_system_prompt


def test_proofread_prompt_includes_paragraph_and_review_keyword():
    paragraph = "유니크식별문자열"
    result = _make_prompt(paragraph, "proofread")
    assert paragraph in result
    assert "검수" in result


def test_proofread_prompt_ignores_language():
    paragraph = "abc"
    en = _make_prompt(paragraph, "proofread", "english")
    vi = _make_prompt(paragraph, "proofread", "vietnamese")
    assert en == vi


def test_translate_prompt_english_includes_language_bible_and_church():
    result = _make_prompt("유니크식별문자열", "translate", "english")
    assert "유니크식별문자열" in result
    assert "영어" in result
    assert "NIV" in result
    assert "Saenuri Church" in result


def test_translate_prompt_vietnamese_includes_rvv_and_church():
    result = _make_prompt("p", "translate", "vietnamese")
    assert "베트남어" in result
    assert "RVV" in result
    assert "Nhà thờ Saenuri" in result


def test_translate_prompt_uyghur_omits_bible_instruction():
    result = _make_prompt("p", "translate", "uyghur")
    assert "위구르어" in result
    assert "사이ېنۇری" not in result
    assert "성경 문장은" not in result
    assert "سائېنۇرى چېركاۋ" in result


def test_extra_prompt_appended_when_provided():
    base = _make_prompt("p", "proofread")
    with_extra = _make_prompt("p", "proofread", extra_prompt="추가지시")
    assert "추가지시" in with_extra
    assert "추가지시" not in base


def test_extra_prompt_skipped_when_empty_string():
    a = _make_prompt("p", "proofread", extra_prompt="")
    b = _make_prompt("p", "proofread")
    assert a == b


def test_extra_prompt_appended_for_translate_mode():
    result = _make_prompt("p", "translate", "english", extra_prompt="추가지시")
    assert "추가지시" in result


def test_system_prompt_proofread_keywords():
    result = _make_system_prompt("proofread")
    assert "검수" in result


def test_system_prompt_translate_includes_language_name():
    assert "영어" in _make_system_prompt("translate", "english")
    assert "베트남어" in _make_system_prompt("translate", "vietnamese")
    assert "위구르어" in _make_system_prompt("translate", "uyghur")


def test_system_prompt_includes_glossary_when_provided():
    glossary = "용탕: 용서받은 탕자"
    result = _make_system_prompt("proofread", glossary=glossary)
    assert "용탕: 용서받은 탕자" in result
    assert "용어집" in result


def test_system_prompt_omits_glossary_block_when_empty():
    base = _make_system_prompt("proofread")
    with_empty = _make_system_prompt("proofread", glossary="")
    with_whitespace = _make_system_prompt("proofread", glossary="   \n  ")
    assert base == with_empty == with_whitespace
    assert "용어집" not in base


def test_system_prompt_glossary_applies_to_translate_mode():
    glossary = "이웅 목사님: 담임목사"
    for language in ("english", "vietnamese", "uyghur"):
        result = _make_system_prompt("translate", language, glossary=glossary)
        assert "이웅 목사님: 담임목사" in result
