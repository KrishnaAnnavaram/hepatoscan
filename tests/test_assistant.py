import pytest

from hepatoscan.assistant.chat import NO_INFO, Assistant, SessionStore, valid_markers
from hepatoscan.assistant.kb import load_kb, parse_document
from hepatoscan.assistant.llm import OfflineLLM, OpenAICompatibleLLM, ScriptedLLM, make_llm
from hepatoscan.assistant.retriever import BM25
from hepatoscan.assistant.safety import DISCLAIMER, ESCALATION, NON_DIAGNOSTIC, has_diagnostic_claim, triage


def test_kb_loads_with_sources():
    passages = load_kb()
    assert len(passages) >= 8 and all(p.source for p in passages)
    with pytest.raises(ValueError):
        parse_document("no front matter", "x")


def test_retriever_finds_the_numbers_page():
    hits = BM25(load_kb()).search("what does the lesion count mean", k=2)
    assert hits and hits[0].passage.pid.startswith("03_reading_the_numbers")


@pytest.mark.parametrize("q", ["I am vomiting blood", "I have severe abdominal pain", "I feel confused and faint",
                               "I can't breathe"])
def test_red_flags_escalate_without_a_model_call(q):
    llm = ScriptedLLM(["should not be used"])
    reply = Assistant(llm).ask(q)
    assert reply.escalated and ESCALATION in reply.answer and reply.answer.endswith(DISCLAIMER)
    assert llm.calls == []


@pytest.mark.parametrize("q", ["Do I have cancer?", "Is my tumor malignant?", "What stage is it?",
                               "How much paracetamol should I take?", "What is my prognosis?"])
def test_diagnosis_and_dose_requests_are_refused(q):
    reply = Assistant().ask(q)
    assert reply.refused and reply.answer.startswith(NON_DIAGNOSTIC)
    assert triage(q).diagnosis_request


def test_invalid_markers_are_removed():
    text, cited = valid_markers("A [1]. B [7]. C [2] [1].", 2)
    assert cited == [1, 2] and "[7]" not in text


def test_diagnostic_claim_from_the_model_is_replaced():
    llm = ScriptedLLM(["You have cancer [1]."])
    reply = Assistant(llm).ask("What does the lesion count mean?")
    assert "You have cancer" not in reply.answer
    assert reply.model == OfflineLLM.name and reply.citations
    assert has_diagnostic_claim("It is malignant.") and not has_diagnostic_claim("Many lesions are benign.")


def test_model_answer_without_citation_falls_back_to_extractive():
    reply = Assistant(ScriptedLLM(["Trust me, all is fine."])).ask("What does the lesion count mean?")
    assert reply.model == OfflineLLM.name and "[1]" in reply.answer


def test_valid_model_answer_is_used_and_history_is_not_duplicated():
    llm = ScriptedLLM(["The lesion count is the number of marked regions [1].", "It is in millilitres [1]."])
    bot = Assistant(llm, max_exchanges=1)
    first = bot.ask("What does the lesion count mean?", summary_text="liver region 1500 mL")
    second = bot.ask("What unit is the liver volume in?", session_id=first.session_id)
    assert second.model == "scripted"
    msgs = llm.calls[1]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert users == ["What does the lesion count mean?", "What unit is the liver volume in?"]
    assert any(m["content"].startswith("PASSAGES\n") for m in msgs)
    third = bot.ask("What is a CT scan?", session_id=first.session_id)
    users3 = [m["content"] for m in llm.calls[-1] if m["role"] == "user"] if len(llm.calls) > 2 else []
    assert third.session_id == first.session_id
    assert "What does the lesion count mean?" not in users3  # capped to one exchange


def test_question_without_sources_gets_no_model_call():
    llm = ScriptedLLM(["x"])
    reply = Assistant(llm).ask("zxqv blorf")
    assert reply.answer.startswith(NO_INFO) and not reply.grounded and llm.calls == []


def test_empty_and_long_questions_are_rejected():
    bot = Assistant()
    with pytest.raises(ValueError):
        bot.ask("  ")
    with pytest.raises(ValueError):
        bot.ask("a" * 2001)


def test_sessions_expire_and_are_capped():
    store = SessionStore(max_exchanges=2, ttl_s=0.0)
    s = store.get(None)
    for i in range(5):
        store.add_exchange(s, f"q{i}", f"a{i}")
    assert len(s.turns) == 4 and s.turns[0] == ("user", "q3")
    assert store.get(s.sid).sid != s.sid  # expired with ttl 0
    assert store.delete("missing") is False


def test_llm_configuration_rules():
    with pytest.raises(ValueError):
        OpenAICompatibleLLM("http://example.com/v1", "m", None)
    assert OpenAICompatibleLLM("http://localhost:11434/v1", "m", None).base_url.endswith("/v1")
    assert isinstance(make_llm("offline", "", "", None, 1), OfflineLLM)
    with pytest.raises(ValueError):
        make_llm("gemini-sdk", "", "", None, 1)


def test_provider_error_falls_back():
    class Broken:
        name = "broken"

        def complete(self, messages):
            raise TimeoutError("down")

    reply = Assistant(Broken()).ask("What is a CT scan?")
    assert reply.model == OfflineLLM.name and reply.citations
