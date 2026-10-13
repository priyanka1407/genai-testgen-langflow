"""GeminiLLM tests with a fake client - no API key or network needed (skipped if the SDK is absent)."""
from types import SimpleNamespace

import pytest

genai_errors = pytest.importorskip("google.genai.errors")

from testgen import llm  # noqa: E402
from testgen.models import RequirementSet  # noqa: E402

GOOD_JSON = ('{"requirements": [{"req_id": "TMP-1", "module": "Payments", "title": "Daily limit", '
             '"description": "d", "acceptance_criteria": ["up to 10,000"], "business_rules": ["10,000"], '
             '"risk": "high", "source": "notes"}]}')


def _response(parsed=None, text=GOOD_JSON, finish="STOP", block=None):
    return SimpleNamespace(
        parsed=parsed, text=text, prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20, thoughts_token_count=5),
    )


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _gemini(outcomes, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    g = llm.GeminiLLM(model="gemini-3.8-flash", api_key="test-key")
    g.client = SimpleNamespace(models=FakeModels(outcomes))
    return g


def test_provider_is_chosen_from_the_model_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(llm.get_llm(offline=False, model="gemini-3.8-flash"), llm.GeminiLLM)
    assert llm.provider_for("claude-opus-5-5") == "anthropic"


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No Gemini API key"):
        llm.GeminiLLM(model="gemini-3.8-flash")


def test_structured_output_request_and_parsing(monkeypatch):
    parsed = RequirementSet.model_validate_json(GOOD_JSON)
    g = _gemini([_response(parsed=parsed)], monkeypatch)
    out = g.generate("system prompt", "user prompt", RequirementSet, {})
    assert out.requirements[0].title == "Daily limit"
    call = g.client.models.calls[0]
    assert call.model == "gemini-3.8-flash" and call.contents == "user prompt"
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_schema is RequirementSet
    assert call.config.system_instruction == "system prompt"


def test_falls_back_to_validating_the_json_text(monkeypatch):
    g = _gemini([_response(parsed=None)], monkeypatch)
    assert g.generate("s", "p", RequirementSet, {}).requirements[0].risk == "high"


def test_rate_limit_is_retried(monkeypatch):
    err = genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}})
    g = _gemini([err, err, _response(parsed=RequirementSet.model_validate_json(GOOD_JSON))], monkeypatch)
    assert g.generate("s", "p", RequirementSet, {}).requirements
    assert len(g.client.models.calls) == 3


def test_bad_key_is_not_retried(monkeypatch):
    err = genai_errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid"}})
    g = _gemini([err], monkeypatch)
    with pytest.raises(genai_errors.ClientError):
        g.generate("s", "p", RequirementSet, {})
    assert len(g.client.models.calls) == 1


@pytest.mark.parametrize("kw, exc", [({"finish": "SAFETY"}, llm.LLMRefusal),
                                     ({"block": "PROHIBITED_CONTENT"}, llm.LLMRefusal),
                                     ({"finish": "MAX_TOKENS"}, RuntimeError)])
def test_blocked_or_truncated_responses_raise(monkeypatch, kw, exc):
    g = _gemini([_response(**kw)], monkeypatch)
    with pytest.raises(exc):
        g.generate("s", "p", RequirementSet, {})


def _quota_error(retry_delay: str, quota_id: str):
    return genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                     "violations": [{"quotaId": quota_id, "quotaValue": "20"}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]}})


def test_daily_quota_fails_immediately_instead_of_hanging(monkeypatch):
    g = _gemini([_quota_error("21129s", "GenerateRequestsPerDayPerProjectPerModel-FreeTier")], monkeypatch)
    with pytest.raises(llm.QuotaExhausted, match="resets in about 5.9 h"):
        g.generate("s", "p", RequirementSet, {})
    assert len(g.client.models.calls) == 1


def test_per_minute_limit_waits_as_told_and_retries(monkeypatch):
    waits = []
    g = _gemini([_quota_error("12s", "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
                 _response(parsed=RequirementSet.model_validate_json(GOOD_JSON))], monkeypatch)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    assert g.generate("s", "p", RequirementSet, {}).requirements
    assert waits == [12.0]


def test_quota_exhaustion_is_not_retried_per_requirement(monkeypatch):
    from testgen import generator
    from testgen.models import Requirement
    calls = []

    class QuotaLLM:
        name = "gemini-test"

        def generate(self, *a):
            calls.append(1)
            raise llm.QuotaExhausted("daily quota")

    req = Requirement(req_id="REQ-PAY-001", module="Payments", title="t", description="t",
                      acceptance_criteria=["a"], business_rules=[], risk="low", source="s")
    res = generator.generate_test_cases(QuotaLLM(), [req], {}, None)
    assert res["REQ-PAY-001"].error == "daily quota" and len(calls) == 1


def test_env_file_loading_and_model_defaults_read_at_call_time(tmp_path, monkeypatch):
    from testgen.env import load_env_file
    monkeypatch.delenv("TESTGEN_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("# comment\nTESTGEN_MODEL=gemini-3.6-flash\nANTHROPIC_API_KEY=from-file\nEMPTY=\n", encoding="utf-8")
    assert load_env_file(env) == ["TESTGEN_MODEL"]            # shell value wins, empty skipped
    assert llm.default_model() == "gemini-3.6-flash"           # picked up after import
    assert llm.default_judge_model() == "gemini-3.6-flash"
    monkeypatch.delenv("TESTGEN_MODEL")
