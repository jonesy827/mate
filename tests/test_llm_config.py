"""Voice-brain LLM option assembly: the default local path gets the
qwen-tuned sampling block; a real API key (hosted provider) must NOT send
it — api.openai.com 400s on top_k, and gpt-5-era models reject any
non-default temperature."""

from mate import agent


def test_local_default_gets_qwen_sampling(monkeypatch):
    monkeypatch.setattr(agent, "LLM_API_KEY", "local")
    opts = agent.llm_options()
    assert opts["api_key"] == "local"
    assert opts["temperature"] == 0.7
    assert opts["extra_body"]["top_k"] == 20
    assert opts["extra_body"]["presence_penalty"] == 0
    assert opts["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False}


def test_hosted_key_sends_only_provider_safe_params(monkeypatch):
    monkeypatch.setattr(agent, "LLM_API_KEY", "sk-test")
    monkeypatch.setattr(agent, "LLM_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(agent, "LLM_MODEL", "gpt-5-mini")
    assert agent.llm_options() == {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test",
        "model": "gpt-5-mini",
    }
