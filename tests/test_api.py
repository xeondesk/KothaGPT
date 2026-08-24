from fastapi.testclient import TestClient

from services.api.app import app
from services.api.core import backend_factory
from services.api.core.example_backend import CannedBackend, HFExampleBackend
from services.api.core.mock_backend import MockBackend

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_default_mock_backend():
    response = client.post(
        "/v1/chat",
        json={"message": "বাংলায় একটি সংক্ষিপ্ত পরিচয় দাও।"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "kothagpt"
    assert body["message"]
    assert body["output"]


def test_chat_completions_default_mock_backend():
    response = client.post(
        "/v1/chat/completions",
        json={"model": "kothagpt", "messages": [{"role": "user", "content": "হ্যালো"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]


def test_canned_backend(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "canned")
    backend = backend_factory.create()
    assert isinstance(backend, CannedBackend)
    output = backend.chat(_chat_request("আমি কোন মডেল ব্যবহার করছি?")).choices[0].message.content
    assert "ক্যানড" in output
    assert "আমি কোন মডেল" in output


def test_hf_backend_selection(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "hf")
    backend = backend_factory.create()
    assert isinstance(backend, HFExampleBackend)


def test_hf_backend_falls_back_to_canned_without_deps(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "hf")
    backend = backend_factory.create()
    output = backend.chat(_chat_request("পরীক্ষা")).choices[0].message.content
    # transformers/torch are not installed in the test env, so the shim
    # must degrade to the canned reply instead of raising.
    assert "ক্যানড" in output


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("KOTHAGPT_BACKEND", "does-not-exist")
    try:
        backend_factory.create()
    except ValueError as exc:
        assert "does-not-exist" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown backend")


def test_factory_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("KOTHAGPT_BACKEND", raising=False)
    assert isinstance(backend_factory.create(), MockBackend)


def _chat_request(message: str):
    from services.api.api.schemas import ChatCompletionRequest

    return ChatCompletionRequest(messages=[{"role": "user", "content": message}])
