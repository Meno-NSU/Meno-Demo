from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
import rag_backend_server as backend


class FakeSessionStore:
    def __init__(self):
        self.synced = []
        self.appended = []
        self.cleared = set()

    async def sync_from_messages(self, session_id, messages):
        self.synced.append((session_id, len(messages)))

    async def get_previous_user_questions(self, session_id, current_question):
        return ["предыдущий вопрос"]

    async def append_assistant_turn(self, session_id, user_question, answer, meta):
        self.appended.append((session_id, user_question, answer, meta))

    async def clear(self, session_id):
        if session_id in self.cleared:
            return False
        self.cleared.add(session_id)
        return True


class FakeInferenceQueue:
    async def submit(self, request_id, session_id, user_question, previous_questions):
        return backend.RAGResult(
            answer="Готово.",
            search_queries=["q1", "q2"],
            relevant_documents=["doc1"],
            num_chunks=3,
            metrics={"total_sec": 0.01},
        )


def build_test_app():
    fake_store = FakeSessionStore()
    fake_queue = FakeInferenceQueue()

    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.ctx = SimpleNamespace(
            settings=backend.BackendSettings(
                model_path="/tmp/model",
                data_dir=Path("/tmp"),
                embedder_dir=Path("/tmp"),
                reranker_dir=Path("/tmp"),
            ),
            resources=None,
            pipeline=None,
            session_store=fake_store,
            inference_queue=fake_queue,
            gc_task=None,
            ready=True,
        )
        yield

    app = backend.FastAPI(title="test", lifespan=fake_lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ready"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: backend.ChatCompletionRequest):
        ctx = app.state.ctx
        session_id = backend.resolve_session_id(request.user, request.chat_id)
        user_question = backend.extract_latest_user_question(request.messages)
        await ctx.session_store.sync_from_messages(session_id, request.messages)
        previous_questions = await ctx.session_store.get_previous_user_questions(session_id, user_question)
        result = await ctx.inference_queue.submit(
            request_id="test",
            session_id=session_id,
            user_question=user_question,
            previous_questions=previous_questions,
        )
        await ctx.session_store.append_assistant_turn(
            session_id=session_id,
            user_question=user_question,
            answer=result.answer,
            meta={"metrics": result.metrics},
        )
        if request.stream:
            return await backend.build_sse_response(result.answer, request.model)
        return backend.JSONResponse(content=backend.format_non_stream_response(result.answer, request.model))

    @app.post("/v1/chat/completions/clear_history")
    async def clear_history(request: backend.ClearHistoryRequest):
        ctx = app.state.ctx
        session_id = backend.resolve_session_id(request.user, request.chat_id)
        cleared = await ctx.session_store.clear(session_id)
        return {"status": "ok", "cleared": cleared, "session_id": session_id}

    return app


def test_resolve_session_id_prefers_user():
    assert backend.resolve_session_id("  user-1  ", "  42 ") == "user-1"


def test_resolve_session_id_uses_chat_id():
    assert backend.resolve_session_id(None, " 42 ") == "42"


def test_resolve_session_id_raises_on_missing():
    with pytest.raises(backend.HTTPException):
        backend.resolve_session_id(None, "")


def test_extract_latest_user_question_returns_latest_nonempty_user():
    messages = [
        backend.ChatMessage(role="assistant", content="x"),
        backend.ChatMessage(role="user", content="  "),
        backend.ChatMessage(role="user", content=" Вопрос? "),
    ]
    assert backend.extract_latest_user_question(messages) == "Вопрос?"


def test_sanitize_assistant_answer_removes_think_block():
    raw = "<think>reasoning</think> Итоговый ответ"
    assert backend.sanitize_assistant_answer(raw) == "Итоговый ответ"


def test_split_for_sse_stream_returns_pieces():
    pieces = backend.split_for_sse_stream("abcdef", chunk_size=2)
    assert pieces == ["ab", "cd", "ef"]


def test_chat_completions_non_stream_shape():
    app = build_test_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "menon-1",
                "user": "chat-1",
                "messages": [{"role": "user", "content": "Привет"}],
                "stream": False,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_stream_has_done_event():
    app = build_test_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "menon-1",
                "user": "chat-1",
                "messages": [{"role": "user", "content": "Привет"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text


def test_clear_history_endpoint():
    app = build_test_app()
    with TestClient(app) as client:
        first = client.post("/v1/chat/completions/clear_history", json={"chat_id": "42"})
        second = client.post("/v1/chat/completions/clear_history", json={"chat_id": "42"})
    assert first.status_code == 200
    assert first.json()["cleared"] is True
    assert second.status_code == 200
    assert second.json()["cleared"] is False
