from __future__ import annotations

import argparse
import asyncio
import codecs
import gc
import json
import logging
import os
import random
import signal
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from annoy import AnnoyIndex
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from nltk.stem.snowball import SnowballStemmer
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, T5EncoderModel
from vllm import LLM, SamplingParams

import process_questions_with_vllm_v2 as rag_core


qa_logger = logging.getLogger("rag_backend_server")


@dataclass
class RAGResult:
    answer: str
    search_queries: List[str]
    relevant_documents: List[str]
    num_chunks: int
    metrics: Dict[str, float]


@dataclass
class SessionState:
    messages: Deque[Dict[str, str]]
    updated_at: float
    last_result_meta: Optional[Dict[str, Any]] = None


@dataclass
class InferenceTask:
    request_id: str
    session_id: str
    user_question: str
    previous_questions: List[str]
    created_at: float
    future: asyncio.Future


@dataclass
class BackendSettings:
    model_path: str
    data_dir: Path
    embedder_dir: Path
    reranker_dir: Path
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_mem_part: float = 0.85
    max_queue_size: int = 256
    inference_workers: int = 1
    request_timeout_sec: int = 180
    history_for_retrieval: int = 3
    max_history_messages: int = 24
    history_ttl_sec: int = 43200
    gc_interval_sec: int = 300
    disable_aggregation: bool = False
    log_level: str = "INFO"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    user: Optional[str] = None
    chat_id: Optional[str] = None


class ClearHistoryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user: Optional[str] = None
    chat_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_identity(self) -> "ClearHistoryRequest":
        if not normalize_session_identity(self.user) and not normalize_session_identity(self.chat_id):
            raise ValueError("Either 'user' or 'chat_id' must be provided.")
        return self


class SessionStore:
    def __init__(self, max_history_messages: int, ttl_sec: int):
        self.max_history_messages = max_history_messages
        self.ttl_sec = ttl_sec
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def sync_from_messages(self, session_id: str, messages: List[ChatMessage]) -> None:
        filtered: List[Dict[str, str]] = []
        for msg in messages:
            if msg.role not in {"user", "assistant"}:
                continue
            content = normalize_text(msg.content)
            if not content:
                continue
            filtered.append({"role": msg.role, "content": content})
        filtered = filtered[-self.max_history_messages :]

        async with self._lock:
            now = time.time()
            self._sessions[session_id] = SessionState(
                messages=deque(filtered, maxlen=self.max_history_messages),
                updated_at=now,
            )

    async def get_previous_user_questions(self, session_id: str, current_question: str) -> List[str]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            questions = [msg["content"] for msg in state.messages if msg["role"] == "user"]
            state.updated_at = time.time()

        if questions and questions[-1] == current_question:
            questions = questions[:-1]
        return questions

    async def append_assistant_turn(self, session_id: str, user_question: str, answer: str, meta: Dict[str, Any]) -> None:
        clean_question = normalize_text(user_question)
        clean_answer = normalize_text(answer)
        now = time.time()

        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = SessionState(messages=deque(maxlen=self.max_history_messages), updated_at=now)
                self._sessions[session_id] = state
            if clean_question:
                if (not state.messages) or (state.messages[-1]["role"] != "user"):
                    state.messages.append({"role": "user", "content": clean_question})
                elif state.messages[-1]["content"] != clean_question:
                    state.messages.append({"role": "user", "content": clean_question})
            if clean_answer:
                state.messages.append({"role": "assistant", "content": clean_answer})
            state.last_result_meta = meta
            state.updated_at = now

    async def clear(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def sweep_expired(self) -> int:
        now = time.time()
        async with self._lock:
            expired = [
                key
                for key, state in self._sessions.items()
                if (now - state.updated_at) > self.ttl_sec
            ]
            for key in expired:
                del self._sessions[key]
        return len(expired)


class RAGResources:
    def __init__(self, settings: BackendSettings):
        self.settings = settings

        self.titles_of_documents: Dict[str, str] = {}
        self.chunk_list: List[Dict[str, Any]] = []
        self.prepared_chunk_list: List[Dict[str, Any]] = []
        self.num_chunks: int = 0
        self.max_input_len: int = 0

        self.russian_stemmer: Optional[SnowballStemmer] = None
        self.llm_tokenizer = None
        self.emb_tokenizer = None
        self.emb_model = None
        self.main_llm: Optional[LLM] = None
        self.reranker: Optional[LLM] = None
        self.annoy_index = None
        self.bm25_index = None
        self.llm_sampling_params: Optional[SamplingParams] = None

    def initialize(self) -> None:
        random.seed(rag_core.RANDOM_SEED)
        np.random.seed(rag_core.RANDOM_SEED)
        torch.random.manual_seed(rag_core.RANDOM_SEED)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this backend.")
        torch.cuda.random.manual_seed(rag_core.RANDOM_SEED)

        self._validate_paths()

        self.russian_stemmer = SnowballStemmer(language="russian")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.settings.model_path)
        if self.llm_tokenizer.padding_side != "left":
            self.llm_tokenizer.padding_side = "left"

        self._load_titles_and_chunks()
        self._configure_llm_sampling()

        self.main_llm = LLM(
            model=self.settings.model_path,
            gpu_memory_utilization=self.settings.gpu_mem_part,
            max_model_len=self.max_input_len + rag_core.MAX_NUMBER_OF_GENERATED_TOKENS,
            max_num_batched_tokens=max(16384, self.max_input_len + rag_core.MAX_NUMBER_OF_GENERATED_TOKENS),
            seed=rag_core.RANDOM_SEED,
        )
        qa_logger.info("Main LLM loaded: %s", self.settings.model_path)

        self.emb_tokenizer = AutoTokenizer.from_pretrained(str(self.settings.embedder_dir))
        self.emb_model = T5EncoderModel.from_pretrained(
            str(self.settings.embedder_dir), torch_dtype=torch.float32
        ).cpu()
        qa_logger.info("Embedder loaded: %s", self.settings.embedder_dir)

        tokenizer_for_reranker = AutoTokenizer.from_pretrained(str(self.settings.reranker_dir))
        max_chunk_length_for_reranker = max(
            len(tokenizer_for_reranker.tokenize(item["chunk_text"], add_special_tokens=True))
            for item in self.prepared_chunk_list
        )
        qa_logger.info("Max chunk length for reranker: %d", max_chunk_length_for_reranker)
        try:
            self.reranker = LLM(
                model=str(self.settings.reranker_dir),
                gpu_memory_utilization=0.06,
                max_model_len=round(1.8 * max_chunk_length_for_reranker),
                runner="pooling",
            )
        except Exception as first_error:
            qa_logger.warning("%s: %s", self.settings.reranker_dir, first_error)
            self.reranker = LLM(
                model=str(self.settings.reranker_dir),
                gpu_memory_utilization=0.985 - self.settings.gpu_mem_part,
                max_model_len=round(1.8 * max_chunk_length_for_reranker),
                # task="score",
            )
        qa_logger.info("Reranker loaded: %s", self.settings.reranker_dir)

        vector_dim = self.emb_model.config.d_model
        self.annoy_index = AnnoyIndex(vector_dim, "angular")
        self.annoy_index.load(str(self.settings.data_dir / "chunk_vectors_v2.ann"))
        qa_logger.info("Annoy index loaded.")

        self.bm25_index = rag_core.prepare_bm25(
            [item["chunk_text"] for item in self.prepared_chunk_list], self.russian_stemmer
        )
        qa_logger.info("BM25 index built.")

    def shutdown(self) -> None:
        rag_core.finalize_vllm()
        gc.collect()

    def _validate_paths(self) -> None:
        expected_data_files = [
            self.settings.data_dir / "chunk_vectors_v2.ann",
            self.settings.data_dir / "final_chunk_list_v2.json",
            self.settings.data_dir / "titles_of_documents.json",
        ]
        if not Path(self.settings.model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {self.settings.model_path}")
        if not self.settings.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.settings.data_dir}")
        if not self.settings.embedder_dir.is_dir():
            raise FileNotFoundError(f"Embedder directory not found: {self.settings.embedder_dir}")
        if not self.settings.reranker_dir.is_dir():
            raise FileNotFoundError(f"Reranker directory not found: {self.settings.reranker_dir}")
        for file_path in expected_data_files:
            if not file_path.is_file():
                raise FileNotFoundError(f"Data file not found: {file_path}")

    def _load_titles_and_chunks(self) -> None:
        titles_path = self.settings.data_dir / "titles_of_documents.json"
        chunks_path = self.settings.data_dir / "final_chunk_list_v2.json"

        with codecs.open(titles_path, "r", "utf-8") as fp:
            titles = json.load(fp)
        if not isinstance(titles, dict):
            raise ValueError(f"Invalid titles format in {titles_path}")
        self.titles_of_documents = titles

        with codecs.open(chunks_path, "r", "utf-8") as fp:
            chunks = json.load(fp)
        if not isinstance(chunks, list):
            raise ValueError(f"Invalid chunk list format in {chunks_path}")
        self.chunk_list = chunks

        prepared: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk {idx} must be dict")
            required_keys = {"chunk_text", "document", "chunk_index", "page_range"}
            missing = required_keys - set(chunk.keys())
            if missing:
                raise ValueError(f"Chunk {idx} missing keys: {sorted(missing)}")
            doc_name = str(chunk["document"])
            if doc_name not in self.titles_of_documents:
                raise ValueError(f"Document {doc_name} from chunk {idx} not found in {titles_path}")
            prepared.append(
                {
                    "chunk_text": self.titles_of_documents[doc_name] + "\n\n" + str(chunk["chunk_text"]),
                    "document": doc_name,
                    "chunk_index": int(chunk["chunk_index"]),
                    "page_range": chunk["page_range"],
                }
            )
        self.prepared_chunk_list = prepared
        self.num_chunks = len(self.prepared_chunk_list)
        qa_logger.info("Loaded %d chunks.", self.num_chunks)

    def _configure_llm_sampling(self) -> None:
        llm_gen_config = GenerationConfig.from_pretrained(self.settings.model_path)
        llm_config = AutoConfig.from_pretrained(self.settings.model_path)

        if not llm_gen_config.do_sample:
            llm_gen_config.do_sample = True
        llm_gen_config.max_new_tokens = rag_core.MAX_NUMBER_OF_GENERATED_TOKENS

        temperature = llm_gen_config.temperature if llm_gen_config.temperature is not None else 0.7
        top_p = llm_gen_config.top_p if llm_gen_config.top_p is not None else 0.95
        top_k = llm_gen_config.top_k if llm_gen_config.top_k is not None else -1

        self.llm_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
            max_tokens=llm_gen_config.max_new_tokens,
            seed=rag_core.RANDOM_SEED,
            skip_special_tokens=True,
        )

        chunk_lengths = [
            len(self.llm_tokenizer.tokenize(item["chunk_text"], add_special_tokens=True))
            for item in self.prepared_chunk_list
        ]
        mean_chunk_length = int(np.mean(chunk_lengths))
        std_chunk_length = int(np.std(chunk_lengths))
        max_chunk_length = max(chunk_lengths)

        max_input_len = min(
            max_chunk_length,
            mean_chunk_length + std_chunk_length,
            3 * rag_core.MAX_TOKENS_PER_CHUNK,
        )
        max_input_len *= rag_core.FINAL_NUM_BEST_CHUNKS * rag_core.MAX_NUMBER_OF_SEARCH_QUERIES

        max_model_len = min(
            llm_gen_config.max_new_tokens + max_input_len,
            self.llm_tokenizer.model_max_length,
            llm_config.max_position_embeddings,
        )
        self.max_input_len = max_model_len - llm_gen_config.max_new_tokens
        qa_logger.info("LLM context length: %d", max_model_len)


class RAGPipeline:
    def __init__(self, resources: RAGResources, settings: BackendSettings):
        self.resources = resources
        self.settings = settings

    @staticmethod
    def _deduplicate_search_queries(raw_search_response: str, emb_tokenizer) -> List[str]:
        raw_lines = [line.strip() for line in raw_search_response.split("\n")]
        normalized_lines: List[str] = []
        for line in raw_lines:
            if not line:
                continue
            while line and line[0] in "-*0123456789.) ":
                line = line[1:].strip()
            if not line:
                continue
            normalized_lines.append(rag_core.reduce_long_text(line, emb_tokenizer, rag_core.MAX_TOKENS_PER_CHUNK))
            if len(normalized_lines) >= rag_core.MAX_NUMBER_OF_SEARCH_QUERIES:
                break

        unique_lines: List[str] = []
        seen = set()
        for line in normalized_lines:
            normalized = normalize_text(line)
            if normalized and normalized not in seen:
                unique_lines.append(normalized)
                seen.add(normalized)
        return unique_lines

    @staticmethod
    def _build_retrieval_question(user_question: str, previous_questions: List[str], history_for_retrieval: int) -> str:
        clean_question = normalize_text(user_question)
        if not previous_questions:
            return clean_question
        relevant_history = previous_questions[-history_for_retrieval:]
        history_text = "\n".join([f"- {item}" for item in relevant_history])
        return (
            "Ниже приведены предыдущие вопросы пользователя в этом же диалоге.\n"
            f"{history_text}\n\n"
            f"Текущий вопрос пользователя:\n{clean_question}"
        )

    def answer_question(self, user_question: str, previous_questions: Optional[List[str]] = None) -> RAGResult:
        total_start = time.perf_counter()
        metrics: Dict[str, float] = {}

        if previous_questions is None:
            previous_questions = []
        clean_user_question = normalize_text(user_question)
        if not clean_user_question:
            raise ValueError("User question is empty.")

        retrieval_question = self._build_retrieval_question(
            clean_user_question,
            previous_questions,
            self.settings.history_for_retrieval,
        )
        reduced_retrieval_question = rag_core.reduce_long_text(
            retrieval_question, self.resources.emb_tokenizer, rag_core.MAX_TOKENS_PER_CHUNK
        )
        reduced_user_question = rag_core.reduce_long_text(
            clean_user_question, self.resources.emb_tokenizer, rag_core.MAX_TOKENS_PER_CHUNK
        )
        num_question_tokens = len(self.resources.llm_tokenizer.tokenize(reduced_user_question, add_special_tokens=True))

        search_start = time.perf_counter()
        prompt_for_search = rag_core.prepare_messages_for_search(
            user_question=reduced_retrieval_question,
            llm_tokenizer=self.resources.llm_tokenizer,
        )
        raw_search_response = rag_core.generate_answer(
            input_prompt=prompt_for_search,
            large_language_model=self.resources.main_llm,
            llm_tokenizer=self.resources.llm_tokenizer,
            sampling_params=self.resources.llm_sampling_params,
        )
        search_queries = self._deduplicate_search_queries(raw_search_response, self.resources.emb_tokenizer)
        metrics["search_generation_sec"] = round(time.perf_counter() - search_start, 6)

        if not search_queries:
            metrics["total_sec"] = round(time.perf_counter() - total_start, 6)
            return RAGResult(
                answer=rag_core.INSUFFICIENT_INFORMATION_ANSWER,
                search_queries=[],
                relevant_documents=[],
                num_chunks=0,
                metrics=metrics,
            )

        for query in search_queries:
            dist = rag_core.calculate_distance_between_texts(
                reference=rag_core.MEANINGLESS_REQUEST_ANSWER,
                hypothesis=query,
            )
            if dist <= rag_core.LEXICAL_DISTANCE_THRESHOLD:
                metrics["total_sec"] = round(time.perf_counter() - total_start, 6)
                return RAGResult(
                    answer=rag_core.INSUFFICIENT_INFORMATION_ANSWER,
                    search_queries=search_queries,
                    relevant_documents=[],
                    num_chunks=0,
                    metrics=metrics,
                )

        retrieval_start = time.perf_counter()
        union_of_relevant_indices: Dict[int, float] = {}
        for search_query in search_queries + [reduced_retrieval_question]:
            indices_from_vector_search = rag_core.find_relevant_chunks_with_vector_search(
                user_question=search_query,
                sent_emb_tokenizer=self.resources.emb_tokenizer,
                sent_embedder=self.resources.emb_model,
                vector_db=self.resources.annoy_index,
            )
            indices_from_bm25 = rag_core.find_relevant_chunks_with_bm25_search(
                user_question=search_query,
                stemmer=self.resources.russian_stemmer,
                bm25_db=self.resources.bm25_index,
                num_chunks=self.resources.num_chunks,
            )
            united_indices = sorted(set(indices_from_vector_search) | set(indices_from_bm25))
            if not united_indices:
                continue

            rerank_start = time.perf_counter()
            reranked = rag_core.rerank_chunks(
                user_question=search_query,
                selected_chunks=united_indices,
                all_chunks=self.resources.prepared_chunk_list,
                reranker=self.resources.reranker,
                num_best=rag_core.FINAL_NUM_BEST_CHUNKS,
            )
            metrics.setdefault("rerank_sec", 0.0)
            metrics["rerank_sec"] += round(time.perf_counter() - rerank_start, 6)
            for chunk_idx, chunk_score in reranked.items():
                if chunk_idx in union_of_relevant_indices:
                    union_of_relevant_indices[chunk_idx] = max(union_of_relevant_indices[chunk_idx], chunk_score)
                else:
                    union_of_relevant_indices[chunk_idx] = chunk_score

        metrics["retrieval_sec"] = round(time.perf_counter() - retrieval_start, 6)
        num_chunks_before_reduction = len(union_of_relevant_indices)
        if num_chunks_before_reduction == 0:
            metrics["total_sec"] = round(time.perf_counter() - total_start, 6)
            return RAGResult(
                answer=rag_core.INSUFFICIENT_INFORMATION_ANSWER,
                search_queries=search_queries,
                relevant_documents=[],
                num_chunks=0,
                metrics=metrics,
            )

        max_context_len = round(0.9 * self.resources.max_input_len) - num_question_tokens
        prepared_context = rag_core.chunk_indices_to_context(
            selected_chunks=union_of_relevant_indices,
            all_chunks=self.resources.chunk_list,
            llm_tokenizer=self.resources.llm_tokenizer,
            max_context_len=max_context_len,
        )
        relevant_documents = sorted(list(prepared_context.keys()))
        if not relevant_documents:
            metrics["total_sec"] = round(time.perf_counter() - total_start, 6)
            return RAGResult(
                answer=rag_core.INSUFFICIENT_INFORMATION_ANSWER,
                search_queries=search_queries,
                relevant_documents=[],
                num_chunks=num_chunks_before_reduction,
                metrics=metrics,
            )

        answer_generation_start = time.perf_counter()
        prompt_for_qa = rag_core.prepare_messages_for_answering(
            user_question=reduced_user_question,
            context=prepared_context,
            doc_titles=self.resources.titles_of_documents,
            llm_tokenizer=self.resources.llm_tokenizer,
        )
        if self.settings.disable_aggregation:
            answer = rag_core.generate_answer(
                input_prompt=prompt_for_qa,
                large_language_model=self.resources.main_llm,
                llm_tokenizer=self.resources.llm_tokenizer,
                sampling_params=self.resources.llm_sampling_params,
            )
            if not normalize_text(answer):
                answer = rag_core.INSUFFICIENT_INFORMATION_ANSWER
        else:
            variants_of_qa_prompts = {prompt_for_qa}
            for _ in range(rag_core.MAX_NUMBER_OF_ANSWER_VARIANTS * 3):
                shuffled_prompt = rag_core.prepare_messages_for_answering(
                    user_question=reduced_user_question,
                    context=prepared_context,
                    doc_titles=self.resources.titles_of_documents,
                    llm_tokenizer=self.resources.llm_tokenizer,
                    shuffle_context=True,
                )
                variants_of_qa_prompts.add(shuffled_prompt)
                if len(variants_of_qa_prompts) >= rag_core.MAX_NUMBER_OF_ANSWER_VARIANTS:
                    break

            possible_answers = rag_core.generate_answers(
                input_prompts=sorted(list(variants_of_qa_prompts)),
                large_language_model=self.resources.main_llm,
                llm_tokenizer=self.resources.llm_tokenizer,
                sampling_params=self.resources.llm_sampling_params,
            )
            nonempty_answers = sorted(list(set(filter(lambda val: len(normalize_text(val)) > 0, possible_answers))))
            if len(nonempty_answers) == 0:
                answer = rag_core.INSUFFICIENT_INFORMATION_ANSWER
            elif len(nonempty_answers) == 1:
                answer = nonempty_answers[0]
            else:
                aggregation_start = time.perf_counter()
                prompt_for_aggregation = rag_core.prepare_messages_for_aggregation(
                    user_question=reduced_user_question,
                    variants_of_answers=nonempty_answers,
                    llm_tokenizer=self.resources.llm_tokenizer,
                )
                answer = rag_core.generate_answer(
                    input_prompt=prompt_for_aggregation,
                    large_language_model=self.resources.main_llm,
                    llm_tokenizer=self.resources.llm_tokenizer,
                    sampling_params=self.resources.llm_sampling_params,
                )
                metrics["aggregation_sec"] = round(time.perf_counter() - aggregation_start, 6)
                if not normalize_text(answer):
                    answer = rag_core.INSUFFICIENT_INFORMATION_ANSWER

        metrics["answer_generation_sec"] = round(time.perf_counter() - answer_generation_start, 6)
        metrics["num_search_queries"] = float(len(search_queries))
        metrics["num_chunks"] = float(num_chunks_before_reduction)
        metrics["total_sec"] = round(time.perf_counter() - total_start, 6)

        return RAGResult(
            answer=sanitize_assistant_answer(answer),
            search_queries=search_queries,
            relevant_documents=relevant_documents,
            num_chunks=num_chunks_before_reduction,
            metrics=metrics,
        )


class QueueFullError(RuntimeError):
    pass


class InferenceQueue:
    def __init__(self, pipeline: RAGPipeline, max_size: int, workers: int, request_timeout_sec: int):
        self.pipeline = pipeline
        self.request_timeout_sec = request_timeout_sec
        self._queue: asyncio.Queue[Optional[InferenceTask]] = asyncio.Queue(maxsize=max_size)
        self._workers: List[asyncio.Task] = []
        self._workers_num = workers

    async def start(self) -> None:
        for idx in range(self._workers_num):
            task = asyncio.create_task(self._worker_loop(idx), name=f"inference-worker-{idx}")
            self._workers.append(task)

    async def stop(self) -> None:
        for _ in self._workers:
            await self._queue.put(None)
        for task in self._workers:
            with suppress(asyncio.CancelledError):
                await task
        self._workers.clear()

    async def submit(
        self,
        request_id: str,
        session_id: str,
        user_question: str,
        previous_questions: List[str],
    ) -> RAGResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        task = InferenceTask(
            request_id=request_id,
            session_id=session_id,
            user_question=user_question,
            previous_questions=previous_questions,
            created_at=time.time(),
            future=future,
        )
        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull as err:
            raise QueueFullError("Inference queue is full") from err

        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout_sec)
        except asyncio.TimeoutError as err:
            if not future.done():
                future.cancel()
            raise TimeoutError("Inference timeout") from err

    async def _worker_loop(self, worker_index: int) -> None:
        while True:
            task = await self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            try:
                started = time.time()
                result = await asyncio.to_thread(
                    self.pipeline.answer_question,
                    task.user_question,
                    task.previous_questions,
                )
                if not task.future.cancelled():
                    task.future.set_result(result)
                qa_logger.info(
                    "worker=%d request_id=%s session_id=%s queued_sec=%.3f infer_sec=%.3f",
                    worker_index,
                    task.request_id,
                    task.session_id,
                    max(0.0, started - task.created_at),
                    max(0.0, time.time() - started),
                )
            except Exception as err:
                if not task.future.cancelled():
                    task.future.set_exception(err)
            finally:
                self._queue.task_done()


@dataclass
class BackendContext:
    settings: BackendSettings
    resources: RAGResources
    pipeline: RAGPipeline
    session_store: SessionStore
    inference_queue: InferenceQueue
    gc_task: Optional[asyncio.Task]
    ready: bool


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split()).strip()


def normalize_session_identity(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = normalize_text(str(value))
    return normalized if normalized else None


def resolve_session_id(user: Optional[str], chat_id: Optional[str]) -> str:
    normalized_user = normalize_session_identity(user)
    if normalized_user:
        return normalized_user
    normalized_chat_id = normalize_session_identity(chat_id)
    if normalized_chat_id:
        return normalized_chat_id
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Either 'user' or 'chat_id' must be provided.",
    )


def extract_latest_user_question(messages: List[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            question = normalize_text(msg.content)
            if question:
                return question
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="No non-empty message with role='user' found.",
    )


def sanitize_assistant_answer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return rag_core.INSUFFICIENT_INFORMATION_ANSWER

    think_open = "<think>"
    think_close = "</think>"
    while think_open in text and think_close in text:
        start = text.find(think_open)
        end = text.find(think_close, start)
        if end < 0:
            break
        text = (text[:start] + text[end + len(think_close) :]).strip()

    if think_open in text and think_close not in text:
        text = text.split(think_open, 1)[0].strip()

    return text if text else rag_core.INSUFFICIENT_INFORMATION_ANSWER


def sse_chunk_payload(
    completion_id: str,
    model_name: str,
    created: int,
    delta_content: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    delta: Dict[str, str] = {}
    if delta_content is not None:
        delta["content"] = delta_content

    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def split_for_sse_stream(text: str, chunk_size: int = 140) -> List[str]:
    normalized = text or ""
    if not normalized:
        return [""]
    pieces: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        pieces.append(normalized[start:end])
        start = end
    return pieces


async def build_sse_response(answer: str, model_name: str) -> Any:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    async def event_gen():
        for piece in split_for_sse_stream(answer):
            payload = sse_chunk_payload(
                completion_id=completion_id,
                model_name=model_name,
                created=created,
                delta_content=piece,
                finish_reason=None,
            )
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\\n\\n"
            await asyncio.sleep(0)

        final_payload = sse_chunk_payload(
            completion_id=completion_id,
            model_name=model_name,
            created=created,
            delta_content=None,
            finish_reason="stop",
        )
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\\n\\n"
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def format_non_stream_response(answer: str, model_name: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
    }


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    qa_logger.setLevel(level)

    fmt_str = "%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s"
    formatter = logging.Formatter(fmt_str)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    qa_logger.addHandler(stdout_handler)

    log_file = Path(__file__).resolve().parent / "rag_backend_server.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    qa_logger.addHandler(file_handler)


async def run_gc_loop(ctx: BackendContext) -> None:
    while True:
        await asyncio.sleep(ctx.settings.gc_interval_sec)
        removed = await ctx.session_store.sweep_expired()
        if removed > 0:
            qa_logger.info("Session GC removed %d expired sessions.", removed)


def create_lifespan(settings: BackendSettings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resources = RAGResources(settings=settings)
        resources.initialize()
        pipeline = RAGPipeline(resources=resources, settings=settings)
        session_store = SessionStore(
            max_history_messages=settings.max_history_messages,
            ttl_sec=settings.history_ttl_sec,
        )
        inference_queue = InferenceQueue(
            pipeline=pipeline,
            max_size=settings.max_queue_size,
            workers=settings.inference_workers,
            request_timeout_sec=settings.request_timeout_sec,
        )
        await inference_queue.start()

        ctx = BackendContext(
            settings=settings,
            resources=resources,
            pipeline=pipeline,
            session_store=session_store,
            inference_queue=inference_queue,
            gc_task=None,
            ready=True,
        )
        ctx.gc_task = asyncio.create_task(run_gc_loop(ctx), name="session-gc")
        app.state.ctx = ctx

        qa_logger.info("RAG backend is ready.")
        try:
            yield
        finally:
            if ctx.gc_task is not None:
                ctx.gc_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ctx.gc_task
            await inference_queue.stop()
            resources.shutdown()
            qa_logger.info("RAG backend stopped.")

    return lifespan


def create_app(settings: BackendSettings) -> FastAPI:
    app = FastAPI(title="RRNCB RAG Backend", lifespan=create_lifespan(settings))

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok", "ts": int(time.time())}

    @app.get("/readyz")
    async def readyz() -> Dict[str, Any]:
        ctx: BackendContext = app.state.ctx
        if not ctx.ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service is not ready")
        return {"status": "ready", "ts": int(time.time())}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        ctx: BackendContext = app.state.ctx
        request_id = uuid.uuid4().hex
        session_id = resolve_session_id(request.user, request.chat_id)
        user_question = extract_latest_user_question(request.messages)

        history_len = sum(1 for item in request.messages if item.role in {"user", "assistant"})
        if history_len > 1:
            await ctx.session_store.sync_from_messages(session_id, request.messages)
        previous_questions = await ctx.session_store.get_previous_user_questions(session_id, user_question)

        start = time.perf_counter()
        try:
            result = await ctx.inference_queue.submit(
                request_id=request_id,
                session_id=session_id,
                user_question=user_question,
                previous_questions=previous_questions,
            )
        except QueueFullError as err:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Inference queue is full. Please retry later.",
            ) from err
        except TimeoutError as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference timeout.",
            ) from err
        except HTTPException:
            raise
        except Exception as err:
            qa_logger.exception("request_id=%s session_id=%s error=%s", request_id, session_id, err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error.",
            ) from err

        await ctx.session_store.append_assistant_turn(
            session_id=session_id,
            user_question=user_question,
            answer=result.answer,
            meta={
                "request_id": request_id,
                "num_chunks": result.num_chunks,
                "relevant_documents": result.relevant_documents,
                "search_queries": result.search_queries,
                "metrics": result.metrics,
            },
        )
        qa_logger.info(
            "request_id=%s session_id=%s total_latency_sec=%.3f search_queries=%d chunks=%d metrics=%s",
            request_id,
            session_id,
            time.perf_counter() - start,
            len(result.search_queries),
            result.num_chunks,
            result.metrics,
        )

        if request.stream:
            return await build_sse_response(result.answer, request.model)

        return JSONResponse(content=format_non_stream_response(result.answer, request.model))

    @app.post("/v1/chat/completions/clear_history")
    async def clear_history(request: ClearHistoryRequest):
        ctx: BackendContext = app.state.ctx
        session_id = resolve_session_id(request.user, request.chat_id)
        cleared = await ctx.session_store.clear(session_id)
        return {"status": "ok", "cleared": cleared, "session_id": session_id}

    @app.post("/v1/chat/completions/clear_history/")
    async def clear_history_slash(request: ClearHistoryRequest):
        return await clear_history(request)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_path", type=str, default=os.getenv("MODEL_PATH"))
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=os.getenv("DATA_DIR"))
    parser.add_argument("--embedder", dest="embedder_dir", type=str, default=os.getenv("EMBEDDER_DIR"))
    parser.add_argument("--reranker", dest="reranker_dir", type=str, default=os.getenv("RERANKER_DIR"))
    parser.add_argument("--host", dest="host", type=str, default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", dest="port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--gpu", dest="gpu_mem_part", type=float, default=float(os.getenv("GPU_MEM_PART", "0.85")))
    parser.add_argument(
        "--max-queue-size",
        dest="max_queue_size",
        type=int,
        default=int(os.getenv("MAX_QUEUE_SIZE", "256")),
    )
    parser.add_argument(
        "--inference-workers",
        dest="inference_workers",
        type=int,
        default=int(os.getenv("INFERENCE_WORKERS", "1")),
    )
    parser.add_argument(
        "--request-timeout",
        dest="request_timeout_sec",
        type=int,
        default=int(os.getenv("REQUEST_TIMEOUT_SEC", "180")),
    )
    parser.add_argument(
        "--history-for-retrieval",
        dest="history_for_retrieval",
        type=int,
        default=int(os.getenv("HISTORY_FOR_RETRIEVAL", "3")),
    )
    parser.add_argument(
        "--max-history-messages",
        dest="max_history_messages",
        type=int,
        default=int(os.getenv("MAX_HISTORY_MESSAGES", "24")),
    )
    parser.add_argument(
        "--history-ttl",
        dest="history_ttl_sec",
        type=int,
        default=int(os.getenv("HISTORY_TTL_SEC", "43200")),
    )
    parser.add_argument(
        "--gc-interval",
        dest="gc_interval_sec",
        type=int,
        default=int(os.getenv("HISTORY_GC_INTERVAL_SEC", "300")),
    )
    parser.add_argument(
        "--disable-aggregation",
        dest="disable_aggregation",
        action="store_true",
    )
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> BackendSettings:
    if not args.model_path:
        raise ValueError("MODEL_PATH is required (or pass --model).")
    if not args.data_dir:
        raise ValueError("DATA_DIR is required (or pass --data-dir).")
    if not args.embedder_dir:
        raise ValueError("EMBEDDER_DIR is required (or pass --embedder).")
    if not args.reranker_dir:
        raise ValueError("RERANKER_DIR is required (or pass --reranker).")
    if not (0.1 < args.gpu_mem_part < 0.95):
        raise ValueError("GPU_MEM_PART must be in range (0.1, 0.95).")
    if args.max_queue_size < 1:
        raise ValueError("MAX_QUEUE_SIZE must be >= 1.")
    if args.inference_workers < 1:
        raise ValueError("INFERENCE_WORKERS must be >= 1.")
    if args.request_timeout_sec < 1:
        raise ValueError("REQUEST_TIMEOUT_SEC must be >= 1.")

    return BackendSettings(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        data_dir=Path(args.data_dir).expanduser().resolve(),
        embedder_dir=Path(args.embedder_dir).expanduser().resolve(),
        reranker_dir=Path(args.reranker_dir).expanduser().resolve(),
        host=args.host,
        port=args.port,
        gpu_mem_part=args.gpu_mem_part,
        max_queue_size=args.max_queue_size,
        inference_workers=args.inference_workers,
        request_timeout_sec=args.request_timeout_sec,
        history_for_retrieval=args.history_for_retrieval,
        max_history_messages=args.max_history_messages,
        history_ttl_sec=args.history_ttl_sec,
        gc_interval_sec=args.gc_interval_sec,
        disable_aggregation=args.disable_aggregation,
        log_level=args.log_level,
    )


def handle_exit(signal_num, frame):  # type: ignore[unused-arg]
    rag_core.finalize_vllm()
    sys.exit(0)


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    configure_logging(settings.log_level)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
