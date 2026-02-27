import asyncio
import json
import logging
import os
import re
import time
from math import sqrt
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import json_repair

from gpt_researcher.actions.retriever import get_retriever
from gpt_researcher.actions.web_scraping import scrape_urls
from gpt_researcher.config.config import Config
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.utils.workers import WorkerPool

from .state_controller import StateController
from .utils.llms import call_model
from .utils.views import print_agent_output


class ScrapAgent:
    """Adaptive Scrap Agent: multi-iteration search + dedup + MMR rerank."""

    ROUND_TO_TIER = {1: 3, 2: 2, 3: 1}
    ROUND_TO_LEVEL = {
        1: "Level_1_Base",
        2: "Level_2_Pro",
        3: "Level_3_Max",
    }
    DOMAIN_TO_ENGINES = {
        "tech": ["arxiv", "semantic_scholar", "google"],
        "medical": ["pubmed_central", "google", "bing"],
        "finance": ["tavily", "google", "bing"],
        "general": ["tavily", "google", "bing"],
    }
    VERTICAL_ENGINES = {"arxiv", "semantic_scholar", "pubmed_central"}
    MODEL_LEVEL_FALLBACK = {
        1: "gpt-4o-mini",
        2: "gpt-4o",
        3: "o1-preview",
    }

    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)
        state_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_state.json")
        self.state_controller = StateController(state_file=state_file)
        self._mmr_embeddings = None
        self._mmr_embeddings_error: Optional[str] = None
        self._use_embedding_mmr = self._env_truthy("SCRAP_MMR_USE_EMBEDDINGS", default=True)
        self._min_search_targets = self._env_int("SCRAP_MIN_SEARCH_TARGETS", default=2, min_value=1, max_value=8)
        self._max_search_targets = self._env_int("SCRAP_MAX_SEARCH_TARGETS", default=4, min_value=1, max_value=12)
        if self._max_search_targets < self._min_search_targets:
            self._max_search_targets = self._min_search_targets

    async def run_depth_scrap(self, draft_state: dict) -> dict:
        task = draft_state.get("task") or {}
        topic = str(draft_state.get("topic") or "").strip()
        if not topic:
            topic = "Main Research Focus"

        research_context = draft_state.get("research_context") or {}
        audit_feedback = draft_state.get("audit_feedback")
        extra_hints = draft_state.get("extra_hints")
        iteration_override = draft_state.get("iteration_index")
        query_domains = task.get("query_domains") or []
        max_results = int(task.get("scrap_max_search_results") or task.get("max_search_results_per_query") or 7)
        max_iterations = int(task.get("scrap_max_iterations") or 3)

        use_single_iteration = iteration_override is not None
        if use_single_iteration:
            single_iteration = self._normalize_iteration(iteration_override)
            iteration_plan = [single_iteration]
        else:
            planned_iterations = min(self._resolve_iterations(audit_feedback), max_iterations)
            iteration_plan = list(range(1, planned_iterations + 1))
        packets: List[dict] = []

        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "scrap_start",
                f"Running ASA Scrap loop for topic: {topic}",
                self.websocket,
            )
        else:
            print_agent_output(f"Running ASA Scrap loop for topic: {topic}", agent="SCRAP")

        try:
            for iteration in iteration_plan:
                tier_idx = self.ROUND_TO_TIER.get(iteration, 3)
                self.state_controller.set_tier("scrap", tier_idx)
                model_name = self._resolve_model_for_iteration(task, iteration)
                extra_hints_applied = self._merge_extra_hints(extra_hints, audit_feedback)
                sub_targets = await self._decompose_targets(
                    topic=topic,
                    research_context=research_context,
                    extra_hints=extra_hints_applied,
                    model_name=model_name,
                )
                engines = self._select_engines(iteration, topic)
                search_log = []
                active_engines = set()

                for target in sub_targets:
                    target_results, used_engines = await self._collect_results_for_target(
                        target=target,
                        engines=engines,
                        query_domains=query_domains,
                        max_results=max_results,
                    )
                    active_engines.update(used_engines)
                    passages = self._build_passages_from_results(target_results)
                    top_passages = await self._select_top_passages_with_mmr(
                        query=target,
                        passages=passages,
                        top_k=10,
                        time_budget_s=5.0,
                    )
                    search_log.append(
                        {
                            "target": target,
                            "extra_hints_applied": extra_hints_applied,
                            "top_10_passages": top_passages,
                        }
                    )

                packet = {
                    "iteration_index": iteration,
                    "model_level": self.ROUND_TO_LEVEL.get(iteration, "Level_1_Base"),
                    "active_engines": sorted(active_engines),
                    "search_log": search_log,
                }
                packets.append(packet)
                self.logger.info(
                    "scrap_iteration_complete",
                    extra={
                        "iteration": iteration,
                        "tier": tier_idx,
                        "engines": packet["active_engines"],
                        "passage_count": sum(len(x["top_10_passages"]) for x in search_log),
                    },
                )
        finally:
            # Reset Scrap to default base tier after loop
            self.state_controller.set_tier("scrap", 3)

        final_packet = packets[-1] if packets else self._empty_packet()
        draft_text = self._build_compatible_draft(topic, final_packet)
        return {
            "draft": {topic: draft_text},
            "scrap_packet": final_packet,
            "iteration_index": final_packet.get("iteration_index", 1),
        }

    def _resolve_iterations(self, audit_feedback: dict | None) -> int:
        if not isinstance(audit_feedback, dict):
            return 1
        if audit_feedback.get("is_satisfied") is not False:
            return 1
        if audit_feedback.get("confidence_score") is None:
            return 1
        try:
            score = float(audit_feedback.get("confidence_score"))
        except (TypeError, ValueError):
            return 1
        return 3 if score < 0.7 else 2

    def _resolve_model_for_iteration(self, task: dict, iteration: int) -> str:
        explicit = task.get(f"scrap_level_{iteration}_model")
        if explicit:
            return str(explicit)
        fallback = self.MODEL_LEVEL_FALLBACK.get(iteration, "gpt-4o-mini")
        return self.state_controller.get_current_model("scrap", fallback_model=fallback)

    def _normalize_iteration(self, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 1
        return min(max(parsed, 1), 3)

    async def _decompose_targets(
        self,
        topic: str,
        research_context: dict,
        extra_hints: str,
        model_name: str,
    ) -> List[str]:
        description = str(research_context.get("description") or "")
        key_points = research_context.get("key_points") or []
        key_points_text = ", ".join(str(x) for x in key_points if str(x).strip())
        min_targets = self._min_search_targets
        max_targets = self._max_search_targets
        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are a search planner. Produce {min_targets} to {max_targets} independent search targets. "
                    f"Return JSON only as a list of strings ({min_targets}-{max_targets} items)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n"
                    f"Description: {description}\n"
                    f"Key points: {key_points_text}\n"
                    f"Extra hints: {extra_hints}\n"
                    "Constraints: convert hints into explicit search constraints. "
                    f"Output JSON list with {min_targets}-{max_targets} concise targets."
                ),
            },
        ]

        try:
            response = await call_model(prompt=prompt, model=model_name)
            parsed = self._parse_targets(response)
            if min_targets <= len(parsed) <= max_targets:
                return parsed
        except Exception as exc:
            self.logger.warning(f"Failed to decompose targets with model {model_name}: {exc}")

        return self._fallback_targets(topic, extra_hints)

    def _parse_targets(self, response: Any) -> List[str]:
        if isinstance(response, list):
            return self._clean_targets(response)

        if isinstance(response, dict):
            for key in ("targets", "queries", "search_targets"):
                if key in response and isinstance(response[key], list):
                    return self._clean_targets(response[key])

        if isinstance(response, str):
            try:
                repaired = json_repair.loads(response)
                return self._parse_targets(repaired)
            except Exception:
                pass
            match = re.findall(r'"([^"]+)"', response)
            if match:
                return self._clean_targets(match)
        return []

    def _clean_targets(self, raw_targets: Iterable[Any]) -> List[str]:
        cleaned = []
        seen = set()
        for item in raw_targets:
            text = re.sub(r"\s+", " ", str(item or "").strip())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
            if len(cleaned) >= self._max_search_targets:
                break
        return cleaned

    def _fallback_targets(self, topic: str, extra_hints: str) -> List[str]:
        hint_tail = f" | constraints: {extra_hints}" if extra_hints else ""
        return [
            f"{topic} key facts and latest updates{hint_tail}",
            f"{topic} opposing viewpoints and contradictory evidence{hint_tail}",
            f"{topic} quantitative data and primary sources{hint_tail}",
        ]

    def _merge_extra_hints(self, extra_hints: str | None, audit_feedback: dict | None) -> str:
        parts = []
        if isinstance(extra_hints, str) and extra_hints.strip():
            parts.append(extra_hints.strip())
        if isinstance(audit_feedback, dict):
            for key in ("new_query_suggestion", "instruction"):
                value = audit_feedback.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        return " ; ".join(parts)

    def _select_engines(self, iteration: int, topic: str) -> List[str]:
        if iteration <= 1:
            return ["tavily"]
        domain = self._classify_domain(topic)
        return self.DOMAIN_TO_ENGINES.get(domain, self.DOMAIN_TO_ENGINES["general"])

    def _classify_domain(self, text: str) -> str:
        lowered = text.lower()
        tech_keywords = {"llm", "model", "software", "chip", "gpu", "ai", "cloud", "open-source", "programming"}
        medical_keywords = {"medical", "clinical", "drug", "patient", "disease", "hospital", "pubmed", "health"}
        finance_keywords = {"finance", "stock", "revenue", "earnings", "fiscal", "market", "bank", "valuation"}
        if any(k in lowered for k in medical_keywords):
            return "medical"
        if any(k in lowered for k in tech_keywords):
            return "tech"
        if any(k in lowered for k in finance_keywords):
            return "finance"
        return "general"

    async def _collect_results_for_target(
        self,
        target: str,
        engines: List[str],
        query_domains: List[str],
        max_results: int,
    ) -> Tuple[List[dict], List[str]]:
        all_results: List[dict] = []
        used_engines = set()

        for engine in engines:
            results, failed = await self._search_with_engine(
                engine=engine,
                query=target,
                query_domains=query_domains,
                max_results=max_results,
            )
            if results:
                all_results.extend(results)
                used_engines.add(engine)
            if failed and engine in self.VERTICAL_ENGINES:
                for fallback_engine in ("google", "bing"):
                    fallback_results, _ = await self._search_with_engine(
                        engine=fallback_engine,
                        query=target,
                        query_domains=query_domains,
                        max_results=max_results,
                    )
                    if fallback_results:
                        all_results.extend(fallback_results)
                        used_engines.add(fallback_engine)

        deduped = self._dedupe_search_results(all_results)
        enriched = await self._materialize_search_results(deduped)
        return enriched, sorted(used_engines)

    async def _search_with_engine(
        self,
        engine: str,
        query: str,
        query_domains: List[str],
        max_results: int,
    ) -> Tuple[List[dict], bool]:
        retriever_cls = get_retriever(engine)
        if retriever_cls is None:
            return [], True
        try:
            retriever = self._instantiate_retriever(retriever_cls, query, query_domains)
            results = await asyncio.to_thread(retriever.search, max_results=max_results)
            normalized = self._normalize_search_results(results, source_engine=engine)
            return normalized, False
        except Exception as exc:
            self.logger.warning(f"Search engine {engine} failed for query '{query}': {exc}")
            return [], True

    def _instantiate_retriever(self, retriever_cls, query: str, query_domains: List[str]):
        attempts = [
            {"query": query, "headers": self.headers, "query_domains": query_domains},
            {"query": query, "query_domains": query_domains},
            {"query": query},
        ]
        positional_attempts = [
            (query, self.headers, query_domains),
            (query, query_domains),
            (query,),
        ]
        for kwargs in attempts:
            try:
                return retriever_cls(**kwargs)
            except TypeError:
                continue
        for args in positional_attempts:
            try:
                return retriever_cls(*args)
            except TypeError:
                continue
        return retriever_cls(query)

    def _normalize_search_results(self, results: Any, source_engine: str) -> List[dict]:
        normalized = []
        if not isinstance(results, list):
            return normalized

        for item in results:
            if not isinstance(item, dict):
                continue
            href = item.get("href") or item.get("url")
            if not href:
                continue
            normalized.append(
                {
                    "href": str(href),
                    "title": str(item.get("title") or ""),
                    "body": str(item.get("body") or item.get("content") or item.get("raw_content") or ""),
                    "source_engine": source_engine,
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                }
            )
        return normalized

    def _dedupe_search_results(self, results: List[dict]) -> List[dict]:
        deduped = {}
        for item in results:
            url = item.get("href")
            if not url:
                continue
            key = self._normalize_url(url)
            if key not in deduped:
                deduped[key] = item
        return list(deduped.values())

    def _normalize_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
            normalized = parsed._replace(
                fragment="",
                query=query,
                path=parsed.path.rstrip("/") or "/",
            )
            return urlunparse(normalized)
        except Exception:
            return url

    async def _materialize_search_results(self, search_results: List[dict]) -> List[dict]:
        urls = [item["href"] for item in search_results if item.get("href")]
        scraped_map = {}
        if urls:
            scraped_data = await self._scrape_urls(urls)
            for item in scraped_data:
                if isinstance(item, dict) and item.get("url"):
                    scraped_map[self._normalize_url(item["url"])] = item

        enriched = []
        for item in search_results:
            url = item.get("href")
            norm_url = self._normalize_url(url)
            scraped = scraped_map.get(norm_url, {})
            raw_content = str(scraped.get("raw_content") or item.get("body") or "")
            if not raw_content.strip():
                continue
            metadata = {}
            metadata.update(item.get("metadata") or {})
            metadata.setdefault("publish_date", scraped.get("publish_date"))
            metadata.setdefault("author", scraped.get("author"))
            enriched.append(
                {
                    "url": url,
                    "title": scraped.get("title") or item.get("title") or "",
                    "raw_content": raw_content,
                    "source_engine": item.get("source_engine"),
                    "metadata": metadata,
                }
            )
        return enriched

    async def _scrape_urls(self, urls: List[str]) -> List[dict]:
        cfg = Config()
        worker_pool = WorkerPool(cfg.max_scraper_workers, cfg.scraper_rate_limit_delay)
        try:
            scraped_data, _ = await scrape_urls(urls, cfg, worker_pool)
            return scraped_data
        except Exception as exc:
            self.logger.warning(f"URL scraping failed: {exc}")
            return []
        finally:
            worker_pool.executor.shutdown(wait=False, cancel_futures=True)

    def _build_passages_from_results(self, results: List[dict]) -> List[dict]:
        passages = []
        for item in results:
            url = item.get("url")
            content = str(item.get("raw_content") or "")
            metadata = item.get("metadata") or {}
            source_engine = item.get("source_engine")
            for chunk in self._split_to_passages(content, min_chars=200, max_chars=500):
                passages.append(
                    {
                        "content": chunk,
                        "source_url": url,
                        "metadata": {
                            "publish_date": metadata.get("publish_date"),
                            "author": metadata.get("author"),
                        },
                        "source_engine": source_engine,
                    }
                )
        return passages

    def _split_to_passages(self, text: str, min_chars: int = 200, max_chars: int = 500) -> List[str]:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if not compact:
            return []
        if len(compact) <= max_chars:
            return [compact]

        sentences = re.split(r"(?<=[.!?。！？])\s+", compact)
        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(" ".join(current).strip())
                    current, current_len = [], 0
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i : i + max_chars].strip())
                continue

            projected = current_len + len(sentence) + (1 if current else 0)
            if projected > max_chars and current:
                chunks.append(" ".join(current).strip())
                current, current_len = [sentence], len(sentence)
            else:
                current.append(sentence)
                current_len = projected

        if current:
            chunks.append(" ".join(current).strip())

        merged = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            if len(chunk) < min_chars and len(merged[-1]) + len(chunk) + 1 <= max_chars:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)
        return merged

    async def _select_top_passages_with_mmr(
        self,
        query: str,
        passages: List[dict],
        top_k: int = 10,
        lambda_param: float = 0.7,
        time_budget_s: float = 5.0,
    ) -> List[dict]:
        if not passages:
            return []

        start = time.perf_counter()
        if self._use_embedding_mmr:
            selected = await self._select_top_passages_with_embedding_mmr(
                query=query,
                passages=passages,
                top_k=top_k,
                lambda_param=lambda_param,
                start_time=start,
                time_budget_s=time_budget_s,
            )
            if selected:
                return selected

        query_vec = self._text_to_vector(query)
        passage_vecs = [self._text_to_vector(p["content"]) for p in passages]
        relevance = [self._cosine_similarity(query_vec, vec) for vec in passage_vecs]

        chosen_indices: List[int] = []
        candidates = set(range(len(passages)))

        while candidates and len(chosen_indices) < min(top_k, len(passages)):
            if time.perf_counter() - start > time_budget_s:
                return self._fallback_rank(passages, relevance, chosen_indices, top_k)

            if not chosen_indices:
                best = max(candidates, key=lambda idx: relevance[idx])
            else:
                best = None
                best_score = float("-inf")
                for idx in candidates:
                    diversity_penalty = max(
                        self._cosine_similarity(passage_vecs[idx], passage_vecs[selected])
                        for selected in chosen_indices
                    )
                    score = (lambda_param * relevance[idx]) - ((1.0 - lambda_param) * diversity_penalty)
                    if score > best_score:
                        best_score = score
                        best = idx
                if best is None:
                    break
            chosen_indices.append(best)
            candidates.remove(best)

        top_passages = []
        for idx in chosen_indices:
            passage = passages[idx]
            top_passages.append(
                {
                    "content": passage["content"],
                    "source_url": passage["source_url"],
                    "metadata": passage.get("metadata") or {},
                }
            )
        return top_passages[:top_k]

    async def _select_top_passages_with_embedding_mmr(
        self,
        query: str,
        passages: List[dict],
        top_k: int,
        lambda_param: float,
        start_time: float,
        time_budget_s: float,
    ) -> List[dict]:
        embeddings = self._get_mmr_embeddings()
        if embeddings is None:
            return []

        remaining = max(0.0, time_budget_s - (time.perf_counter() - start_time))
        if remaining <= 0.05:
            return []

        max_embed_passages = self._env_int("SCRAP_MMR_MAX_EMBED_PASSAGES", default=80, min_value=10, max_value=500)
        candidates = passages
        if len(passages) > max_embed_passages:
            query_vec = self._text_to_vector(query)
            passage_vecs = [self._text_to_vector(p["content"]) for p in passages]
            relevance = [self._cosine_similarity(query_vec, vec) for vec in passage_vecs]
            ranked = sorted(range(len(passages)), key=lambda idx: relevance[idx], reverse=True)[:max_embed_passages]
            candidates = [passages[idx] for idx in ranked]

        texts = [query] + [p.get("content") or "" for p in candidates]
        texts = [str(t) for t in texts]
        if not texts[0].strip():
            return []

        remaining = max(0.0, time_budget_s - (time.perf_counter() - start_time))
        if remaining <= 0.05:
            return []

        try:
            vectors = await asyncio.wait_for(
                asyncio.to_thread(embeddings.embed_documents, texts),
                timeout=remaining,
            )
        except Exception as exc:
            self._use_embedding_mmr = False
            self._mmr_embeddings_error = f"embedding_call_failed:{type(exc).__name__}"
            self.logger.debug("Embedding MMR failed, falling back to BOW MMR: %s", exc)
            return []

        if not vectors or len(vectors) != len(texts):
            return []

        query_vec = vectors[0]
        passage_vecs = vectors[1:]
        if not passage_vecs:
            return []

        norms = [self._l2_norm(query_vec)] + [self._l2_norm(v) for v in passage_vecs]
        if norms[0] == 0.0:
            return []

        relevance = [
            self._cosine_similarity_dense(query_vec, norms[0], vec, norms[i + 1])
            for i, vec in enumerate(passage_vecs)
        ]

        chosen_indices: List[int] = []
        remaining_indices = set(range(len(candidates)))

        def sim(i: int, j: int) -> float:
            return self._cosine_similarity_dense(
                passage_vecs[i],
                norms[i + 1],
                passage_vecs[j],
                norms[j + 1],
            )

        while remaining_indices and len(chosen_indices) < min(top_k, len(candidates)):
            if time.perf_counter() - start_time > time_budget_s:
                return self._fallback_rank(candidates, relevance, chosen_indices, top_k)

            if not chosen_indices:
                best = max(remaining_indices, key=lambda idx: relevance[idx])
            else:
                best = None
                best_score = float("-inf")
                for idx in remaining_indices:
                    diversity_penalty = max(sim(idx, selected) for selected in chosen_indices)
                    score = (lambda_param * relevance[idx]) - ((1.0 - lambda_param) * diversity_penalty)
                    if score > best_score:
                        best_score = score
                        best = idx
                if best is None:
                    break

            chosen_indices.append(best)
            remaining_indices.remove(best)

        top_passages = []
        for idx in chosen_indices[:top_k]:
            passage = candidates[idx]
            top_passages.append(
                {
                    "content": passage["content"],
                    "source_url": passage["source_url"],
                    "metadata": passage.get("metadata") or {},
                }
            )
        return top_passages

    def _fallback_rank(
        self,
        passages: List[dict],
        relevance: List[float],
        chosen_indices: List[int],
        top_k: int,
    ) -> List[dict]:
        remaining = [idx for idx in range(len(passages)) if idx not in chosen_indices]
        remaining.sort(key=lambda idx: relevance[idx], reverse=True)
        picked = chosen_indices + remaining
        result = []
        seen_snippets = set()
        for idx in picked:
            content = passages[idx]["content"]
            key = re.sub(r"\s+", " ", content.lower())[:180]
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            result.append(
                {
                    "content": content,
                    "source_url": passages[idx]["source_url"],
                    "metadata": passages[idx].get("metadata") or {},
                }
            )
            if len(result) >= top_k:
                break
        return result

    def _text_to_vector(self, text: str) -> Counter:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower())
        return Counter(tokens)

    def _cosine_similarity(self, v1: Counter, v2: Counter) -> float:
        if not v1 or not v2:
            return 0.0
        common = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[token] * v2[token] for token in common)
        norm1 = sum(value * value for value in v1.values()) ** 0.5
        norm2 = sum(value * value for value in v2.values()) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _l2_norm(self, vec: Any) -> float:
        if not vec:
            return 0.0
        try:
            return sqrt(sum(float(x) * float(x) for x in vec))
        except Exception:
            return 0.0

    def _cosine_similarity_dense(self, v1: Any, n1: float, v2: Any, n2: float) -> float:
        if not v1 or not v2 or n1 == 0.0 or n2 == 0.0:
            return 0.0
        try:
            dot = 0.0
            for a, b in zip(v1, v2):
                dot += float(a) * float(b)
            return dot / (n1 * n2)
        except Exception:
            return 0.0

    def _env_truthy(self, key: str, default: bool = False) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_int(self, key: str, default: int, min_value: int, max_value: int) -> int:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            parsed = int(str(raw).strip())
        except ValueError:
            return default
        return max(min_value, min(max_value, parsed))

    def _get_mmr_embeddings(self):
        if self._mmr_embeddings_error is not None:
            return None
        if self._mmr_embeddings is not None:
            return self._mmr_embeddings
        try:
            cfg = Config()
            provider = getattr(cfg, "embedding_provider", None)
            model = getattr(cfg, "embedding_model", None)
            kwargs = getattr(cfg, "embedding_kwargs", {}) or {}
            if not provider or not model:
                self._mmr_embeddings_error = "missing_embedding_config"
                return None
            self._mmr_embeddings = Memory(provider, model, **kwargs).get_embeddings()
            return self._mmr_embeddings
        except Exception as exc:
            self._mmr_embeddings_error = str(exc)
            self.logger.debug("MMR embedding init failed, falling back to BOW MMR: %s", exc)
            return None

    def _empty_packet(self) -> dict:
        return {
            "iteration_index": 1,
            "model_level": "Level_1_Base",
            "active_engines": [],
            "search_log": [],
        }

    def _build_compatible_draft(self, topic: str, scrap_packet: dict) -> str:
        lines = [f"### {topic}", "", "#### Curated Objective Passages"]
        counter = 1
        for target_log in scrap_packet.get("search_log", []):
            target = target_log.get("target", "")
            lines.append("")
            lines.append(f"Target: {target}")
            for passage in target_log.get("top_10_passages", []):
                content = str(passage.get("content") or "").strip()
                source = str(passage.get("source_url") or "").strip()
                if not content:
                    continue
                lines.append(f"{counter}. {content}")
                if source:
                    lines.append(f"   Source: {source}")
                counter += 1
                if counter > 15:
                    break
            if counter > 15:
                break

        if counter == 1:
            lines.append("")
            lines.append("No reliable passages were retrieved in this run.")
        return "\n".join(lines).strip()
