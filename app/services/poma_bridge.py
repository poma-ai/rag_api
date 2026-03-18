import json
import os
import traceback
from typing import Any, Iterable

from langchain_core.documents import Document

from app.config import (
    logger,
    POMA_STORE_DIR,
    POMA_TIMEOUT_SECONDS,
    POMA_POLL_INTERVAL_SECONDS,
    POMA_INGEST_METHOD,
)


class PomaTooManyJobsError(RuntimeError):
    def __init__(
        self,
        message: str = "Too many jobs",
        *,
        upstream_status: int | None = None,
        upstream_detail: str | None = None,
        upstream_code: str | int | None = None,
    ) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status
        self.upstream_detail = upstream_detail
        self.upstream_code = upstream_code


class PomaRetryableUpstreamError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        upstream_status: int | None = None,
        upstream_detail: str | None = None,
        upstream_code: str | int | None = None,
    ) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status
        self.upstream_detail = upstream_detail
        self.upstream_code = upstream_code


class PomaJobFailedError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        upstream_status: int | None = None,
        job_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status
        self.job_status = job_status


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_error_message(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("detail", "message", "error", "errors"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list) and value:
                return "; ".join(str(v) for v in value)
    return str(obj)


def _extract_upstream_code(obj: Any) -> str | int | None:
    if not isinstance(obj, dict):
        return None
    code = obj.get("code")
    if isinstance(code, (str, int)):
        return code
    return None


def _parse_json_dict_if_possible(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_upstream_detail_and_code(obj: Any) -> tuple[str | None, str | int | None]:
    if obj is None:
        return None, None

    if isinstance(obj, dict):
        detail = _extract_error_message(obj)
        return (detail if detail else None), _extract_upstream_code(obj)

    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return None, None
        parsed = _parse_json_dict_if_possible(text)
        if parsed is not None:
            return _extract_upstream_detail_and_code(parsed)
        return text, None

    return str(obj), None


def _extract_upstream_status_from_error(err: Exception) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        status = _safe_int(getattr(err, attr, None))
        if status is not None:
            return status

    response = getattr(err, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            status = _safe_int(getattr(response, attr, None))
            if status is not None:
                return status

    return None


def _extract_upstream_status_from_response(resp: Any) -> int | None:
    if not isinstance(resp, dict):
        return None

    for key in ("status_code", "http_status", "status", "code"):
        status = _safe_int(resp.get(key))
        if status is not None:
            return status

    return None


def _is_too_many_jobs_message(message: str) -> bool:
    lowered = message.lower()
    return "too many jobs" in lowered


def _is_retryable_create_job_failure_message(message: str) -> bool:
    lowered = message.lower()
    return "failed to create job" in lowered


def _is_terminal_job_failure_message(message: str) -> bool:
    lowered = message.lower()
    return (
        "job failed" in lowered
        or "job cancelled" in lowered
        or "job canceled" in lowered
    )


def _is_terminal_job_status(status: str) -> bool:
    return status in {"failed", "error", "cancelled", "canceled"}


def _get_http_response_from_exception(err: Exception) -> tuple[int | None, str]:
    """Extract HTTP status code and response body from an exception (e.g. httpx.HTTPStatusError or POMA RemoteServerError)."""
    ex: Exception | None = err
    while ex is not None:
        if hasattr(ex, "response") and ex.response is not None:
            resp = ex.response
            try:
                status: int | None = getattr(resp, "status_code", None)
            except Exception:
                status = None
            try:
                body = resp.text
            except Exception:
                try:
                    body = (getattr(resp, "content", None) or b"").decode(
                        "utf-8", errors="replace"
                    )
                except Exception:
                    body = repr(resp)
            return status, body
        ex = getattr(ex, "__cause__", None)
    return None, ""


def _extract_upstream_detail_and_code_from_error(
    err: Exception,
) -> tuple[str | None, str | int | None]:
    detail, code = _extract_upstream_detail_and_code(getattr(err, "response_body", None))
    if detail is not None or code is not None:
        return detail, code

    _, body = _get_http_response_from_exception(err)
    return _extract_upstream_detail_and_code(body)


def _raise_if_poma_too_many_jobs_error(err: Exception) -> None:
    upstream_status = _extract_upstream_status_from_error(err)
    upstream_detail, upstream_code = _extract_upstream_detail_and_code_from_error(err)
    match_text = upstream_detail or str(err)
    if _is_too_many_jobs_message(match_text):
        raise PomaTooManyJobsError(
            "Too many jobs",
            upstream_status=upstream_status,
            upstream_detail=upstream_detail,
            upstream_code=upstream_code,
        ) from err


def _raise_if_poma_too_many_jobs_response(resp: Any) -> None:
    if not isinstance(resp, dict):
        return

    message, upstream_code = _extract_upstream_detail_and_code(resp)
    if not _is_too_many_jobs_message(message):
        return

    upstream_status = _extract_upstream_status_from_response(resp)

    raise PomaTooManyJobsError(
        "Too many jobs",
        upstream_status=upstream_status,
        upstream_detail=message,
        upstream_code=upstream_code,
    )


def _is_retryable_poma_create_job_failure(
    *, upstream_status: int | None, upstream_detail: str | None
) -> bool:
    if not upstream_detail or not _is_retryable_create_job_failure_message(upstream_detail):
        return False

    # POMA currently returns a generic "Failed to create job" for transient capacity/job creation
    # problems. Treat these as retryable regardless of whether upstream used 400 or 5xx.
    return upstream_status in {None, 400, 403, 429, 500, 502, 503, 504}


def _raise_if_poma_retryable_create_job_error(err: Exception) -> None:
    upstream_status = _extract_upstream_status_from_error(err)
    upstream_detail, upstream_code = _extract_upstream_detail_and_code_from_error(err)
    if not _is_retryable_poma_create_job_failure(
        upstream_status=upstream_status,
        upstream_detail=upstream_detail,
    ):
        return

    raise PomaRetryableUpstreamError(
        "POMA temporarily failed to create a chunking job",
        upstream_status=upstream_status,
        upstream_detail=upstream_detail,
        upstream_code=upstream_code,
    ) from err


def _raise_if_poma_retryable_create_job_response(resp: Any) -> None:
    if not isinstance(resp, dict):
        return

    upstream_detail, upstream_code = _extract_upstream_detail_and_code(resp)
    upstream_status = _extract_upstream_status_from_response(resp)
    if not _is_retryable_poma_create_job_failure(
        upstream_status=upstream_status,
        upstream_detail=upstream_detail,
    ):
        return

    raise PomaRetryableUpstreamError(
        "POMA temporarily failed to create a chunking job",
        upstream_status=upstream_status,
        upstream_detail=upstream_detail,
        upstream_code=upstream_code,
    )


def _raise_if_poma_terminal_job_failure_error(err: Exception) -> None:
    message = str(err)
    if not _is_terminal_job_failure_message(message):
        return

    raise PomaJobFailedError(
        message,
        upstream_status=_extract_upstream_status_from_error(err),
    ) from err


def _raise_if_poma_terminal_job_failure_response(resp: Any) -> None:
    if not isinstance(resp, dict):
        return

    status_value = resp.get("status")
    status_text = str(status_value).lower() if status_value is not None else ""
    if not _is_terminal_job_status(status_text):
        return

    detail = _extract_error_message(resp)
    if detail and detail.lower() != status_text:
        message = f"POMA job failed (status={status_text}): {detail}"
    else:
        message = f"POMA job failed (status={status_text})"

    upstream_status = None
    for key in ("status_code", "http_status", "error_code", "code"):
        upstream_status = _safe_int(resp.get(key))
        if upstream_status is not None:
            break

    raise PomaJobFailedError(
        message,
        upstream_status=upstream_status,
        job_status=status_text,
    )


def _get_poma_client():
    """Instantiate the POMA SDK client.

    This code is intentionally defensive about import paths and method names,
    because the poma SDK is the integration boundary.
    """

    try:
        from poma import PrimeCut  # type: ignore
    except Exception:
        try:
            from poma.client import PrimeCut  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "POMA chunking requested but 'poma' package is not installed. "
                "Install it with: pip install poma"
            ) from e

    api_key = os.getenv("POMA_API_KEY")
    if not api_key:
        raise RuntimeError("POMA_API_KEY environment variable is required")

    # Support both positional and keyword constructors.
    try:
        return PrimeCut(api_key=api_key, timeout=POMA_TIMEOUT_SECONDS)
    except TypeError:
        try:
            return PrimeCut(api_key=api_key)
        except TypeError:
            return PrimeCut(api_key)


# def _extract_job_id(start_resp: Any) -> str:
#     if isinstance(start_resp, dict):
#         for k in ("job_id", "jobId", "id", "job", "task_id", "taskId"):
#             v = start_resp.get(k)
#             if isinstance(v, str) and v:
#                 return v
#     raise RuntimeError(f"Could not determine POMA job id from response: {start_resp}")


def _normalize_poma_ingest_method(ingest_method: str | None) -> str:
    value = (ingest_method or POMA_INGEST_METHOD).strip().lower()
    if value not in {"ingest", "ingest_eco"}:
        raise ValueError(
            f"Unsupported POMA ingest method '{value}'. Expected 'ingest' or 'ingest_eco'."
        )
    return value


def _coerce_poma_result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            images = getattr(result, "images", None)
            if isinstance(images, dict):
                payload["images"] = images
            return payload

    raise RuntimeError(
        f"Unsupported POMA result type '{type(result).__name__}'; expected dict-like payload or PomaResult."
    )


def poma_chunk_file(
    file_path: str, *, ingest_method: str | None = None
) -> dict[str, Any]:
    """Run PrimeCut chunking on a file path and return dictified PomaResult data."""

    client = _get_poma_client()
    file_name = os.path.basename(file_path)
    resolved_ingest_method = _normalize_poma_ingest_method(ingest_method)
    ingest_fn = getattr(client, resolved_ingest_method, None)
    if not callable(ingest_fn):
        raise RuntimeError(
            f"POMA SDK client does not expose callable method '{resolved_ingest_method}'."
        )

    logger.info(
        "POMA chunking start | file=%s | file_path=%s | ingest_method=%s",
        file_name,
        file_path,
        resolved_ingest_method,
    )

    try:
        result = ingest_fn(
            file_path,
            initial_delay=0.0,
            poll_interval=POMA_POLL_INTERVAL_SECONDS,
            max_interval=max(POMA_POLL_INTERVAL_SECONDS, 15.0),
            show_progress=False,
        )
    except Exception as e:
        _raise_if_poma_too_many_jobs_error(e)
        _raise_if_poma_retryable_create_job_error(e)
        _raise_if_poma_terminal_job_failure_error(e)
        status_code, body = _get_http_response_from_exception(e)
        logger.error(
            "POMA PrimeCut exception | file=%s | ingest_method=%s | error=%s",
            file_name,
            resolved_ingest_method,
            str(e),
        )
        logger.error(
            "POMA PrimeCut error response (full) | file=%s | ingest_method=%s | status_code=%s | body=%s",
            file_name,
            resolved_ingest_method,
            status_code,
            body or "(no body)",
        )
        raise RuntimeError(
            f"PrimeCut.{resolved_ingest_method} failed: {e}"
        ) from e

    result_dict = _coerce_poma_result_to_dict(result)
    logger.info(
        "POMA chunking complete | file=%s | ingest_method=%s | chunks=%s | chunksets=%s | images=%s",
        file_name,
        resolved_ingest_method,
        len(result_dict.get("chunks", []))
        if isinstance(result_dict.get("chunks"), list)
        else "n/a",
        len(result_dict.get("chunksets", []))
        if isinstance(result_dict.get("chunksets"), list)
        else "n/a",
        len(result_dict.get("images", {}))
        if isinstance(result_dict.get("images"), dict)
        else "n/a",
    )
    return result_dict


def _store_path_for_file_id(file_id: str) -> str:
    # Store by file_id only (LibreChat already treats file_id as the unit of access).
    safe_id = "".join(c for c in file_id if c.isalnum() or c in ("_", "-"))
    return os.path.join(POMA_STORE_DIR, f"{safe_id}.json")


def poma_store_chunking_result(
    *, file_id: str, filename: str | None, user_id: str | None, result: dict[str, Any]
) -> None:
    path = _store_path_for_file_id(file_id)
    payload = {
        "file_id": file_id,
        "filename": filename,
        "user_id": user_id,
        "result": result,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def poma_load_chunking_result(file_id: str) -> dict[str, Any] | None:
    path = _store_path_for_file_id(file_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        res = payload.get("result")
        return res if isinstance(res, dict) else None
    except Exception:
        logger.warning(
            "Failed to load POMA cached chunking result for file_id=%s | Traceback: %s",
            file_id,
            traceback.format_exc(),
        )
        return None


def poma_delete_chunking_result(file_id: str) -> None:
    path = _store_path_for_file_id(file_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        logger.warning(
            "Failed to delete POMA cached chunking result for file_id=%s | Traceback: %s",
            file_id,
            traceback.format_exc(),
        )


def poma_chunksets_to_documents(
    *, file_id: str, user_id: str, chunking_result: dict[str, Any]
) -> list[Document]:
    """Convert POMA chunksets to LangChain Documents for embedding + vector storage."""

    chunksets = chunking_result.get("chunksets")
    if not isinstance(chunksets, list):
        raise RuntimeError("POMA chunking result has no 'chunksets' list")

    docs: list[Document] = []
    for i, cs in enumerate(chunksets):
        if not isinstance(cs, dict):
            continue

        chunkset_index = cs.get("chunkset_index", i)
        text = (
            cs.get("to_embed")
            or cs.get("contents")
            or ""
        )
        if not isinstance(text, str):
            text = str(text)

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "file_id": file_id,
                    "user_id": user_id,
                    "chunkset_index": int(chunkset_index),
                    "source": "poma_chunkset",
                },
            )
        )

    return docs


def _get_poma_cheatsheet_fn(client) -> Any:
    # Prefer documented name, fall back to alternate.
    for name in ("create_cheatsheets", "generate_cheatsheets"):
        fn = getattr(client, name, None)
        if callable(fn):
            return fn

    # As a last resort, try module-level.
    try:
        import poma  # type: ignore

        for name in ("create_cheatsheets", "generate_cheatsheets"):
            fn = getattr(poma, name, None)
            if callable(fn):
                return fn
    except Exception:
        pass

    raise RuntimeError(
        "Could not find POMA cheatsheet assembly function (create_cheatsheets/generate_cheatsheets)."
    )


def poma_build_cheatsheet_documents(
    *,
    query: str,
    retrieved: list[tuple[Document, float]],
    k: int,
) -> list[tuple[Document, float]]:
    """Transform retrieved chunkset Documents into per-file cheatsheet Documents."""
    # Group by file_id; preserve ordering (documents are ranked by score)
    per_file: dict[str, list[tuple[Document, float]]] = {}
    for doc, score in retrieved:
        fid = doc.metadata.get("file_id")
        if not isinstance(fid, str) or not fid:
            continue
        per_file.setdefault(fid, []).append((doc, score))

    out: list[tuple[Document, float]] = []
    client = _get_poma_client()
    cheatsheet_fn = _get_poma_cheatsheet_fn(client)

    for fid, docs_scores in per_file.items():
        # Use top-k chunksets for this file as input to cheatsheet assembly.
        docs_scores = docs_scores[:k]

        cached = poma_load_chunking_result(fid)
        if not cached:
            # If cache is missing (e.g. legacy ingests), fall back to returning raw docs.
            logger.warning(
                "POMA_RETURN_CHEATSHEETS enabled but no cached chunking result for file_id=%s; "
                "returning raw retrieved chunksets.",
                fid,
            )
            out.extend(docs_scores)
            continue

        chunksets = cached.get("chunksets")
        chunks = cached.get("chunks")
        if not isinstance(chunksets, list) or not isinstance(chunks, list):
            out.extend(docs_scores)
            continue

        # Map chunkset_index -> chunkset dict (must include 'chunks' list for assembly)
        chunkset_by_idx: dict[int, dict[str, Any]] = {}
        for i, cs in enumerate(chunksets):
            if not isinstance(cs, dict):
                continue
            idx = cs.get("chunkset_index", i)
            try:
                idx_i = int(idx)
            except Exception:
                continue
            chunkset_by_idx[idx_i] = cs

        relevant_chunksets: list[dict[str, Any]] = []
        needed_chunk_ids: set[int] = set()

        for doc, _score in docs_scores:
            try:
                cs_idx = int(doc.metadata.get("chunkset_index"))
            except Exception:
                continue
            cs = chunkset_by_idx.get(cs_idx)
            if not cs:
                continue
            # Keep the original chunkset namespace (file_id/tag) consistent with chunks.
            ids = cs.get("chunks")
            if not isinstance(ids, list):
                continue
            cs2 = dict(cs)
            relevant_chunksets.append(cs2)
            for x in ids:
                try:
                    needed_chunk_ids.add(int(x))
                except Exception:
                    continue

        if not relevant_chunksets:
            out.extend(docs_scores)
            continue

        # Build list of necessary chunk dicts
        chunk_by_idx: dict[int, dict[str, Any]] = {}
        for i, ch in enumerate(chunks):
            if not isinstance(ch, dict):
                continue
            idx = ch.get("chunk_index", i)
            try:
                idx_i = int(idx)
            except Exception:
                continue
            chunk_by_idx[idx_i] = ch

        necessary_chunks: list[dict[str, Any]] = []
        for idx in sorted(needed_chunk_ids):
            ch = chunk_by_idx.get(idx)
            if ch:
                necessary_chunks.append(ch)

        try:
            cs_out = cheatsheet_fn(relevant_chunksets, necessary_chunks)
            print("cs_out", cs_out)
        except TypeError:
            # Some SDKs might require keyword args
            try:
                cs_out = cheatsheet_fn(
                    relevant_chunksets=relevant_chunksets, all_chunks=necessary_chunks
                )
            except TypeError:
                cs_out = cheatsheet_fn(
                    relevant_chunksets=relevant_chunksets, all_necessary_chunks=necessary_chunks
                )

        # Normalize output to list
        cheatsheets: list[Any]
        if isinstance(cs_out, dict) and "cheatsheets" in cs_out:
            cheatsheets = cs_out.get("cheatsheets")  # type: ignore
        else:
            cheatsheets = cs_out if isinstance(cs_out, list) else [cs_out]

        # One cheatsheet per file (as per docs); if multiple provided, concatenate.
        parts: list[str] = []
        for cs in cheatsheets:
            if isinstance(cs, str):
                parts.append(cs)
            elif isinstance(cs, dict):
                for k2 in ("cheatsheet", "content", "markdown", "text"):
                    v = cs.get(k2)
                    if isinstance(v, str) and v:
                        parts.append(v)
                        break
            else:
                parts.append(str(cs))

        text = "\n\n".join([p for p in parts if p])

        if not text:
            out.extend(docs_scores)
            continue

        # Score: take best (lowest) score among retrieved chunksets for that file.
        best_score = min(s for _, s in docs_scores)
        out.append(
            (
                Document(
                    page_content=text,
                    metadata={
                        "file_id": fid,
                        "source": "poma_cheatsheet",
                        "query": query,
                    },
                ),
                best_score,
            )
        )

    return out
