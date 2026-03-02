import os
import jwt
import datetime
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

from main import app

client = TestClient(app)


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def override_vector_store(monkeypatch):
    from app.config import vector_store
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from app.routes import document_routes

    # Clear the LRU cache and patch the cached function to return dummy embeddings
    document_routes.get_cached_query_embedding.cache_clear()

    def dummy_get_cached_query_embedding(query):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        document_routes, "get_cached_query_embedding", dummy_get_cached_query_embedding
    )

    # Initialize thread pool for tests since TestClient doesn't run lifespan
    if not hasattr(app.state, "thread_pool") or app.state.thread_pool is None:
        app.state.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-worker"
        )

    # Override get_all_ids as an async function - patch at CLASS level to bypass run_in_executor
    async def dummy_get_all_ids(self, executor=None):
        return ["testid1", "testid2"]

    monkeypatch.setattr(AsyncPgVector, "get_all_ids", dummy_get_all_ids)

    # Override get_filtered_ids as an async function.
    async def dummy_get_filtered_ids(self, ids, executor=None):
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    monkeypatch.setattr(AsyncPgVector, "get_filtered_ids", dummy_get_filtered_ids)

    # Override get_documents_by_ids as an async function.
    async def dummy_get_documents_by_ids(self, ids, executor=None):
        return [
            Document(page_content="Test content", metadata={"file_id": id})
            for id in ids
        ]

    monkeypatch.setattr(
        AsyncPgVector, "get_documents_by_ids", dummy_get_documents_by_ids
    )

    # Override embedding_function with a dummy that doesn't call OpenAI
    class DummyEmbedding:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    vector_store.embedding_function = DummyEmbedding()

    # Override similarity search to return a tuple (Document, score).
    def dummy_similarity_search_with_score_by_vector(self, embedding, k, filter=None):
        file_id = "testid1"
        user_id = "testuser"
        if isinstance(filter, dict):
            filter_file_id = filter.get("file_id")
            if isinstance(filter_file_id, str):
                file_id = filter_file_id
            elif (
                isinstance(filter_file_id, dict)
                and isinstance(filter_file_id.get("$in"), list)
                and filter_file_id["$in"]
            ):
                file_id = filter_file_id["$in"][0]

            filter_user_id = filter.get("user_id")
            if isinstance(filter_user_id, str):
                user_id = filter_user_id

        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": file_id,
                "user_id": user_id,
            },
        )
        return [(doc, 0.9)]

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        file_id = "testid1"
        user_id = "testuser"
        if isinstance(filter, dict):
            filter_file_id = filter.get("file_id")
            if isinstance(filter_file_id, str):
                file_id = filter_file_id
            elif (
                isinstance(filter_file_id, dict)
                and isinstance(filter_file_id.get("$in"), list)
                and filter_file_id["$in"]
            ):
                file_id = filter_file_id["$in"][0]

            filter_user_id = filter.get("user_id")
            if isinstance(filter_user_id, str):
                user_id = filter_user_id

        doc = Document(
            page_content="Queried content",
            metadata={
                "file_id": file_id,
                "user_id": user_id,
            },
        )
        return [(doc, 0.9)]

    monkeypatch.setattr(
        AsyncPgVector,
        "similarity_search_with_score_by_vector",
        dummy_similarity_search_with_score_by_vector,
    )
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    # Override document addition functions.
    def dummy_add_documents(self, docs, ids):
        return ids

    async def dummy_aadd_documents(self, docs, ids=None, executor=None):
        return ids

    monkeypatch.setattr(AsyncPgVector, "add_documents", dummy_add_documents)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", dummy_aadd_documents)

    # Override delete function.
    async def dummy_delete(self, ids=None, collection_only=False, executor=None):
        return None

    monkeypatch.setattr(AsyncPgVector, "delete", dummy_delete)


def test_get_all_ids(auth_headers):
    response = client.get("/ids", headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert "testid1" in json_data


def test_get_documents_by_ids(auth_headers):
    response = client.get(
        "/documents", params={"ids": ["testid1"]}, headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0]["page_content"] == "Test content"
    assert json_data[0]["metadata"]["file_id"] == "testid1"


def test_delete_documents(auth_headers):
    response = client.request(
        "DELETE", "/documents", json=["testid1"], headers=auth_headers
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "Documents for" in json_data["message"]


def test_query_embeddings_by_file_id(auth_headers):
    data = {
        "query": "Test query",
        "file_id": "testid1",
        "k": 4,
        "entity_id": "testuser",
    }
    response = client.post("/query", json=data, headers=auth_headers)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_embed_local_file(tmp_path, auth_headers, monkeypatch):
    # Create a temporary file.
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document.")

    data = {
        "filepath": str(test_file),
        "filename": "test.txt",
        "file_content_type": "text/plain",
        "file_id": "testid1",
    }
    response = client.post("/local/embed", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_embed_file(tmp_path, auth_headers):
    file_content = "This is a test file for the embed endpoint."
    test_file = tmp_path / "test_embed.txt"
    test_file.write_text(file_content)
    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("test_embed.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert set(json_data.keys()) == {
        "status",
        "message",
        "file_id",
        "filename",
        "known_type",
    }
    assert json_data["status"] is True
    assert json_data["message"] == "File processed successfully."
    assert json_data["file_id"] == "testid1"
    assert json_data["filename"] == "test_embed.txt"


def test_embed_file_too_many_jobs_returns_structured_403(
    tmp_path, auth_headers, monkeypatch
):
    from app.routes import document_routes
    from app.services.poma_bridge import PomaTooManyJobsError

    file_content = "This should trigger a mocked POMA concurrency error."
    test_file = tmp_path / "too_many_jobs.txt"
    test_file.write_text(file_content)

    def raise_too_many_jobs(_file_path):
        raise PomaTooManyJobsError(
            "Too many jobs",
            upstream_status=403,
            upstream_detail="Too many jobs",
            upstream_code="TOO_MANY_JOBS",
        )

    monkeypatch.setattr(document_routes, "CHUNKER_PROVIDER", "poma")
    monkeypatch.setattr(document_routes, "poma_chunk_file", raise_too_many_jobs)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("too_many_jobs.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 403, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["code"] == "TOO_MANY_JOBS"
    assert json_data["detail"] == "Too many jobs"
    assert json_data["message"] == "Too many jobs"
    assert json_data["upstream_status"] == 403
    assert json_data["upstream_code"] == "TOO_MANY_JOBS"
    assert json_data["upstream_detail"] == "Too many jobs"
    assert json_data["retryable"] is True
    assert json_data["source"] == "poma"


def test_embed_file_poma_retryable_upstream_error_returns_structured_503(
    tmp_path, auth_headers, monkeypatch
):
    from app.routes import document_routes
    from app.services.poma_bridge import PomaRetryableUpstreamError

    file_content = "This should trigger a mocked retryable POMA create-job error."
    test_file = tmp_path / "poma_retryable_create_job.txt"
    test_file.write_text(file_content)

    def raise_retryable_poma_error(_file_path):
        raise PomaRetryableUpstreamError(
            "POMA temporarily failed to create a chunking job",
            upstream_status=400,
            upstream_detail="Failed to create job",
            upstream_code=400,
        )

    monkeypatch.setattr(document_routes, "CHUNKER_PROVIDER", "poma")
    monkeypatch.setattr(document_routes, "poma_chunk_file", raise_retryable_poma_error)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("poma_retryable_create_job.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 503, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["code"] == "POMA_RETRYABLE_UPSTREAM_ERROR"
    assert json_data["message"] == "POMA temporarily failed to create a chunking job"
    assert json_data["detail"] == "POMA temporarily failed to create a chunking job"
    assert json_data["upstream_status"] == 400
    assert json_data["upstream_code"] == 400
    assert json_data["upstream_detail"] == "Failed to create job"
    assert json_data["retryable"] is True
    assert json_data["source"] == "poma"


def test_embed_file_retryable_store_error_returns_structured_503(
    tmp_path, auth_headers, monkeypatch
):
    from app.routes import document_routes

    file_content = "This should trigger a mocked transient vector DB error."
    test_file = tmp_path / "vector_db_retryable.txt"
    test_file.write_text(file_content)

    async def mocked_store_error(*args, **kwargs):
        return {
            "message": "An error occurred while adding documents.",
            "error": "SSL SYSCALL error: EOF detected",
            "code": "VECTOR_DB_TRANSIENT_ERROR",
            "retryable": True,
            "source": "vector_db",
            "attempts": 3,
        }

    monkeypatch.setattr(document_routes, "store_data_in_vector_db", mocked_store_error)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("vector_db_retryable.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 503, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["code"] == "VECTOR_DB_TRANSIENT_ERROR"
    assert json_data["retryable"] is True
    assert json_data["source"] == "vector_db"
    assert json_data["attempts"] == 3
    assert "EOF detected" in json_data["detail"]


def test_embed_local_file_store_error_not_reported_as_success(
    tmp_path, auth_headers, monkeypatch
):
    from app.routes import document_routes

    test_file = tmp_path / "local_store_error.txt"
    test_file.write_text("content")

    async def mocked_store_error(*args, **kwargs):
        return {
            "message": "An error occurred while adding documents.",
            "error": "SSL SYSCALL error: EOF detected",
            "code": "VECTOR_DB_TRANSIENT_ERROR",
            "retryable": True,
            "source": "vector_db",
            "attempts": 3,
        }

    monkeypatch.setattr(document_routes, "store_data_in_vector_db", mocked_store_error)

    data = {
        "filepath": str(test_file),
        "filename": "local_store_error.txt",
        "file_content_type": "text/plain",
        "file_id": "testid1",
    }
    response = client.post("/local/embed", json=data, headers=auth_headers)
    assert response.status_code == 503, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["code"] == "VECTOR_DB_TRANSIENT_ERROR"
    assert json_data["retryable"] is True
    assert json_data["source"] == "vector_db"


def test_embed_file_upload_store_error_not_reported_as_success(
    tmp_path, auth_headers, monkeypatch
):
    from app.routes import document_routes

    test_file = tmp_path / "upload_store_error.txt"
    test_file.write_text("content")

    async def mocked_store_error(*args, **kwargs):
        return {
            "message": "An error occurred while adding documents.",
            "error": "SSL SYSCALL error: EOF detected",
            "code": "VECTOR_DB_TRANSIENT_ERROR",
            "retryable": True,
            "source": "vector_db",
            "attempts": 3,
        }

    monkeypatch.setattr(document_routes, "store_data_in_vector_db", mocked_store_error)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed-upload",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"uploaded_file": ("upload_store_error.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 503, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["code"] == "VECTOR_DB_TRANSIENT_ERROR"
    assert json_data["retryable"] is True
    assert json_data["source"] == "vector_db"


@pytest.mark.asyncio
async def test_store_data_in_vector_db_retries_transient_db_error(monkeypatch):
    from app.routes import document_routes
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    call_count = {"value": 0}

    async def flaky_aadd_documents(self, docs, ids=None, executor=None):
        call_count["value"] += 1
        if call_count["value"] < 3:
            raise RuntimeError("SSL SYSCALL error: EOF detected")
        return ids

    monkeypatch.setattr(document_routes, "CHUNKER_PROVIDER", "langchain")
    monkeypatch.setattr(document_routes, "EMBEDDING_BATCH_SIZE", 0)
    monkeypatch.setattr(document_routes, "VECTOR_DB_RETRY_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(document_routes, "VECTOR_DB_RETRY_BASE_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(document_routes, "VECTOR_DB_RETRY_MAX_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(document_routes, "VECTOR_DB_RETRY_JITTER_SECONDS", 0.0)
    monkeypatch.setattr(AsyncPgVector, "aadd_documents", flaky_aadd_documents)

    docs = [Document(page_content="retry test content", metadata={})]
    result = await document_routes.store_data_in_vector_db(
        data=docs,
        file_id="retry-test-id",
        user_id="testuser",
        executor=app.state.thread_pool,
    )

    assert result["message"] == "Documents added successfully"
    assert call_count["value"] == 3


def test_embed_file_poma_job_failed_returns_upstream_status(
    tmp_path, auth_headers, monkeypatch
):
    from app.routes import document_routes
    from app.services.poma_bridge import PomaJobFailedError

    file_content = "This should trigger a mocked terminal POMA job failure."
    test_file = tmp_path / "poma_job_failed.txt"
    test_file.write_text(file_content)

    def raise_poma_job_failed(_file_path):
        raise PomaJobFailedError(
            "POMA-AI job polling failed: Job failed: Job failed with code 503: No details provided.",
            upstream_status=503,
            job_status="failed",
        )

    monkeypatch.setattr(document_routes, "CHUNKER_PROVIDER", "poma")
    monkeypatch.setattr(document_routes, "poma_chunk_file", raise_poma_job_failed)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("poma_job_failed.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 503, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["code"] == "POMA_JOB_FAILED"
    assert "Job failed" in json_data["detail"]
    assert json_data["upstream_status"] == 503
    assert json_data["job_status"] == "failed"


def test_embed_file_unrelated_poma_error_not_misclassified(
    tmp_path, auth_headers, monkeypatch
):
    from app.routes import document_routes

    file_content = "This should trigger a generic mocked POMA error."
    test_file = tmp_path / "other_poma_error.txt"
    test_file.write_text(file_content)

    def raise_other_error(_file_path):
        raise RuntimeError("POMA exploded for another reason")

    monkeypatch.setattr(document_routes, "CHUNKER_PROVIDER", "poma")
    monkeypatch.setattr(document_routes, "poma_chunk_file", raise_other_error)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"file": ("other_poma_error.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 400, f"Response: {response.text}"
    json_data = response.json()
    assert json_data.get("code") is None
    assert "TOO_MANY_JOBS" not in response.text
    assert "Error during file processing" in json_data["detail"]


def test_load_document_context(auth_headers):
    response = client.get("/documents/testid1/context", headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    content = response.text
    assert "testid1" in content or "Test content" in content


def test_embed_file_upload(tmp_path, auth_headers, monkeypatch):
    file_content = "Test content for embed upload."
    test_file = tmp_path / "upload_test.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/embed-upload",
            data={"file_id": "testid1", "entity_id": "testuser"},
            files={"uploaded_file": ("upload_test.txt", f, "text/plain")},
            headers=auth_headers,
        )
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert json_data["status"] is True
    assert json_data["file_id"] == "testid1"


def test_query_multiple(auth_headers):
    data = {
        "query": "Test query multiple",
        "file_ids": ["testid1", "testid2"],
        "k": 4,
    }
    response = client.post("/query_multiple", json=data, headers=auth_headers)
    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()
    assert isinstance(json_data, list)
    if json_data:
        doc = json_data[0][0]
        assert doc["page_content"] == "Queried content"


def test_query_global_user_scoped(auth_headers, monkeypatch):
    from app.routes import document_routes
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    captured = {}

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        captured["filter"] = filter
        return [
            (
                Document(
                    page_content="Global queried content",
                    metadata={"file_id": "global_testid", "user_id": "testuser"},
                ),
                0.9,
            )
        ]

    monkeypatch.setattr(document_routes, "QUERY_GLOBAL", False)
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    response = client.post(
        "/query_global",
        json={"query": "Global query", "k": 4},
        headers=auth_headers,
    )
    assert response.status_code == 200, f"Response: {response.text}"
    assert captured["filter"] == {"user_id": "testuser"}

    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0][0]["page_content"] == "Global queried content"
    assert json_data[0][0]["metadata"]["user_id"] == "testuser"


def test_query_global_user_scoped_public_fallback(monkeypatch):
    from app.routes import document_routes
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    captured = {}

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        captured["filter"] = filter
        return [
            (
                Document(
                    page_content="Public global queried content",
                    metadata={"file_id": "global_public", "user_id": "public"},
                ),
                0.9,
            )
        ]

    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setattr(document_routes, "QUERY_GLOBAL", False)
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    response = client.post("/query_global", json={"query": "Global query public", "k": 4})
    assert response.status_code == 200, f"Response: {response.text}"
    assert captured["filter"] == {"user_id": "public"}

    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0][0]["metadata"]["user_id"] == "public"


def test_query_global_full_global(auth_headers, monkeypatch):
    from app.routes import document_routes
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    captured = {}

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        captured["filter"] = filter
        return [
            (
                Document(
                    page_content="Full global queried content",
                    metadata={"file_id": "global_any", "user_id": "otheruser"},
                ),
                0.9,
            )
        ]

    monkeypatch.setattr(document_routes, "QUERY_GLOBAL", True)
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    response = client.post(
        "/query_global",
        json={"query": "Global query full", "k": 4},
        headers=auth_headers,
    )
    assert response.status_code == 200, f"Response: {response.text}"
    assert captured["filter"] is None

    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0][0]["metadata"]["user_id"] == "otheruser"


def test_query_global_empty_results(auth_headers, monkeypatch):
    from app.routes import document_routes
    from app.services.vector_store.async_pg_vector import AsyncPgVector

    captured = {}

    async def dummy_asimilarity_search_with_score_by_vector(
        self, embedding, k, filter=None, executor=None
    ):
        captured["filter"] = filter
        return []

    monkeypatch.setattr(document_routes, "QUERY_GLOBAL", False)
    monkeypatch.setattr(
        AsyncPgVector,
        "asimilarity_search_with_score_by_vector",
        dummy_asimilarity_search_with_score_by_vector,
    )

    response = client.post(
        "/query_global",
        json={"query": "Global query no results", "k": 4},
        headers=auth_headers,
    )
    assert response.status_code == 200, f"Response: {response.text}"
    assert response.json() == []
    assert captured["filter"] == {"user_id": "testuser"}


def test_extract_text_from_file(tmp_path, auth_headers):
    """Test the /text endpoint for text extraction without embeddings."""
    file_content = "This is a test file for text extraction.\nIt has multiple lines.\nAnd should be extracted properly."
    test_file = tmp_path / "test_text_extraction.txt"
    test_file.write_text(file_content)

    with test_file.open("rb") as f:
        response = client.post(
            "/text",
            data={"file_id": "test_text_123", "entity_id": "testuser"},
            files={"file": ("test_text_extraction.txt", f, "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200, f"Response: {response.text}"
    json_data = response.json()

    # Check response structure
    assert "text" in json_data
    assert "file_id" in json_data
    assert "filename" in json_data
    assert "known_type" in json_data

    # Check response content
    assert json_data["text"] == file_content
    assert json_data["file_id"] == "test_text_123"
    assert json_data["filename"] == "test_text_extraction.txt"
    assert json_data["known_type"] is True  # text files are known types
