from app.config import (
    RAG_HOST,
    RAG_PORT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_POMA_ACCEPTED_EXTENSIONS,
    PDF_EXTRACT_IMAGES,
    VECTOR_DB_TYPE,
    get_default_embedding_batch_size,
    get_default_poma_ingest_method,
    parse_extension_list,
)

def test_config_defaults():
    assert RAG_HOST is not None
    assert isinstance(RAG_PORT, int)
    assert isinstance(CHUNK_SIZE, int)
    assert isinstance(CHUNK_OVERLAP, int)
    assert isinstance(PDF_EXTRACT_IMAGES, bool)
    assert VECTOR_DB_TYPE is not None


def test_poma_default_embedding_batch_size_when_env_unset():
    assert get_default_embedding_batch_size("poma", None) == 250


def test_explicit_embedding_batch_size_overrides_poma_default():
    assert get_default_embedding_batch_size("poma", "0") == 0
    assert get_default_embedding_batch_size("poma", "64") == 64
    assert get_default_embedding_batch_size("langchain", None) == 0


def test_default_poma_ingest_method():
    assert get_default_poma_ingest_method(None) == "ingest"
    assert get_default_poma_ingest_method("ingest") == "ingest"
    assert get_default_poma_ingest_method("ingest_eco") == "ingest_eco"


def test_invalid_poma_ingest_method_raises():
    try:
        get_default_poma_ingest_method("nope")
    except ValueError as exc:
        assert "POMA_INGEST_METHOD" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid POMA ingest method")


def test_parse_extension_list_normalizes_and_deduplicates():
    assert parse_extension_list(" .PDF, txt,TXT,,.md ") == ("pdf", "txt", "md")


def test_default_poma_accepted_extensions_include_common_text_types():
    assert "txt" in DEFAULT_POMA_ACCEPTED_EXTENSIONS
    assert "pdf" in DEFAULT_POMA_ACCEPTED_EXTENSIONS
    assert "md" in DEFAULT_POMA_ACCEPTED_EXTENSIONS
