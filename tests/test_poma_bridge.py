from app.services import poma_bridge


class DummyPomaResult:
    def __init__(self):
        self.images = {"image_00001": "data:image/png;base64,abc"}

    def to_dict(self):
        return {
            "chunks": [{"chunk_index": 0, "content": "chunk"}],
            "chunksets": [{"chunkset_index": 0, "chunks": [0], "to_embed": "chunk"}],
        }


class DummyPrimeCutClient:
    def __init__(self):
        self.calls = []

    def ingest(self, file_path, **kwargs):
        self.calls.append(("ingest", file_path, kwargs))
        return DummyPomaResult()

    def ingest_eco(self, file_path, **kwargs):
        self.calls.append(("ingest_eco", file_path, kwargs))
        return DummyPomaResult()


def test_poma_chunk_file_uses_requested_ingest_method(monkeypatch):
    client = DummyPrimeCutClient()
    monkeypatch.setattr(poma_bridge, "_get_poma_client", lambda: client)

    result = poma_bridge.poma_chunk_file(
        "/tmp/test.pdf", ingest_method="ingest_eco"
    )

    assert client.calls[0][0] == "ingest_eco"
    assert client.calls[0][1] == "/tmp/test.pdf"
    assert result["chunksets"][0]["to_embed"] == "chunk"
    assert result["images"]["image_00001"].startswith("data:image/png")


def test_poma_chunk_file_uses_default_ingest_method(monkeypatch):
    client = DummyPrimeCutClient()
    monkeypatch.setattr(poma_bridge, "_get_poma_client", lambda: client)
    monkeypatch.setattr(poma_bridge, "POMA_INGEST_METHOD", "ingest")

    poma_bridge.poma_chunk_file("/tmp/test.pdf")

    assert client.calls[0][0] == "ingest"
