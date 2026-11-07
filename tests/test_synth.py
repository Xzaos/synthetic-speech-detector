import base64
import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from synthdet import synth
from synthdet.data import fake_path, split_train_val


def wav_b64(sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(sr // 10, dtype="float32"), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def reals(n):
    return [{"sent": f"r{i}", "text": f"वाक्य {i}"} for i in range(n)]


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "k")

    async def nosleep(_):
        pass

    monkeypatch.setattr(synth.asyncio, "sleep", nosleep)


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_retries_429_and_5xx_then_writes(tmp_path):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={})
        if calls["n"] == 2:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"audios": [wav_b64()]})

    n = synth.synth_bulbul(reals(1), str(tmp_path), "test", client=client(handler))
    assert n == 1 and calls["n"] == 3
    assert fake_path(str(tmp_path), "bulbul", "test", "r0").exists()


def test_request_body_and_speaker_rotation(tmp_path):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        assert req.headers["api-subscription-key"] == "k"
        return httpx.Response(200, json={"audios": [wav_b64()]})

    synth.synth_bulbul(reals(3), str(tmp_path), "test", speakers=["a", "b"], client=client(handler))
    by_text = {b["text"]: b for b in seen}
    assert by_text["वाक्य 0"]["speaker"] == "a" and by_text["वाक्य 1"]["speaker"] == "b" and by_text["वाक्य 2"]["speaker"] == "a"
    assert all(b["language_code"] == "hi-IN" and b["speech_sample_rate"] == 16000 for b in seen)


def test_other_4xx_fails_loudly_and_writes_nothing(tmp_path):
    def handler(req):
        return httpx.Response(422, json={"error": "bad speaker"})

    with pytest.raises(synth.SarvamError, match="422"):
        synth.synth_bulbul(reals(2), str(tmp_path), "test", client=client(handler))
    assert not list(tmp_path.rglob("*.wav"))


def test_gives_up_after_retries(tmp_path):
    with pytest.raises(synth.SarvamError, match="gave up"):
        synth.synth_bulbul(reals(1), str(tmp_path), "test", max_retries=3, client=client(lambda r: httpx.Response(500, json={})))


def test_wrong_sample_rate_is_an_error(tmp_path):
    with pytest.raises(synth.SarvamError, match="16000"):
        synth.synth_bulbul(reals(1), str(tmp_path), "test", client=client(lambda r: httpx.Response(200, json={"audios": [wav_b64(24000)]})))


def test_resume_skips_existing(tmp_path):
    first = {"n": 0}

    def ok(req):
        first["n"] += 1
        return httpx.Response(200, json={"audios": [wav_b64()]})

    synth.synth_bulbul(reals(2), str(tmp_path), "test", client=client(ok))
    first["n"] = 0
    assert synth.synth_bulbul(reals(2), str(tmp_path), "test", client=client(ok)) == 0
    assert first["n"] == 0


def test_split_holds_out_whole_sentences():
    rows = [{"id": f"c{i}", "text": f"s{i // 2}"} for i in range(200)]
    train, val = split_train_val(rows, 0.1, seed=0)
    assert {r["text"] for r in train}.isdisjoint({r["text"] for r in val})
    assert len({r["text"] for r in val}) == 10
    assert len(train) + len(val) == 200
