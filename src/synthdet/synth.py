"""Synthetic speech for the FLEURS sentences: MMS-TTS locally, Bulbul over
REST. Both write one wav per real clip id and skip files that already exist,
so a rerun picks up where an interrupted one stopped.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import random
from pathlib import Path

import soundfile as sf

from .data import fake_path

URL = "https://api.sarvam.ai/text-to-speech"
MODEL = "bulbul:v3"
# bulbul:v3 speakers from docs.sarvam.ai (text-to-speech convert). Short names
# are only valid for v3, so another model needs its own list via --speakers.
V3_SPEAKERS = [
    "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran", "kavya", "amit", "dev",
    "ishita", "shreya", "ratan", "varun", "manan", "sumit", "roopa", "kabir", "aayan", "ashutosh", "advait",
    "anand", "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha",
    "rehan", "soham", "rupali",
]
RETRY_STATUS = (429, 500, 502, 503, 504)


def synth_mms(reals: list[dict], data_dir: str, split: str, seed: int = 0, model_id: str = "facebook/mms-tts-hin") -> int:
    import torch
    from transformers import AutoTokenizer, VitsModel

    todo = [r for r in reals if not fake_path(data_dir, "mms", split, r["sent"]).exists()]
    if not todo:
        return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(dev).eval()
    for i, r in enumerate(todo):
        enc = tok(r["text"], return_tensors="pt")
        if enc["input_ids"].shape[1] == 0:
            raise ValueError(f"{r['sent']}: tokenizer produced no tokens for {r['text'][:60]!r}")
        torch.manual_seed(seed + i)
        with torch.inference_mode():
            wav = model(**{k: v.to(dev) for k, v in enc.items()}).waveform[0].float().cpu().numpy()
        p = fake_path(data_dir, "mms", split, r["sent"])
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, wav, model.config.sampling_rate, subtype="PCM_16")
    return len(todo)


class SarvamError(RuntimeError):
    pass


def speaker_for(index: int, speakers: list[str]) -> str:
    return speakers[index % len(speakers)]


async def _one(client, sem, key, r, path, speaker, model, max_retries):
    import httpx

    body = {"text": r["text"], "language_code": "hi-IN", "speaker": speaker, "model": model, "speech_sample_rate": 16000}
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.post(URL, json=body, headers={"api-subscription-key": key})
            except httpx.TransportError:
                resp = None
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                wav = base64.b64decode("".join(data["audios"]))
                info = sf.info(io.BytesIO(wav))
                if info.samplerate != 16000:
                    raise SarvamError(f"{r['sent']}: got {info.samplerate} Hz, asked for 16000")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                tmp.write_bytes(wav)
                os.replace(tmp, path)
                return
            if resp is not None and resp.status_code not in RETRY_STATUS:
                raise SarvamError(f"{r['sent']}: HTTP {resp.status_code} {resp.text[:300]}")
            wait = 2**attempt + random.random()
            if resp is not None and resp.headers.get("retry-after", "").isdigit():
                wait = max(wait, int(resp.headers["retry-after"]))
            await asyncio.sleep(wait)
        raise SarvamError(f"{r['sent']}: gave up after {max_retries} tries")


async def _run(reals, data_dir, split, speakers, model, concurrency, max_retries, key, client):
    sem = asyncio.Semaphore(concurrency)
    jobs = []
    for i, r in enumerate(reals):
        p = fake_path(data_dir, "bulbul", split, r["sent"])
        if not p.exists():
            jobs.append(_one(client, sem, key, r, p, speaker_for(i, speakers), model, max_retries))
    await asyncio.gather(*jobs)
    return len(jobs)


def synth_bulbul(reals, data_dir, split, speakers=None, model=MODEL, concurrency=2, max_retries=6, client=None) -> int:
    """Any non-retryable 4xx, or exhausted retries, raises and stops the run.
    Clips already written stay on disk for the next attempt."""
    key = os.environ.get("SARVAM_API_KEY")
    if not key and client is None:
        raise RuntimeError("set SARVAM_API_KEY")
    speakers = speakers or V3_SPEAKERS

    async def main():
        import httpx

        c = client or httpx.AsyncClient(timeout=120)
        try:
            return await _run(reals, data_dir, split, speakers, model, concurrency, max_retries, key, c)
        finally:
            if client is None:
                await c.aclose()

    return asyncio.run(main())
