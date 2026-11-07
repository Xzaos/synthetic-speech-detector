# synthetic-speech-detector

Does a real-vs-synthetic Hindi speech detector trained on one TTS system generalise to another it never saw? I train on real FLEURS Hindi against facebook/mms-tts-hin, then test on MMS again (seen) and on Sarvam Bulbul (unseen), clean and through a phone-line channel.

No run has happened yet, so every result cell below is `_`. The numbers will come from `results/report.json` and `results/report.md` after `scripts/run_kaggle.sh` finishes on a Kaggle T4.

## Results

EER (lower is better) and ROC AUC for real vs fake, 95% bootstrap interval over clips in brackets.

| model | condition | EER | AUC |
|---|---|---|---|
| baseline (utterance stats) | seen generator (MMS) | _ | _ |
| baseline (utterance stats) | unseen generator (Bulbul) | _ | _ |
| baseline (utterance stats) | seen, phone channel | _ | _ |
| baseline (utterance stats) | unseen, phone channel | _ | _ |
| CNN detector | seen generator (MMS) | _ | _ |
| CNN detector | unseen generator (Bulbul) | _ | _ |
| CNN detector | seen, phone channel | _ | _ |
| CNN detector | unseen, phone channel | _ | _ |

The decision threshold is the score at the validation EER point (validation clips are real vs MMS only). Applied to the test sets:

| model | false-accept rate on Bulbul (fake passed as real) | same, phone channel |
|---|---|---|
| baseline | _ | _ |
| CNN detector | _ | _ |

Wall-clock for the whole script on the T4: _. Baseline leak check: _ (the report sets it to "leaks" if the baseline's AUC on MMS is 0.8 or higher, and if it does, this README has to say the task leaks shortcuts).

## Design

The data is paired by sentence. I sample 800 FLEURS hi_in train clips and 200 test clips. MMS speaks the same 800 train sentences and the same 200 test sentences, and Bulbul speaks the same 200 test sentences. For each real clip there is a fake clip with the same text, so what is said cannot separate the classes. Ten percent of the train sentences are held out for validation, along with both their real and fake clips. FLEURS repeats some sentences across speakers, so the split is by sentence text, not by clip.

Fake clips are named after the real clip they pair with, and a missing one stops the run instead of being dropped, since dropping pairs would quietly change the test set.

Shortcut controls are applied to every clip, real or fake, in the same order:

- resample to 16 kHz mono (Bulbul is requested at 16 kHz directly, so none of the three sources is resampled)
- trim leading and trailing silence (frames more than 35 dB under the loudest frame)
- normalise RMS to a fixed level
- crop to 3 s: random crops in training, the centre crop for evaluation, and short clips repeated to length instead of zero padded so the padding does not encode duration

The phone-channel condition passes each processed test clip through a 300 to 3400 Hz bandpass, an 8 kHz round trip and 8-bit mu-law, then restores the RMS. It is only applied at test time. The detector never sees it in training, so the channel columns measure a second shift on top of the generator shift.

The baseline is a logistic regression on nine statistics of the full processed clip: duration before cropping, RMS, mean and standard deviation of spectral centroid, rolloff and bandwidth, and zero-crossing rate. Its RMS is measured after trimming but before normalisation, and its duration is the trimmed duration. Both are values the detector is denied, so the baseline shows what the controls remove as well as what is left. If this model scores well, the task leaks shortcuts and the detector's score means less.

The detector is a 64-bin log-mel front end (per-clip normalised), four conv blocks (3x3 conv, BatchNorm, ReLU, 2x2 max pool), a mean over time, and a small two-layer head that outputs one logit. It has about 91K parameters. Training uses AdamW with a one-cycle schedule, SpecAugment-style frequency and time masks, a 10 minute wall-clock cap, and keeps the checkpoint with the lowest validation EER (ties go to the lower validation loss).

Confidence intervals resample clips within each class, 1000 times. The same 200 real test clips are scored in every condition, so the seen and unseen intervals are correlated through the real side.

Bulbul is called with `bulbul:v3` at `https://api.sarvam.ai/text-to-speech` (checked against docs.sarvam.ai on 2026-11-07), `language_code` hi-IN, 16 kHz output. I chose v3 over v4-flash because its speakers are plain names and all 37 are listed in the docs, so the speaker for sentence i is `speakers[i % 37]`. Requests run two at a time, retry on 429 and 5xx with backoff, fail on any other 4xx, and skip clips already on disk.

## Running it

On Kaggle (GPU on, internet on, secrets `SARVAM_API_KEY` and `HF_TOKEN`):

```
git clone https://github.com/Xzaos/synthetic-speech-detector
cd synthetic-speech-detector
bash scripts/run_kaggle.sh
```

Locally, `pip install -e .[dev]` and `pytest -q`. The tests use synthetic audio and a mocked HTTP transport, so they need no network.

## Limitations

- MMS has a single voice. A detector trained against it can learn "this is MMS's speaker" and not "this is synthetic", and the seen-generator score cannot tell those apart. The unseen-generator test is meant to expose exactly that, because Bulbul is a different system with a different voice for each sentence.
- There are only two generators. One seen and one unseen system is a single data point about generalisation, not an estimate of it, and Bulbul may be easy or hard for reasons that have nothing to do with TTS in general.
- FLEURS is read speech recorded for a benchmark. Conversational or noisy real speech is not covered.
- Real and fake differ in speaker count as well as in origin: the real side has many speakers and each Bulbul clip uses one of 37 voices.
- The phone channel is one fixed degradation applied after synthesis, not a real telephone recording.
- Bulbul voices were not checked individually for Hindi quality.
