# Synthetic Speech Detector

Does a real-vs-synthetic Hindi speech detector trained on one TTS system generalise to another it never saw? I trained on real FLEURS Hindi against facebook/mms-tts-hin, then tested on MMS again (seen) and on Sarvam Bulbul (unseen), clean and through a phone-line channel. The numbers below are from one run on a Kaggle T4 (`results/report.md`, `results/report.json`).

## What the run found

The CNN separates real from MMS perfectly (EER 0.000, AUC 1.000) and does much worse on Bulbul: EER 0.185 [0.150, 0.230], AUC 0.909 [0.881, 0.935]. The AUC makes that look like a mild drop. It is worse than that at the operating point. The threshold is set at the validation EER, and at that threshold the CNN passes 94% of Bulbul fakes as real, while it passes 2% of MMS fakes and rejects none of the real clips. The ranking carries over to a new generator, but the score scale does not, so a threshold tuned on one generator is useless on the other.

The cheap-stats baseline, a logistic regression on nine utterance statistics, generalises better than the CNN: unseen EER 0.080 [0.055, 0.115], AUC 0.949 [0.925, 0.971]. The leak check fired (its AUC on MMS is 0.996, the cutoff was 0.8). A model with no access to fine spectro-temporal detail separates real from fake almost perfectly on both generators, so a large part of what both models learn is probably FLEURS recording conditions against any TTS output, and not synthesis artifacts. Both models partly detect what the real data sounds like.

I have not run an ablation over the nine statistics, so I can't say which ones carry the separation. The report does show two things that differ by class even after the controls. Mean raw RMS (before normalisation, which the baseline sees and the CNN does not) is 0.042 for real, 0.153 for MMS and 0.114 for Bulbul. Mean trimmed duration is 10.7 s for real and 9.0 s for both fake sets. Those are candidates, not findings.

The phone channel moves the two generators in opposite directions for the CNN. Unseen EER improves from 0.185 to 0.120 [0.085, 0.160], while seen EER worsens from 0.000 to 0.065 [0.040, 0.090]. That fits the CNN having leaned on high-band cues specific to MMS that the 300 to 3400 Hz bandpass removes, but the run doesn't test that directly. The baseline's EER hardly moves under the channel (0.040 to 0.030 seen, 0.080 to 0.085 unseen), yet its threshold breaks: it passes 100% of MMS fakes and 97.5% of Bulbul fakes as real, with no real clip rejected.

## Results

EER (lower is better) and ROC AUC for real vs fake, 95% bootstrap interval over clips in brackets.

| model | condition | EER | AUC |
|---|---|---|---|
| baseline (utterance stats) | seen generator (MMS) | 0.040 [0.020, 0.060] | 0.996 [0.992, 0.998] |
| baseline (utterance stats) | unseen generator (Bulbul) | 0.080 [0.055, 0.115] | 0.949 [0.925, 0.971] |
| baseline (utterance stats) | seen, phone channel | 0.030 [0.010, 0.050] | 0.998 [0.995, 0.999] |
| baseline (utterance stats) | unseen, phone channel | 0.085 [0.060, 0.115] | 0.960 [0.941, 0.978] |
| CNN detector | seen generator (MMS) | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |
| CNN detector | unseen generator (Bulbul) | 0.185 [0.150, 0.230] | 0.909 [0.881, 0.935] |
| CNN detector | seen, phone channel | 0.065 [0.040, 0.090] | 0.987 [0.979, 0.993] |
| CNN detector | unseen, phone channel | 0.120 [0.085, 0.160] | 0.945 [0.920, 0.966] |

The threshold is the score at the validation EER point (clean validation clips, real vs MMS only): 0.057 for the baseline and 11.640 for the CNN. Rates on the test sets at that threshold. False-accept is a fake passed as real, false-reject is a real clip flagged as fake.

| model | condition | false-accept rate | false-reject rate |
|---|---|---|---|
| baseline | seen (MMS) | 0.045 | 0.030 |
| baseline | unseen (Bulbul) | 0.320 | 0.030 |
| baseline | seen, phone channel | 1.000 | 0.000 |
| baseline | unseen, phone channel | 0.975 | 0.000 |
| CNN detector | seen (MMS) | 0.020 | 0.000 |
| CNN detector | unseen (Bulbul) | 0.940 | 0.000 |
| CNN detector | seen, phone channel | 0.370 | 0.000 |
| CNN detector | unseen, phone channel | 0.885 | 0.000 |

Setup facts from the report: 715 training sentences and 83 validation sentences (each with a real and an MMS clip), 200 test sentences, none of them also in the training set. The CNN has 91,249 parameters, ran all 120 epochs in 179 s, and the best epoch by validation EER was 93. The whole script took 626 s.

## Design

The data is paired by sentence. I sample 800 FLEURS hi_in train clips and 200 test clips. MMS speaks the same 800 train sentences and the same 200 test sentences, and Bulbul speaks the same 200 test sentences. For each real clip there is a fake clip with the same text, so what is said cannot separate the classes. Ten percent of the train sentences are held out for validation, along with both their real and fake clips. FLEURS repeats some sentences across speakers, so the split is by sentence text, not by clip.

The report counts 715 + 83 = 798 training sentences, not 800. asrbench prepare dropped two clips for being longer than 30 s (the Kaggle log printed "dropped 2 hi clips longer than 30 s").

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

- Validation EER for the CNN reached 0.000, so checkpoint selection by validation EER was uninformative. Ties were broken by validation loss, which is why epoch 93 was kept, and the validation set (real vs MMS) says nothing about an unseen generator.
- There is one real corpus. Real-side shortcuts (FLEURS microphones, rooms, speakers) can't be separated from fake-side artifacts, because every real clip comes from the same place.
- There are only two generators. One seen and one unseen system is a single data point about generalisation, and Bulbul may be easy or hard for reasons that have nothing to do with TTS in general.
- MMS has a single voice, so "detects MMS's speaker" is a confound. The Bulbul test was meant to expose it, and the CNN's collapse there is consistent with it, though the run doesn't prove it.
- The real side has many speakers and each Bulbul clip uses one of 37 voices.
- It is one training seed and one run, so differences of a few points between conditions are within what I'd expect from noise I haven't measured. The intervals cover resampling of test clips only.
- The phone channel is one fixed degradation applied after synthesis, and FLEURS is read speech, so conversational speech is not covered.
- I did not check the Bulbul voices individually for Hindi quality.

## Next steps

A second real corpus with different recording conditions would show how much of the separation is FLEURS itself. After that, more generators with leave-one-generator-out training would turn the single seen/unseen comparison into an estimate, and calibration across generators (so one threshold works on a system the detector hasn't seen) is the problem the 94% false-accept rate points at.
