# AffectLens model card

## Model

- Active model: CompactFER trained on FER+ ten-rater soft labels, exported to ONNX.
- Input: one aligned, grayscale 48×48 face crop.
- Output: seven temperature-scaled scores plus a split-conformal candidate set.
- Training: FER-2013 training images annotated by FER+ raters; exact duplicate pixels are deduplicated and split by image content.
- Weights and data are local artifacts and are not included in this repository.

## Evaluation

On the FER+ derived internal holdout, accuracy was 44.14%, macro-F1 36.51%, ECE 0.034, and 90.73% conformal coverage. On the supplied FER-2013 PublicTest split (3,589 images), accuracy was 39.54%, macro-F1 34.93%, ECE 0.038, and coverage was 83.17% at a nominal 90% target. Mean conformal set size was 4.20 of seven labels. See [the full report](../reports/model-comparison.json) and [controlled shift results](../reports/ferplus-publictest-robustness.json).

These are single-split estimates. FER-2013 has small grayscale crops, class imbalance, label disagreement, no reliable identity IDs, and no demographic fields. Cross-dataset performance and subgroup fairness have not been evaluated.

## Intended use

Voluntary experimentation and educational model evaluation on facial expression image datasets. The model reports patterns associated with dataset labels. It cannot determine a person’s inner emotional state.

## Out-of-scope use

Do not use for hiring, education, health care, policing, access control, identity inference, or decisions about individuals. Do not interpret a score as a diagnosis or objective measurement of emotion.

## Limitations and risks

- Low-resolution inputs discard detail; camera crops differ from benchmark crops.
- Scores and conformal coverage can degrade under lighting, pose, occlusion, camera, demographic, and dataset shifts.
- Even on the PublicTest split, nominal 90% conformal coverage measured 83.17%; the advertised target is not met there.
- A mean candidate set of 4.20 classes means the model usually abstains. The UI also shows the highest-scoring class as a best guess, not as a certainty.
- Demographic performance gaps are unknown because the supplied training data lacks appropriate metadata.

## Privacy

The API processes input bytes in memory and does not save them. The browser uses MediaPipe locally to align one face and sends only that crop to the local API. See the [privacy section](../README.md#privacy-and-data-handling).
