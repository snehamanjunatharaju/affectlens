# AffectLens model card

## Model

- Active model: EmotiEffNet-B2, VGGFace2-pretrained and fine-tuned on AffectNet, distributed under Apache-2.0.
- Input: one browser-aligned 260×260 BGR face crop, cubic resized and ImageNet mean/std normalized.
- Output: seven temperature-scaled scores plus a split-conformal candidate set.
- Calibration: temperature and conformal threshold fitted on FER+ validation data; these are not guaranteed to transfer outside this benchmark distribution.
- The model weights are a local artifact and are not included in this repository. SHA-256: `7863032d8c0c0bc259aa0dd7b8ebf6b607af454e7a0ea82be2b37767914ecd0f`.

## Evaluation

On FER+ validation (4,078 images), accuracy was 62.82% and macro-F1 52.14%. On the untouched FER+ test split (4,078 images), accuracy was 63.76% and macro-F1 54.05%. The exact same aligned crops scored 36.98% / 31.83% with the previous CompactFER model. Validation-fitted conformal coverage was 90.04% on validation and 90.49% on test; test mean set size was 2.16, singleton coverage 25.63%, and singleton accuracy 89.19%. These are single-split results and do not establish real-world coverage. See the [validation report](../reports/affectnet-external-validation.json), [same-crop test comparison](../reports/affectnet-external-test.json), and [older CompactFER reports](../reports/model-comparison.json).

These are single-split estimates. This evaluates transfer from the AffectNet-trained model to FER+ crops; it is not an in-domain AffectNet test. FER-2013 has small grayscale crops, class imbalance, label disagreement, no reliable identity IDs, and no demographic fields. Broader real-world performance and subgroup fairness have not been evaluated.

## Intended use

Voluntary experimentation and educational model evaluation on facial expression image datasets. The model reports patterns associated with dataset labels. It cannot determine a person’s inner emotional state.

## Out-of-scope use

Do not use for hiring, education, health care, policing, access control, identity inference, or decisions about individuals. Do not interpret a score as a diagnosis or objective measurement of emotion.

## Limitations and risks

- Low-resolution inputs discard detail; camera crops differ from benchmark crops.
- Scores and conformal coverage can degrade under lighting, pose, occlusion, camera, demographic, and dataset shifts.
- Test-set conformal coverage has not been reported; validation calibration coverage is not a test guarantee.
- The mean validation candidate set size was 2.15 classes. The UI abstains when its conformal set is not a singleton and may show a best-scoring class separately.
- Demographic performance gaps are unknown because the supplied training data lacks appropriate metadata.

## Privacy

The API processes input bytes in memory and does not save them. The browser uses MediaPipe locally to align one face and sends only that crop to the local API. See the [privacy section](../README.md#privacy-and-data-handling).
