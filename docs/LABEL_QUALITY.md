# FER+ label-quality audit

`affectlens.label_quality` runs five-fold stratified out-of-fold prediction with a class-balanced HOG plus logistic-regression model, then passes those probabilities and the majority FER+ labels to Cleanlab. No sample is scored by a classifier trained on that sample's fold.

The audit covers 23,112 train and validation images. It flags 9,643 (41.72%) as likely label issues. FER+ has genuine annotator disagreement, so a flag is a review candidate, not a confirmed error. The feature model is intentionally independent and lightweight; its out-of-fold accuracy was 48.00% and macro-F1 was 37.58%. See [`reports/cleanlab-label-audit.json`](../reports/cleanlab-label-audit.json) and the ranked [`data/flagged_labels.csv`](../data/flagged_labels.csv).

For the controlled filtering experiment, the top-ranked 10% of train-plus-validation examples are removed. No labels are automatically changed, and the test split is kept untouched. The cleanlab and original hard-label compact models use the same split, seed, augmentation, epoch budget, and training configuration. Results will be reported after both exported models are evaluated on the same internal and official PublicTest splits; filtering should not be presented as an improvement unless those measurements support it.

```bash
python -m affectlens.label_quality --data data/fer-2013-ferplus --output data/flagged_labels.csv --report reports/cleanlab-label-audit.json
python -m affectlens.filter_label_issues --data data/fer-2013-ferplus --issues data/flagged_labels.csv --output data/fer-2013-ferplus-cleaned --fraction 0.10
```
