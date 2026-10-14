# Third-party attribution

Financial PhraseBank v1.0 is by Pekka Malo, Ankur Sinha, Pekka Korhonen,
Jyrki Wallenius and Pyry Takala.

- Source: https://huggingface.co/datasets/takala/financial_phrasebank
- Paper: https://arxiv.org/abs/1307.5336
- License: CC BY-NC-SA 3.0 — https://creativecommons.org/licenses/by-nc-sa/3.0/

Only the original `Sentences_75Agree.txt` subset is used. Its original README,
license and source metadata are in `data/raw/`. Derived data normalize whitespace,
remove duplicates and conflicting groups, and partition sentences; labels are
not changed. The dataset, learned model and data-derived outputs retain the
CC BY-NC-SA 3.0 conditions. The MIT code license does not relicense them.

Citation: Malo et al. (2014), *Good debt or bad debt: Detecting semantic
orientations in economic texts*, Journal of the Association for Information
Science and Technology, volume 65.

pandas, scikit-learn, joblib, matplotlib, Streamlit and their dependencies are
installed separately; their respective licenses apply.
