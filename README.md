# Preliminary Evidence of Representation Changes Induced by Associative Memorization on Transformer-based Language Models, Using MeMo Models

---

##### By MSc. Jesús Armenta-Segura


This is the repo to reproducibility the experiments presented at the paper "Preliminary Evidence of Representation Changes Induced by Associative Memorization on Transformer-based Language Models, Using MeMo Models" (TO BE PUBLISHED)

· UPDATE MAY-31: This repo has become public!

---

### Repo map:

```text
├── MeMoPyTorch/              ### (lil) modified MeMo from Zanzotto et. al.
|    ├── evaluating_memo.py
|    ├── modelling_memo.py
|    ├── modelling_memo_embedding.py
|    ├── modelling_memo_exception.py
|    ├── modelling_memo_layer.py
|    ├── modelling_memo_tokenizer.py
|    └── modelling_memo_tokenizer_original.py
├── GPT2_with_memo.py        ### MeMoGPT-2 implementation.
├── Vanilla_experiments.py   ### Module for GPT-2 vanilla ds training.
├── data_curation.py         ### Module for SimpleText Dataset processing.
├── experiment_setting.py    ### Module for general downstream training.
├── main.py                  ### Main script.
└── README.md
```

### Instructions:
1. Fork it if you need it, and download it on a suitable service (e.g. a google cloud environment, or your local computer).
2. Using a terminal, go to the download path and initialize a virtual enviroment. Then, install all required packages:
```bash
~/path/where/you/downloaded $ python3.11 -m venv my_venv
~/path/where/you/downloaded $ source my_venv/bin/activate
(my_venv) ~/path/where/you/downloaded $ python3.11 -m pip install -r requirements.txt
```
3. Run main.py in order to reproduce all experiments
```bash
(my_venv) ~/path/where/you/downloaded $ python3.11 main.py
```

### Hardware and Software specs for reproducibility:
1. OS: Ubuntu 20.04
2. Python version: 3.11.4
3. Cuda 13
4. 64gb RAM
5. 12gb VRAM, NVidia RTX GeForce 5070
