# DQN4P2Pmarket
Master thesis' appendix. It includes the environment implemented according to the OpenAI Gym framework and the DQN algorithm implemented using PyTorch. The thesis is available in the `MScThesis.pdf` document, and chapters 5, 6, 7 have separate directories in the repository.

## Structure of the repository

```bash
├─── chapter5
│   ├─── env
│   └─── runs
├───chapter6
│   ├─── env
│   ├─── runs
└─── chapter7
    ├─── env
    └─── runs
```

- `env` directories contain the environments used in each chapter.
- `runs` directories contain the results for each each trained.
- `train_{}.py` scripts contains the DQN algorithm implemented.
- `results.ipynb` is a jupyternotebook with the results analyses.
