# CharSeqGen

Проект посимвольной генерации по книгам Достоевского Фёдора Михайловича. 

### Context

В проект используются следующие книги автора:

- Преступление и наказание
- Братья Карамазовы
- Бесы
- Идиот
- Подросток

<figure>
<img src="/figure/InOut.jpg" alt="figure example target input">
<figcaption>Fig. 1. Пример смещения Input and Target</figcaption>
</figure>


### Architecture

```mermaid
flowchart TD

    A[Input]
    --> B[Embedding]
    --> C[LSTM]
    --> D[Dropout]
    --> E[Dense]

    style A fill:orange,color:#fff,stroke:red
    style B fill:orange,color:#fff,stroke:red
    style C fill:orange,color:#fff,stroke:red
    style D fill:orange,color:#fff,stroke:red
    style E fill:orange,color:#fff,stroke:red
```

### Estimated

<div align="center">

| Epoch | Loss |
|---|---|
| 0 | 1.466 |
| 1 | 1.266 |
| 2 | 1.222 |
| 3 | 1.200 |
| 4 | 1.185 |
| 5 | 1.172 |
| 6 | 1.163 |

</div>

<figure>
<img src="/figure/train_loss.png" alt="figure training loss">
<figcaption>Fig. 2. Train-Loss</figcaption>
</figure>


### Structure

```text
project/
│
├── figure/                    # Figures and visualizations
├── models/                    # Neural network architectures
│   ├── tokenConfig/
|   |   ├── token.json
|   |   └── vocab.npy
|   |
|   ├── weight/
|   |   └── model.pth
|   |
│   ├── gen.py
|   └── pipeline.py
|
├── pre-book/                  # Processed text files / datasets
|   └── conversion.py
│
├── utils/                     # Utility functions
|    ├── BytePair.py
|    ├── CharTokenize.py
|    └── figure.py
│
├── TestCases.ipynb            # Testing notebooks
├── WordVec.ipynb              # Word embeddings experiments
├── main.ipynb                 # Main training notebook
├── main.py                    # Main training script
└──
```