from typing import Self, Set, Dict, List
from pathlib import Path
import numpy as np
import numpy.typing as npt 
import json
import os

class CharTokenizer:
    def __init__(self) -> None:
        self._CharSet: Set[str] = None
        self._CharSetSorted: Set[str] = None
        self._char2int: Dict[str, int] = None
        self._CharList: npt.NDArray[np.int32] = None

    def fit(self, corpus: str) -> Self:
        self._CharSet = set(corpus)
        self._CharSetSorted = ['<unk>'] + sorted(self._CharSet)

        self._char2int = {ch:i for i, ch in enumerate(self._CharSetSorted[1:], start=1)}
        self._char2int['<unk>'] = 0
        self._CharList = np.array(self._CharSetSorted)
        return self

    def encode(self, doc: str) -> npt.NDArray[np.int32]:
        tokens = np.array([self._char2int[ch] if ch in self._CharSetSorted 
            else self._char2int['<unk>'] for ch in doc], dtype=np.int32)
        return tokens

    def decode(self, tokens: npt.NDArray[np.int32]) -> str:
        doc = "".join(self._CharList[tokens])
        return doc
    
    @property
    def get_vocab_size(self) -> int:
        return len(self._CharSetSorted)

    @property
    def get_vocab(self) -> Set[str]:
        return self._CharSetSorted
    
    @property
    def get_vocab_from_id(self) -> Dict[str, int]:
        return self._char2int
    
    def save(self) -> None:
        with open(os.path.join(Path.cwd() / "models/tokenConfig/vocab.json"), "w", encoding="utf-8") as file:
            json.dump(self._char2int, file, ensure_ascii=False, indent=2)

        np.save(os.path.join(Path.cwd() / "models/tokenConfig/vocab.npy"), self._CharList)

    @classmethod
    def load(cls) -> Self:
        tokenizer = cls()

        with open(os.path.join(Path.cwd() / "models/tokenConfig/vocab.json"), "r", encoding="utf-8") as file:
            tokenizer._char2int = json.load(file)

        tokenizer._CharList = np.load(os.path.join(Path.cwd() / "models/tokenConfig/vocab.npy"), allow_pickle=True)
        tokenizer._CharSetSorted = set(tokenizer._CharList.tolist())
        return tokenizer