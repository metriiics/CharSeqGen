from typing import Self, Set, Dict, List
import numpy as np
import numpy.typing as npt 

class CharTokenizer:
    def __init__(self) -> None:
        self._CharSet: Set[str] = None
        self._CharSetSorted: Set[str] = None
        self._char2int: Dict[str, int] = None
        self._CharList: npt.NDArray[np.int32] = None

    def fit(self, corpus: str) -> Self:
        self._CharSet = set(corpus)
        self._CharSetSorted = sorted(self._CharSet)

        self._char2int = {ch:i for i, ch in enumerate(self._CharSetSorted)}
        self._CharList = np.array(self._CharSetSorted)
        return self

    def encode(self, doc: str) -> npt.NDArray[np.int32]:
        tokens = np.array([self._char2int[ch] for ch in doc], dtype=np.int32)
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