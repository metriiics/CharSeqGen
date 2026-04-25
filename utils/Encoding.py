from collections import Counter
from typing import Set, List, Tuple, Dict
import numpy as np


class BPETok:
    """
        BPE (Byte Pair Encoding) токенизатор.
        
        Реализует классический алгоритм BPE для обучения на корпусе текста
        и последующей токенизации. Поддерживает кодирование текста в последовательность
        числовых ID и обратное декодирование.
    """
    def __init__(self, num_merges: int = 4) -> None:
        """
        Инициализация BPE токенизатора.

        Args:
            num_merges: Количество итераций слияния при обучении.
                       Чем больше значение, тем больше субсловных токенов
                       будет создано. По умолчанию: 4.
        """
        self.num_merges = num_merges
        self._numVocab = None
        self._vocab = None
        self._orderedVocab = None
        self._reverseOrderedVocab = None

    def fit(self, corpus: List) -> Tuple[List[List], Set]:
        """  
        Обучение BPE на корпусе текстов.
        
        Выполняет итеративное слияние самых частотных пар символов/токенов
        для построения словаря субсловных единиц.
        
        Args:
            corpus: Список строк для обучения. Каждая строка рассматривается
                   как отдельное слово или фрагмент текста.
                   
        Returns:
            BPETok: Возвращает самого себя для цепочки вызовов.

        Note:
            Каждая итерация добавляет один новый токен в словарь.
            Алгоритм останавливается, когда достигнуто num_merges итераций
            или когда не осталось пар для слияния.
        """
        words = [[char for char in word] for word in corpus]      # список слов корпуса => [['l', 'o', 'w'], ['l', 'o', 'w', 'e', 'r'], ['l', 'o', 'w', 'e', 's', 't']]
        self._vocab = set(''.join(corpus))                              # словарь начальных слов от корпуса 
                                                                # {l, o, w, e, r, s, t}
        for _ in range(self.num_merges):                                
            # Тк. алгоритм итеративный, мы должны совершать итерации для обновления vocab
            pairs = Counter()                                     # каунтре для подсчета самого частого char
            for word in words:                                    # идем по словам корпуса: ['l','o','w'], ['l','o','w','e','r'], ['l','o','w','e','s','t']
                for i in range(len(word) - 1):                    # идем по каждому char в word: l, o, w -> low
                    pair = (word[i], word[i + 1])                 # берем текущий char и по индексу + 1. Создаем пару => 'lo'
                    pairs[pair] += 1                              # считываем частоту этого токена
            if not pairs:
                break     # если пар нет, завершаем программу     

            most_freq = max(pairs, key=pairs.get)                 # возвращаем самый часто встречающийся токен => ('l', 'o')'

            new_token = most_freq[0] + most_freq[1]               # объединяем этот часто встречающийся токен => 'lo'
            self._vocab.add(new_token)                                  # добавляем этот новый токен в наш vocab(словарь) 

            new_words = []                                        # массив с новыми словами
            for word in words:                                    # снова идем по словам из words
                new_word = []                                     # 
                i = 0                                             # итерация = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == most_freq:
                        # если счетчик меньше длины слова - 1 и пара символов является самой частотной
                        # то мы добавляем этот токен в массив с новыми словами и прибавляем к счетчику два
                        new_word.append(new_token)                # добавили в new_word токен - 'lo'
                        i += 2                                    # прибавляем 2 тк. слили два токена(один токен вместо двух)
                    else:
                        new_word.append(word[i])                  # добавляем этот же токен, в нашем  случае для первого слова - 'w'
                        i += 1                                    # прибавляем 1, тк сливаем этот же токен
                new_words.append(new_word)                        # результат первого слова -> ['lo', 'w']
            words = new_words

        self._mappings_vocab()
        return self
    
    def _mappings_vocab(self):
        """
        Построение отображений между токенами и их числовыми ID.
        
        Создаёт:
            - numVocab: словарь токен → ID (ID=0 для UNK)
            - _reverse_vocab: словарь ID → токен
            - _orderedVocab: список токенов, отсортированный по убыванию длины
                         для оптимального жадного поиска при токенизации
        """
        self._numVocab = {ch: idx for idx, ch in enumerate(self._vocab, start=1)}
        self._numVocab["</unk>"] = 0

        self._orderedVocab = dict(sorted(self._numVocab.items(), 
                    key=lambda item: len(item[0]), reverse=True))
        
        self._reverse_vocab = {idx: token for token, idx in self._orderedVocab.items()}
    
    def encode(self, text) -> List:
        """
        Кодирование текста в последовательность числовых ID токенов.
        
        Использует жадный алгоритм: на каждой позиции выбирается самый длинный
        токен из словаря, подходящий к текущей позиции в тексте.
        
        Args:
            text: Входная строка для токенизации.
            
        Returns:
            List[int]: Список числовых ID токенов. ID=0 означает неизвестный токен.
        """
        tokens = []
        i = 0
        text_len = len(text)

        while i < text_len:
            matched = False

            for token in list(self._orderedVocab.keys()):
                token_len = len(token)
                if i + token_len <= text_len and text[i:i+token_len] == token:
                    tokens.append(self._orderedVocab[token])
                    i += token_len
                    matched = True
                    break
            if not matched:
                tokens.append(text[i])
                i += 1
        return tokens
    
    def decode(self, encoded: List) -> str:
        """ 
        Декодирование последовательности ID обратно в текст.
        
        Args:
            encoded: Список числовых ID токенов.
            
        Returns:
            str: Восстановленная строка. 
        """
        tokens = [self._reverse_vocab[ch] for ch in encoded]
        return ''.join(tokens)
 
    @property
    def get_vocab(self) -> Set:
        """ Возвращает словарь """
        return self._vocab

    @property
    def get_size_vocab(self) -> int:
        """ Возвращает размер словаря """
        return len(self._vocab)
    
    @property
    def get_numbeded_vocab(self) -> Dict:
        """ Возвращает нумерованный словарь """
        return self._numVocab

# sample
corpus = ['low', 'lower', 'lowest']

tiktok = BPETok(num_merges=3)
tiktok.fit(corpus)

text = "lowlowerlowest"
text_encoded = tiktok.encode(text)
text_decoded = tiktok.decode(text_encoded)

print(text, " === Encoding ==>", text_encoded)
print(text_encoded, " === Decoding ==>", text_decoded)

size = tiktok.get_size_vocab
print(size)

vocab = tiktok.get_vocab
print(vocab)

numbVocab = tiktok.get_numbeded_vocab
print(numbVocab)