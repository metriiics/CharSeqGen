from utils.CharTokenize import CharTokenizer
from pathlib import Path

corpus = "ПРивет, друг, как ты поживаешь. Я вот тут кручусь верчусь. Совершаю попытки чего то добиться, а ты как?"

# tok = CharTokenizer()
# tok.fit(corpus)

# tok.save()

tok = CharTokenizer.load()
print(tok.encode("Hello, мой друг"))