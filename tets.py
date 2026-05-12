from utils.CharTokenize import CharTokenizer
from pathlib import Path

path_token = Path.cwd() / "models/tokenConfig"

# corpus = "ПРивет, друг, как ты поживаешь. Я вот тут кручусь верчусь. Совершаю попытки чего то добиться, а ты как?"

# tok = CharTokenizer()
# tok.fit(corpus)

# tok.save(path_token)

tok = CharTokenizer.load(path_token)
print(tok.encode("лаывопр"))