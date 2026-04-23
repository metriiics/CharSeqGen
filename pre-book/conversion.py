import re
from pathlib import Path

pwd = Path(__file__).parent
booksPath = pwd / 'BookInText'

nameBook = [item.name for item in list(booksPath.iterdir())]

allText = ""

def clsEng(text: str) -> str:
    pattern = r'[a-zA-Z]+'

    clsText = re.sub(pattern, "", text)
    return clsText

for book in nameBook:
    print(f"Process book - {book}")
    with open(booksPath / book, "r", encoding="cp1251") as file:
        text = file.read()

    allText += text
    print("Fine!")

allText = clsEng(allText)

setText = set(allText)
wordText = allText.split()

print(f"Total Lenght: {len(allText)}")
print(f"Unique Characters: {len(setText)}")
print(f"Word Count: {len(wordText)}")

with open(booksPath / "allTxt", "w", encoding="utf-8") as file:
    file.write(allText)