from pathlib import Path

textPath = Path(__file__).parent / 'pre-book/BookInText/allTxt'

with open(textPath, "r", encoding="utf-8") as file:
    text = file.read()

setText = set(text)
wordText = text.split()

print(f"Total Lenght: {len(text)}")
print(f"Unique Characters: {len(setText)}")
print(f"Word Count: {len(wordText)}")