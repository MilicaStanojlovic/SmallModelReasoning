from pathlib import Path
import urllib.request

url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
Path("data").mkdir(exist_ok=True)
urllib.request.urlretrieve(url, "data/wiki2_train.txt")
print("Downloaded.")