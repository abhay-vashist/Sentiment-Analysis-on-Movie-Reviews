import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenization
    words = nltk.word_tokenize(text)
    # Remove stopwords + stemming
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)
