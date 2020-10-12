from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
# import nltk
# nltk.download('punkt')
input_text = "Do you know how tokenization works? It's actually quite interesting! Let's analyze a couple of sentences and figure it out."

print("\ntokenizer:")
s = sent_tokenize(input_text)
w = word_tokenize(input_text)
ww = WordPunctTokenizer().tokenize(input_text)

print(s)
print(w)
print(ww)
