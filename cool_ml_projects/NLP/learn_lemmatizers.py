from nltk.stem import WordNetLemmatizer
# import nltk
# nltk.download('wordnet')

names = ['writing', 'calves', 'be', 'branded', 'horse', 'randomize',
         'possibly', 'provision', 'hospital', 'kept', 'scratchy', 'code']


lemmatizer = WordNetLemmatizer()

# display part
stemmer_names = ['Lemmatizer-noun', 'Lemmatizer-verb']
formatted_text = '{:>16}' * (len(stemmer_names) + 1)

print('\n', formatted_text.format('INPUT WORD', *stemmer_names), '\n', '=' * 68)
for word in names:
    output = [word, lemmatizer.lemmatize(word, pos='n'), lemmatizer.lemmatize(word, pos='v')]
    print(formatted_text.format(*output))
