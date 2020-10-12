from nltk.corpus import brown
# import nltk
# nltk.download('brown')


# make sentences containing 700 words each
def chunker(input, size):
    chunks = input.split(" ")
    count = 0
    output = []
    chunk_list = []

    for word in chunks:
        chunk_list.append(word)
        if count == size:
            output.append(' '.join(chunk_list))
            count = 0
            chunk_list = []
            continue
        count += 1
    return output


input_text = " ".join(brown.words()[: 12000])
chunks = chunker(input_text, 700)

print(len(chunks))

for i, sent in enumerate(chunks):
    print(i, ' ==> ', sent[: 50], '\r\n')

##########################################################################################################
