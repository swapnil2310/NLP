import nltk
from nltk.corpus import state_union,stopwords
from nltk.tokenize import PunktSentenceTokenizer ,word_tokenize

#######

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r""" Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}  """
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            
    except Exception as e:
        print(str(e))

process_content()

##########

example_sent = "This is an example showing off stop word filteration."
stop_word = set(stopwords.words("english"))

words = word_tokenize(example_sent)

filtered_words = []

for w in words:
    if w not in stop_word:
        filtered_words.append(w)
print(filtered_words)