class Tokenizer:
    def __init__(self, text):
        # clean and sort data
        cleaned = self.clean_text(text)
        self.chars = sorted(set(cleaned))
        # vocab size
        self.vocab_size = len(self.chars)

        # vocab map
        self.encode = {}
        self.decode = {}

        for i, chars in enumerate(self.chars):
            self.encode[chars] = i
            self.decode[i] = chars

        # sanity check
        print("Vocab chars:", self.chars)
        print("Vocab size:", self.vocab_size)

    # simplify
    # all letters lowercase
    # each punctuation into a token each letter a token
    # get a set of tokens
    # this set is d_vocab

    # clean the text
    def clean_text(self, text: str) -> list[str]:
        return [x for x in text.lower() if x.isalpha() or x in " .!?"]

    # encoder
    def tokenize(self, text):
        # will update words to nums with vocab map 
        cleaned = self.clean_text(text)

        tokens = []

        for char in cleaned:
            if char in self.encode:
                tokens.append(self.encode[char])

        print("Encoded tokens:", tokens)
        return tokens

    # decoder 
    def detokenize(self, tokens):
        # inverse of tokenize (nums to words )
        words = []
        for id in tokens:
            if id in self.decode:
                words.append(self.decode[id])

        final = "".join(words)
        print("Decoded text:", final)

        return final
