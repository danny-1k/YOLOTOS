class Vocab:
    def __init__(self, classes):
        self.classes = classes
        self.tokens = ["<s>", "</s>", "<p/>", *classes]
        self.n_tokens = len(self.tokens)

        # index 1 always corresponds to EOS and index 0 to SOS

        self.token_ix = {t: i for i, t in enumerate(self.tokens)}
        self.ix_token = {i: t for i, t in enumerate(self.tokens)}

    def encode(self, classes):
        tokens = ["<s>", *classes, "</s>"]

        indices = [self.token_ix[t] for t in tokens]

        return indices

    def decode(self, tokens):
        # tokens of shape (N)

        decoded = []

        for n in tokens:
            decoded.append(self.ix_token[n])

        return decoded


if __name__ == "__main__":

    classes = [
        "Person",
        "Dog",
        "Table",
        "Television"
        #...
    ]

    v =  Vocab(classes=classes)

    print(v.encode(["Table", "Television"]))