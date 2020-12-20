import unittest
from src.features.build_features import tokenize, stem, bag_of_words


class Build_Features_Tests(unittest.TestCase):

    def test_tokenize(self):
        sut = "How are you doing?"
        self.assertEqual(tokenize(sut), ['How', 'are', 'you', 'doing', '?'])

    def test_stem(self):
        words = ["Organize", "organizes", "organizing"]
        sut = [stem(w) for w in words]
        self.assertEqual(sut, ['organ', 'organ', 'organ'])

    def test_bag_of_words(self):
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thanks", "cool"]
        sut = bag_of_words(sentence, words)
        self.assertEqual(sut.tolist(), [0, 1, 0, 1, 0, 0, 0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
