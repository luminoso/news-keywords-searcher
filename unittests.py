import unittest

class TokenizerTests(unittest.TestCase):

    def test_tokenizer(self):
        """
        tests if tokenizer actualy tokenizing words
        """
        from talkdesk import Tokenizer

        tokenizer = Tokenizer()

        text = "aveiro's the best looking city. cities. killing it administrator's"

        tokenized_text = tokenizer.tokenize(text, filter_lemma=False)

        self.assertListEqual(tokenized_text, ['aveiro', "'s", 'the', 'best', 'looking', 'city', '.', 'cities', '.', 'killing', 'it', 'administrator', "'s"])

    def test_lemma_filter(self):
        """
        tests if lemmas are being filtered
        :return:
        """
        from talkdesk import Tokenizer
        tokenizer = Tokenizer()

        text = '    !  123 1.3€ 1€ myself will in has is'

        tokenized_text = tokenizer.tokenize(text)
        print(tokenized_text)
        print(len(tokenized_text))

        self.assertListEqual(tokenized_text, [])


class InputTester(unittest.TestCase):
    def test_multiple_tags(self):

        # do we catch 4 tags?
        input_text = "content:word1 word2 word3 AND title:some really long title AND tag1:tagg tag2:taggg"

        from auxfunctions import parse_input
        parsed_input = parse_input(input_text)

        self.assertEqual(len(parsed_input), 2)

    def test_quoted_search(self):
        input_text = '"President Obama"'

        from auxfunctions import parse_input
        parsed_input = parse_input(input_text)

        query = parsed_input.pop()

        self.assertEqual(query[2], True)

if __name__ == '__main__':
    unittest.main()
