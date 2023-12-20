import data_processing
from tqdm import tqdm
import random


class Dataset:

    def __init__(self, subjects=['CS'], context_size=3):

        self.data = data_processing.process_data(subjects)
        self.data_size = min([len(self.data[i]) for i in range(len(self.data))])
        self.context_size = context_size
        self.end_description_token = "<end>"
        self.unknown_word_token = "<unk>"

        self.common_query_starts = ["Tell me more about", "Want to learn about", "I want to explore"]
        self.common_response_starts = ["You should consider taking", "I would definitely recommend"]
        assert self.context_size < self.data_size

        self.dataset, self.sentences = self.build_dataset()
        self.vocabulary, self.classes = self.inspect_dataset()

    def build_dataset(self):
        dataset = []
        # common_query_starts = ["Tell me more about", "Want to learn about", "I want to explore"]
        # common_response_starts = ["You should consider taking", "I would definitely recommend"]

        num_epochs = 50
        for _ in tqdm(range(num_epochs)):
            query_start = random.choice(self.common_query_starts)
            response_start = random.choice(self.common_response_starts)

            # random_subject = random.choice(self.data)
            # print(random_subject)
            # random_course = random.choice(list(random_subject.items()))[0]
            # print(list(random_subject.items()))
            # print(random_course)

            # loop over all the subjects
            for random_subject in self.data:
                for course in list(random_subject.items()):

                    random_course = course[0]
                    words, word = [], ""

                    for i in random_subject[random_course][0]:
                        if i == ' ':
                            words.append(word)
                            word = ''
                        else:
                            word = word + i
                    if word != '':
                        words.append(word)

                    if len(words) <= self.context_size:
                        continue

                    # get random word
                    representative_word_idx = random.randint(0, len(words) - self.context_size)
                    random_course = "".join((e for e in random_course if e != " "))
                    # dataset.append({f"{query_start} {' '.join((e for e in [words[i] for i in range(representative_word_idx, representative_word_idx + 3)]))}": f"{response_start} {random_course}"})
                    dataset.append({
                                       f"{' '.join((e for e in [words[i] for i in range(representative_word_idx, representative_word_idx + self.context_size)]))}": f"{random_course}"})

        sentences = []
        for example in dataset:
            for context in example:
                sentences.append(f"{context} {self.end_description_token} {example[context]}")

        return dataset, sentences

    def inspect_dataset(self):
        # Return the distinct classes, vocabulary
        vocab = set()
        classes = set()

        for training_example in self.dataset:
            for sentence in training_example:
                unique_words = data_processing.split_sentence(sentence)
                class_name = training_example[sentence][::-1][0: (sentence.find(' ', sentence.find(' ') + 1))][::-1]

                for word in unique_words:
                    vocab.add(word)
                classes.add(class_name)
        vocab.add(self.unknown_word_token)
        vocab.add(self.end_description_token)

        return vocab, classes

    def preprocess_sentences(self):
        data = []
        correct_output = []
        for sentence in self.sentences:
            distinct_words = []
            word = ""
            for i in range(len(sentence)):
                if sentence[i] != " ":
                    word = word + sentence[i]
                else:
                    distinct_words.append(word)
                    word = ""
            if word != "":
                distinct_words.append(word)
            correct_output.append(distinct_words[-1])
            distinct_words = distinct_words[:-2]
            data.append(distinct_words)

        return data, correct_output
