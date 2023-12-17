import data_processing
import numpy as np
import random
from tqdm import tqdm
import re
import torch
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize


class Main:

    def __init__(self, init_model):
        self.model = init_model
        self.model_embeddings = self.model.embeddings
        self.model_classes = self.model.dataset.classes
        self.correct_outputs = self.model.correct_outputs

        self.class_to_number = {}

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)

        for i in range(self.model.num_classes):
            self.class_to_number[list(self.model.dataset.classes)[i]] = i

        self.train()

    def train(self):
        num_epochs = 50
        for _ in tqdm(range(num_epochs)):
            self.model.train()
            num_iter = len(self.model.sentences)

            for i in range(num_iter):
                sentence = self.model.sentences[i]
                # print(sentence)

                embedded_sentence = [self.model_embeddings.wv.get_vector(j) for j in sentence]

                individual_word_predictions = []
                for j in range(len(embedded_sentence)):
                    model_output = self.model(torch.tensor(embedded_sentence[j]))
                    individual_word_predictions.append(torch.argmax(model_output).item())

                prediction = torch.mode(torch.tensor(individual_word_predictions)).values.item()
                # print(f"Prediction {prediction}")
                true_label = self.class_to_number[self.correct_outputs[i]]
                # print(f"True Label{true_label}")

                # print(prediction.shape)
                # print(torch.tensor(true_label).shape)
                loss = self.loss_function(torch.tensor([float(prediction)], requires_grad=True).float(), torch.tensor([float(true_label)], requires_grad=True).float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class Model(torch.nn.Module):

    def __init__(self, dataset):

        super().__init__()
        self.dataset = dataset
        self.sentences, self.correct_outputs = dataset.preprocess_sentences()
        self.embeddings = gensim.models.Word2Vec(self.sentences, min_count=1, vector_size=10)
        self.num_classes = len(self.dataset.classes)

        self.hidden1 = torch.nn.Linear(10, 100)
        self.hidden2 = torch.nn.Linear(100, self.num_classes)
        self.activation = torch.nn.ReLU()

    def forward(self, sentence):
        sentence = self.activation(self.hidden1(sentence))
        return self.hidden2(sentence)


class Dataset:

    def __init__(self, subjects=['CS'], context_size=3, embedding_dim=10):

        self.data = data_processing.process_data(subjects)
        self.data_size = min([len(self.data[i]) for i in range(len(self.data))])
        self.context_size = context_size
        self.end_description_token = "<end>"
        self.unknown_word_token = "<unk>"
        assert self.context_size < self.data_size

        self.dataset, self.sentences = self.build_dataset()
        self.vocabulary, self.classes = self.inspect_dataset()
        self.embedding_dim = embedding_dim

    def build_dataset(self):
        dataset = []
        common_query_starts = ["Tell me more about", "Want to learn about", "I want to explore"]
        common_response_starts = ["You should consider taking", "I would definitely recommend"]

        for _ in tqdm(range(2000)):
            query_start = random.choice(common_query_starts)
            response_start = random.choice(common_response_starts)
            random_subject = random.choice(self.data)
            random_course = random.choice(list(random_subject.items()))[0]

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
            dataset.append({f"{' '.join((e for e in [words[i] for i in range(representative_word_idx,representative_word_idx + self.context_size)]))}": f"{random_course}"})

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


if __name__ == "__main__":
    sentence_data = Dataset(context_size=8)
    # print(model.dataset)
    # print(model.sentences)
    # print(sentence_data.classes)
    model = Model(sentence_data)
    print(model.num_classes)
    main = Main(model)

