import torch
import gensim
from tqdm import tqdm
import dataset


class Model(torch.nn.Module):

    def __init__(self, data):
        super().__init__()
        self.dataset = data
        self.sentences, self.correct_outputs = data.preprocess_sentences()
        self.embeddings = gensim.models.Word2Vec(self.sentences, min_count=1, vector_size=150)
        self.num_classes = len(self.dataset.classes)
        self.training_data_words = self.dataset.vocabulary

        self.class_to_number = {}
        for i in range(self.num_classes):
            self.class_to_number[list(self.dataset.classes)[i]] = i

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.hidden1 = torch.nn.Linear(150, 1000)
        self.hidden2 = torch.nn.Linear(1000, self.num_classes)
        self.activation = torch.nn.ReLU()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, sentence):
        sentence = self.activation(self.hidden1(sentence))
        return self.hidden2(sentence)

    def test(self, sentence):
        self.eval()
        embedded_test_sentence = [self.embeddings.wv.get_vector(j) for j in sentence if
                                  j in self.training_data_words]
        individual_word_predictions = []
        for j in range(len(embedded_test_sentence)):
            model_output = self.forward(torch.tensor(embedded_test_sentence[j]))
            individual_word_predictions.append(torch.argmax(model_output).item())
        prediction = torch.mode(torch.tensor(individual_word_predictions)).values.item()
        return prediction

    def train_model(self):
        num_epochs = 100
        for _ in tqdm(range(num_epochs)):
            self.train()
            num_iter = len(self.sentences)

            for i in range(num_iter):
                sentence = self.sentences[i]
                # print(sentence)

                embedded_sentence = [self.embeddings.wv.get_vector(j) for j in sentence]

                individual_word_predictions = []
                for j in range(len(embedded_sentence)):
                    model_output = self.forward(torch.tensor(embedded_sentence[j]))
                    individual_word_predictions.append(torch.argmax(model_output).item())

                prediction = torch.mode(torch.tensor(individual_word_predictions)).values.item()
                # print(f"Prediction {prediction}")
                true_label = self.class_to_number[self.correct_outputs[i]] if self.correct_outputs[i] in self.class_to_number else -1
                # print(f"True Label{true_label}")
                loss = self.loss_function(torch.tensor([float(prediction)], requires_grad=True).float(), torch.tensor([float(true_label)], requires_grad=True).float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()