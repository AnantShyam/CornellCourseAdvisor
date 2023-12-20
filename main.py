import data_processing
import numpy as np
import random
from tqdm import tqdm
import re
import torch
import gensim
import argparse
import dataset
import model


class Main:

    def __init__(self, init_model, query=[]):
        self.model = init_model
        self.model_embeddings = self.model.embeddings
        self.model_classes = self.model.dataset.classes
        self.correct_outputs = self.model.correct_outputs

        self.query = query

        self.model.train_model()

        model_prediction = self.model.test(self.query)
        course_number_predicted = [key for key in model.class_to_number if model.class_to_number[key] == model_prediction][0]
        model_response_start = random.choice(self.model.dataset.common_response_starts)
        print(f"{model_response_start} {course_number_predicted}")


def extract_information_from_user_arguments(arg):
    result = []
    subject = ""
    for i in arg:
        if i == " ":
            result.append(subject)
            subject = ""
        else:
            subject = subject + i
    if subject != "":
        result.append(subject)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process User Command Line Arguments")
    parser.add_argument("subjects")
    parser.add_argument("query")
    arguments = parser.parse_args()

    user_subjects = extract_information_from_user_arguments(arguments.subjects)
    user_query = extract_information_from_user_arguments(arguments.query)

    sentence_data = dataset.Dataset(subjects=user_subjects, context_size=8)
    # print(sentence_data.dataset)
    # print(model.dataset)
    # print(model.sentences)
    # print(sentence_data.classes)

    model = model.Model(sentence_data)
    # print(model.sentences)
    # print(model.num_classes)
    main = Main(model, user_query)
    # print(main.class_to_number)

