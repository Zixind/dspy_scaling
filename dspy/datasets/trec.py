import random
import os
import json
import numpy as np

from dspy.datasets.dataset import Dataset

class trec:
    def __init__(self):
        official_train = []
        official_test = []
        self.dataset_name = "trec"
        
        data_shuffling_rng = np.random.RandomState(42)
        self.do_shuffle = False
        self.label_mapping = {'0': 'description', '1': 'entity', '2': 'expression', '3': 'human','4': 'location', '5': 'number'}
        print('current working directory {}'.format(os.getcwd()))
        
        # Use absolute path resolution
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Directory of this script
        data_dir = os.path.join(base_dir, "..", "scripts", "data", "ordered_prompt")

        file_path = os.path.join(data_dir, self.dataset_name + ".json")

        print("Resolved file path:", file_path)  # Debugging print
        
        sentence_list, label_list = [], []
        # load dataset from file
        self.dataset = dict(
            train_per_class=dict(),
            train=dict(sentence=[], label=[]),
            dev=dict(sentence=[], label=[]),
            test=dict(sentence=[], label=[]),
        )
        splits = ['train', 'test']
        for split in splits:
            _split = "dev_subsample" if split == "dev" else split
            file_path = os.path.join(
                    data_dir, self.dataset_name, _split + ".jsonl"
            )
            sentence_list, label_list = [], []

            with open(file_path) as fin:
                for line in fin:
                    datapoint = json.loads(line)
                    sentence, label = datapoint["sentence"], datapoint["label"]
                    
                    sentence_list.append(sentence)
                    label_list.append(self.label_mapping[label])

            self.dataset[split]["sentence"] = sentence_list
            self.dataset[split]["label"] = label_list
        
        import dspy
        #using question, answer as input and output        
        trainset = [
            dspy.Example(question=sentence, label=label).with_inputs("question")
            for sentence, label in zip(self.dataset['train']['sentence'], self.dataset['train']['label'])
            ]
        devset = [
            dspy.Example(question=sentence, label=label).with_inputs("question")
            for sentence, label in zip(self.dataset['dev']['sentence'], self.dataset['dev']['label'])
        ]   
        testset = [
            dspy.Example(question=sentence, label=label).with_inputs("question")
            for sentence, label in zip(self.dataset['test']['sentence'], self.dataset['test']['label'])
        ]
        # devset = [dspy.Example(**x).with_inputs("question") for x in self.dataset['dev']]
        # testset = [dspy.Example(**x).with_inputs("question") for x in self.dataset['test']]

        self.train = trainset
        self.dev = devset
        self.test = testset
        
        
        
        
        
        
        