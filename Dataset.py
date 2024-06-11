import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict as ddict


class KnowledgeGraph:
    def __init__(self, params):
        self.p = params
        if self.p.dataset == 'icews14':
            self.start_date = '2014-01-01'
        elif self.p.dataset == 'icews05-15':
            self.start_date = '2005-01-01'
        else:
            raise Exception('dataset not found')
        self.start_sec = time.mktime(
            time.strptime(self.start_date, '%Y-%m-%d'))
        self.ent2id = {}
        self.rel2id = {}
        self.n_entity = 0
        self.n_relation = 0
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0

        self.training_time_num = []
        self.training_time_str = []

        self.validation_time_num = []
        self.validation_time_str = []

        self.test_time_num = []
        self.test_time_str = []

        self.id2ent = {}
        self.id2rel = {}

        self.data = ddict(list)
        self.sr2o = {}
        self.sr2o_all = {}

        self.triples = ddict(list)

        # --------
        # time processing
        self.year_set = set()
        self.month_set = set()
        self.day_set = set()
        self.year2id = dict()
        self.month2id = dict()
        self.day2id = dict()
        self.training_facts_time = []
        self.validation_facts_time = []
        self.test_facts_time = []
        self.n_year = 0
        self.n_month = 0
        self.n_day = 0
        # --------

        self.load_dicts()
        self.load_triples()
        self.load_time()
        self.load_filters()

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join('./data/', self.p.dataset, entity_dict_file),
                                  header=None)
        self.ent2id = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.ent2id)
        self.entities = list(self.ent2id.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join('./data/', self.p.dataset, relation_dict_file), header=None)
        self.rel2id = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.rel2id)
        self.rel2id.update({str(rel) + '_reverse': idx + len(self.rel2id) for rel, idx in self.rel2id.items()})

        print('#relation: {}'.format(self.n_relation))

        self.rel2id.update({str(rel) + '_reverse': idx + len(self.rel2id) for rel, idx in self.rel2id.items()})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join('./data/', self.p.dataset, training_file), header=None)
        training_df = np.array(training_df).tolist()
        for triple in training_df:
            event_time = time.strptime(triple[3], '%Y-%m-%d')
            self.year_set.add(event_time.tm_year)
            self.month_set.add(event_time.tm_mon)
            self.day_set.add(event_time.tm_mday)
            end_sec = time.mktime(event_time)
            day = int((end_sec - self.start_sec) / (1 * 24 * 60 * 60))
            self.training_time_num.append(
                [self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]],
                 day])

            self.training_time_str.append(
                [self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]],
                 triple[3]])

        self.n_training_triple = len(self.training_time_num)
        print('#training triple: {}'.format(self.n_training_triple))

        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join('./data/', self.p.dataset, validation_file), header=None)
        validation_df = np.array(validation_df).tolist()
        for triple in validation_df:
            event_time = time.strptime(triple[3], '%Y-%m-%d')
            self.year_set.add(event_time.tm_year)
            self.month_set.add(event_time.tm_mon)
            self.day_set.add(event_time.tm_mday)
            end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
            day = int((end_sec - self.start_sec) / (1 * 24 * 60 * 60))
            self.validation_time_num.append(
                [self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]],
                 day])
            self.validation_time_str.append(
                [self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]], triple[3]])

        self.n_validation_triple = len(self.validation_time_num)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join('./data/', self.p.dataset, test_file), header=None)
        test_df = np.array(test_df).tolist()
        for triple in test_df:
            event_time = time.strptime(triple[3], '%Y-%m-%d')
            self.year_set.add(event_time.tm_year)
            self.month_set.add(event_time.tm_mon)
            self.day_set.add(event_time.tm_mday)
            end_sec = time.mktime(time.strptime(triple[3], '%Y-%m-%d'))
            day = int((end_sec - self.start_sec) / (1 * 24 * 60 * 60))
            self.test_time_num.append(
                [self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]], day])
            self.test_time_str.append(
                [self.ent2id[triple[0]], self.rel2id[triple[1]], self.ent2id[triple[2]], triple[3]])

        self.n_test_triple = len(self.test_time_num)
        print('#test triple: {}'.format(self.n_test_triple))

    def load_filters(self):
        print("creating filtering lists")
        sr2o = ddict(set)
        self.data['train'] = self.training_facts_time
        self.data['valid'] = self.validation_facts_time
        self.data['test'] = self.test_facts_time
        for fact in self.training_facts_time:
            ent1, rel, ent2, year, month, day = fact
            sr2o[(ent1, rel, year, month, day)].add(ent2)
            sr2o[(ent2, rel+self.n_relation, year, month, day)].add(ent1)
        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        for fact in self.validation_facts_time:
            ent1, rel, ent2, year, month, day = fact
            sr2o[(ent1, rel, year, month, day)].add(ent2)
            sr2o[(ent2, rel + self.n_relation, year, month, day)].add(ent1)

        for fact in self.test_facts_time:
            ent1, rel, ent2, year, month, day = fact
            sr2o[(ent1, rel, year, month, day)].add(ent2)
            sr2o[(ent2, rel + self.n_relation, year, month, day)].add(ent1)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        if self.p.train_strategy == 'one_to_n':
            for (sub, rel, year, month, day), obj in self.sr2o.items():
                self.triples['train'].append({'triple': (sub, rel, -1, year, month, day), 'label': self.sr2o[(sub, rel, year, month, day)], 'sub_samp': 1})
        else:
            for sub, rel, obj, year, month, day in self.data['train']:
                rel_inv = rel + self.n_relation
                sub_samp = len(self.sr2o[(sub, rel, year, month, day)]) + len(self.sr2o[(obj, rel_inv, year, month, day)])
                sub_samp = np.sqrt(1 / sub_samp)

                self.triples['train'].append(
                    {'triple': (sub, rel, obj, year, month, day), 'label': self.sr2o[(sub, rel, year, month, day)], 'sub_samp': sub_samp})
                self.triples['train'].append(
                    {'triple': (obj, rel_inv, sub, year, month, day), 'label': self.sr2o[(obj, rel_inv, year, month, day)], 'sub_samp': sub_samp})

        for split in ['test', 'valid']:
            for sub, rel, obj, year, month, day in self.data[split]:
                rel_inv = rel + self.n_relation
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj, year, month, day), 'label': self.sr2o_all[(sub, rel, year, month, day)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub, year, month, day), 'label': self.sr2o_all[(obj, rel_inv, year, month, day)]})

        self.triples = dict(self.triples)


    def load_time(self):
        for i in sorted(list(self.year_set)):
            self.year2id[i] = len(self.year2id)
        for i in sorted(list(self.month_set)):
            self.month2id[i] = len(self.month2id)
        for i in sorted(list(self.day_set)):
            self.day2id[i] = len(self.day2id)
        self.n_year = len(self.year2id)
        self.n_month = len(self.month2id)
        self.n_day = len(self.day2id)
        for i in self.training_time_str:
            head, rel, tail, event_time = i
            event_time = time.strptime(event_time, '%Y-%m-%d')
            year = event_time.tm_year
            month = event_time.tm_mon
            day = event_time.tm_mday
            self.training_facts_time.append(
                (head, rel, tail, self.year2id[year], self.month2id[month], self.day2id[day]))
        for i in self.validation_time_str:
            head, rel, tail, event_time = i
            event_time = time.strptime(event_time, '%Y-%m-%d')
            year = event_time.tm_year
            month = event_time.tm_mon
            day = event_time.tm_mday
            self.validation_facts_time.append(
                (head, rel, tail, self.year2id[year], self.month2id[month], self.day2id[day]))

        for i in self.test_time_str:
            head, rel, tail, event_time = i
            event_time = time.strptime(event_time, '%Y-%m-%d')
            year = event_time.tm_year
            month = event_time.tm_mon
            day = event_time.tm_mday
            self.test_facts_time.append((head, rel, tail, self.year2id[year], self.month2id[month], self.day2id[day]))


class KnowledgeGraphWY:
    def __init__(self, params):
        self.p = params
        self.ent2id = {}
        self.rel2id = {}
        self.n_entity = 0
        self.n_relation = 0
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0

        self.id2ent = {}
        self.id2rel = {}

        self.data = ddict(list)
        self.sr2o = {}
        self.sr2o_all = {}

        self.triples = ddict(list)

        # --------
        # time processing
        self.year_set = set()
        self.training_facts_time = []
        self.validation_facts_time = []
        self.test_facts_time = []
        self.n_year = 0
        self.n_month = 1
        self.n_day = 1
        # --------

        self.load_dicts()
        self.load_triples()
        self.load_filters()

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join('./data/', self.p.dataset, entity_dict_file),
                                  header=None)
        self.ent2id = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.ent2id)
        self.entities = list(self.ent2id.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join('./data/', self.p.dataset, relation_dict_file), header=None)
        self.rel2id = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.rel2id)
        self.rel2id.update({str(rel) + '_reverse': idx + len(self.rel2id) for rel, idx in self.rel2id.items()})

        print('#relation: {}'.format(self.n_relation))

        self.rel2id.update({str(rel) + '_reverse': idx + len(self.rel2id) for rel, idx in self.rel2id.items()})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

    def load_filters(self):
        print("creating filtering lists")
        sr2o = ddict(set)
        self.data['train'] = self.training_facts_time
        self.data['valid'] = self.validation_facts_time
        self.data['test'] = self.test_facts_time
        for fact in self.training_facts_time:
            ent1, rel, ent2, year, month, day = fact
            sr2o[(ent1, rel, year, month, day)].add(ent2)
            sr2o[(ent2, rel+self.n_relation, year, month, day)].add(ent1)
        self.data = dict(self.data)
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        for fact in self.validation_facts_time:
            ent1, rel, ent2, year, month, day = fact
            sr2o[(ent1, rel, year, month, day)].add(ent2)
            sr2o[(ent2, rel + self.n_relation, year, month, day)].add(ent1)

        for fact in self.test_facts_time:
            ent1, rel, ent2, year, month, day = fact
            sr2o[(ent1, rel, year, month, day)].add(ent2)
            sr2o[(ent2, rel + self.n_relation, year, month, day)].add(ent1)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

        if self.p.train_strategy == 'one_to_n':
            for (sub, rel, year, month, day), obj in self.sr2o.items():
                self.triples['train'].append({'triple': (sub, rel, -1, year, month, day), 'label': self.sr2o[(sub, rel, year, month, day)], 'sub_samp': 1})
        else:
            for sub, rel, obj, year, month, day in self.data['train']:
                rel_inv = rel + self.n_relation
                sub_samp = len(self.sr2o[(sub, rel, year, month, day)]) + len(self.sr2o[(obj, rel_inv, year, month, day)])
                sub_samp = np.sqrt(1 / sub_samp)

                self.triples['train'].append(
                    {'triple': (sub, rel, obj, year, month, day), 'label': self.sr2o[(sub, rel, year, month, day)], 'sub_samp': sub_samp})
                self.triples['train'].append(
                    {'triple': (obj, rel_inv, sub, year, month, day), 'label': self.sr2o[(obj, rel_inv, year, month, day)], 'sub_samp': sub_samp})

        for split in ['test', 'valid']:
            for sub, rel, obj, year, month, day in self.data[split]:
                rel_inv = rel + self.n_relation
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj, year, month, day), 'label': self.sr2o_all[(sub, rel, year, month, day)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub, year, month, day), 'label': self.sr2o_all[(obj, rel_inv, year, month, day)]})

        self.triples = dict(self.triples)

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join('./data/', self.p.dataset, training_file), header=None)
        training_df = np.array(training_df).tolist()
        for triple in training_df:
            self.year_set.add(triple[-1])
            head, rel, tail, year = triple
            self.training_facts_time.append((head, rel, tail, year, 0, 0))

        self.n_training_triple = len(self.training_facts_time)
        print('#training triple: {}'.format(self.n_training_triple))

        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join('./data/', self.p.dataset, validation_file), header=None)
        validation_df = np.array(validation_df).tolist()
        for triple in validation_df:
            self.year_set.add(triple[-1])
            head, rel, tail, year = triple
            self.validation_facts_time.append((head, rel, tail, year, 0, 0))

        self.n_validation_triple = len(self.validation_facts_time)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join('./data/', self.p.dataset, test_file), header=None)
        test_df = np.array(test_df).tolist()
        for triple in test_df:
            self.year_set.add(triple[-1])
            head, rel, tail, year = triple
            self.test_facts_time.append((head, rel, tail, year, 0, 0))

        self.n_test_triple = len(self.test_facts_time)
        print('#test triple: {}'.format(self.n_test_triple))

        self.n_year = len(self.year_set)