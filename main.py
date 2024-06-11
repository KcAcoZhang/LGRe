# Acknowledgement: The basic framework is referenced from SANE (https://github.com/codeofpaper/SANe).
import os

GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
import torch
import argparse
import random
import numpy as np
from pprint import pprint
from helper import get_logger
from Dataset import KnowledgeGraph, KnowledgeGraphWY
from torch.utils.data import DataLoader
from data_loader import TrainDataset, TestDataset
from model import LGRe
from setproctitle import setproctitle
from regularizers import TL

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Main(object):
    def __init__(self, params):
        self.p = params
        assert self.p.embed_dim == self.p.k_w * self.p.k_h
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
        self.logger.info(vars(self.p))
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = self.add_model()
        self.optimizer = self.add_optimizer(self.model.parameters())

    def load_data(self):
        if self.p.dataset == 'wikidata' or self.p.dataset == 'yago':
            kg = KnowledgeGraphWY(self.p)
        else:
            kg = KnowledgeGraph(self.p)
        self.p.num_ent = kg.n_entity
        self.p.num_rel = kg.n_relation
        self.p.n_year = kg.n_year
        self.p.n_month = kg.n_month
        self.p.n_day = kg.n_day

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(kg.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        self.p.chequer_perm = self.get_chequer_perm()

    def add_model(self):
        if self.p.model == 'LGRe':
            model = LGRe(self.p)
        else:
            raise Exception('Please Define Model')
        model.to(self.device)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('{}：{}'.format(name, param.size()))
        total_params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        print(f"Total number of parameters: {total_params}")
        return model

    def add_optimizer(self, parameters):
        if self.p.opt == 'adam':
            return torch.optim.Adam(parameters, lr=self.p.lr)
        else:
            return torch.optim.SGD(parameters, lr=self.p.lr)

    def fit(self):
        self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0
        save_path = os.path.join('./torch_saved', self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')
            self.evaluate('test', self.best_epoch)

        for epoch in range(self.best_epoch, self.p.max_epochs):
            train_loss = self.run_epoch(epoch)
            val_results = self.evaluate('valid', epoch)

            if val_results['mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
            self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5}, \n\n\n'.format(epoch, train_loss,
                                                                                                   self.best_val_mrr))

        self.logger.info('Loading best model with epoch {}, evaluating on test data'.format(self.best_epoch))
        self.load_model(save_path)
        self.evaluate('test')

    def save_model(self, save_path):
        state = {
            # 'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p),
            'model': self.model
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        state = torch.load(load_path)
        self.best_val_mrr = state['best_val']['mrr']
        self.best_val = state['best_val']
        self.model = state['model'].to(self.device)
        self.optimizer = self.add_optimizer(self.model.parameters())
        self.optimizer.load_state_dict(state['optimizer'])
        self.best_epoch = state['best_epoch']

    def run_epoch(self, epoch):
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        time_reg = self.p.treg
        self.time_reg = TL(time_reg)
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            sub, rel, obj, year, month, day, label, neg_net, sub_samp = self.read_batch(batch, 'train')
            pred, loss_t = self.model.forward(sub, rel, year, month, day, neg_net, self.p.train_strategy)
            l_t = self.time_reg.forward(loss_t)
            loss_bce = self.model.loss(pred, label)
            loss = loss_bce + l_t
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if step % 100 == 0:
                self.logger.info(
                    '[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}, \t{}'.format(epoch, step, np.mean(losses),
                                                                                self.best_val_mrr, self.p.name))
        loss = np.mean(losses)
        self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss

    def read_batch(self, batch, split):
        if split == 'train':
            triple, label, neg_ent, sub_samp = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], triple[:, 3], triple[:, 4], triple[:, 5], label, neg_ent, sub_samp
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], triple[:, 3], triple[:, 4], triple[:, 5], label

    def evaluate(self, split, epoch=0):
        self.model.eval()
        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = self.get_combined_results(left_results, right_results)
        if split == 'test':
            self.logger.info('Test: H@10 {:.5}, H@3 {:.5}, H@1 {:.5}, MRR {:.5}, MR {:.5}'.format(results['hits@10'],
                                                                                                  results['hits@3'],
                                                                                                  results['hits@1'],
                                                                                                  results['mrr'],
                                                                                                  results['mr']))
        elif split == 'valid':
            self.logger.info('Valid: H@10 {:.5}, H@3 {:.5}, H@1 {:.5}, MRR {:.5}, MR {:.5}'.format(results['hits@10'],
                                                                                                   results['hits@3'],
                                                                                                   results['hits@1'],
                                                                                                   results['mrr'],
                                                                                                   results['mr']))
        self.logger.info(
            '[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'],
                                                                                 results['right_mrr'], results['mrr']))
        return results

    def predict(self, split='valid', mode='tail_batch'):
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, year, month, day, label = self.read_batch(batch, split)
                pred, _ = self.model.forward(sub, rel, year, month, day, None, 'one_to_n')
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(label.byte(), torch.zeros_like(pred), pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                if step % 100 == 0:
                    self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
        return results

    def get_combined_results(self, left_results, right_results):
        results = {}
        count = float(left_results['count'])

        results['left_mr'] = round(left_results['mr'] / count, 5)
        results['left_mrr'] = round(left_results['mrr'] / count, 5)
        results['right_mr'] = round(right_results['mr'] / count, 5)
        results['right_mrr'] = round(right_results['mrr'] / count, 5)
        results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
        results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

        for k in range(10):
            results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
            results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
            results['hits@{}'.format(k + 1)] = round(
                (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
        return results

    def get_chequer_perm(self):
        self.p.perm = 1
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])  # 同上

        comb_idx = []
        for k in range(self.p.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.k_h):
                for j in range(self.p.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parser For Arguments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', dest="dataset", default='icews14', help='Dataset to use for the experiment')
    parser.add_argument("--name", help='Name of the experiment')
    parser.add_argument("--gpu", type=str, default='0', help='GPU to use, set -1 for CPU')
    parser.add_argument("--opt", type=str, default='adam', help='Optimizer to use for training')
    parser.add_argument('--batch', dest="batch_size", default=256, type=int, help='Batch size')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning Rate')
    parser.add_argument("--epoch", dest='max_epochs', default=500, type=int, help='Maximum number of epochs')
    parser.add_argument("--num_workers", type=int, default=10, help='Maximum number of workers used in DataLoader')
    parser.add_argument('--seed', dest="seed", default=123, type=int, help='Seed to reproduce results')
    parser.add_argument('--restore', dest="restore", action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument("--lbl_smooth", dest='lbl_smooth', default=0.1, type=float,
                        help='Label smoothing for true labels')
    parser.add_argument("--embed_dim", type=int, default=200,
                        help='Embedding dimension for entity and relation')
    parser.add_argument('--k_h', dest="k_h", default=20, type=int, help='Height of the reshaped matrix')
    parser.add_argument('--k_w', dest="k_w", default=10, type=int, help='Width of the reshaped matrix')
    parser.add_argument('--num_filt', dest="num_filt", default=64, type=int, help='Number of filters in convolution')
    parser.add_argument('--ker_sz', dest="ker_sz", default=3, type=int, help='Kernel size to use')
    parser.add_argument('--hid_drop', dest="hid_drop", default=0.4, type=float, help='Dropout for Hidden layer')
    parser.add_argument('--feat_drop', dest="feat_drop", default=0.3, type=float, help='Dropout for Feature')
    parser.add_argument('--inp_drop', dest="inp_drop", default=0.1, type=float, help='Dropout for Input layer')
    parser.add_argument('--logdir', dest="log_dir", default='./log/', help='Log directory')
    parser.add_argument('--config', dest="config_dir", default='./config/', help='Config directory')
    parser.add_argument('--model', default='LGRe', help='')
    parser.add_argument("--train_strategy", type=str, default='one_to_x', help='Training strategy to use')
    parser.add_argument('--neg_num', dest="neg_num", default=1000, type=int,
                        help='Number of negative samples to use for loss calculation')
    #* add temporal regularization
    parser.add_argument('--treg', dest="treg", default=1e-5, type=float, help='temporal regularization rate for training')

    args = parser.parse_args()
    setproctitle(args.name)
    seed_torch(args.seed)

    model = Main(args)
    model.fit()