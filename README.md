# LGRe
Temporal Knowledge Graph Completion.

The official code of LGRe submitted to ICONIP 2024

Learning Granularity Representation for Temporal Knowledge Graph Completion (Submitted)

## Commands
### Training and Testing:
All hyper-parameter settings can be found in our paper.

ICEWS14
```
python main.py --model SANe --name icews14 --lr 0.001 --data icews14 --train_strategy one_to_x --treg 1e-5
```
ICEWS05-15
```
python main.py --model SANe --name icews05-15 --lr 0.001 --data icews05-15 --k_h 30 --embed_dim 300 --feat_drop 0.2 --hid_drop 0.3 --ker_sz 7 --train_strategy one_to_x --batch 512 --treg 5e-4
```
YAGO11k
```
python main.py --model SANe --name yago --lr 0.001 --data yago --ker_sz 5 --train_strategy one_to_x --treg 5e-6
```
Wikida12k
```
python main.py --model SANe --name wiki --lr 0.001 --data wikidata --ker_sz 5 --train_strategy one_to_x --treg 5e-4
```
## Acknowledge
The basic framework of our code is referenced from SANe, and the original datasets can be found here: https://github.com/codeofpaper/SANe.

## Citation
```
@article{zhang-etal-2024-lgre,
title = {Learning Granularity Representation for Temporal Knowledge Graph Completion},
year = {2024},
author = {Jinchuan Zhang, Ming Sun, Qian Huang, Ling Tian},
keywords = {Temporal Knowledge Graph; Knowledge Graph Completion; Representation Learning; Link Prediction}
}
```
