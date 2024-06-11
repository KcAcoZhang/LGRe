# LGRe
Temporal Knowledge Graph Completion.
The official code of LGRe submitted to ICONIP 2024
Learning Granularity Representation for Temporal Knowledge Graph Completion (Submitted)

## Dependencies
The required framework and other libraries can be found in the requirements.txt.

## Commands
### Training and Testing:
All hyper-parameter settings can be found in our paper.

ICEWS14
```
python main_tt.py -d YAGO --description yago_hard --max-epochs 30 --oracle-epochs 20 --valid-epochs 5 --alpha 0.1 --lambdax 2 --batch-size 1024 --lr 0.001 --oracle_lr 0.001 --oracle_mode hard --save_dir SAVE --eva_dir SAVE --k 15 --beta 0.6 --dropout 0.2 --gamma 0.1 --static False --time_span 1 --timestamps 189
```
ICEWS18
```
python main_tt.py -d ICEWS18 --description icews18_soft --max-epochs 50 --oracle-epochs 20 --valid-epochs 10 --alpha 0.2 --lambdax 2 --batch-size 1024 --lr 0.001 --oracle_lr 0.001 --oracle_mode soft --save_dir SAVE --eva_dir SAVE --k 45 --beta 0.6 --gamma 0.1 --dropout 0.2
```
YAGO11k
```
python main_tt.py -d ICEWS14T --description icews14T_soft --max-epochs 50 --oracle-epochs 20 --valid-epochs 10 --alpha 0.2 --lambdax 2 --batch-size 1024 --lr 0.001 --oracle_lr 0.001 --oracle_mode soft --save_dir SAVE --eva_dir SAVE --k 45 --beta 0.6 --gamma 0.1 --dropout 0.2 --static False --time_span 24 --timestamps 365
```
Wikida12k
```
python main_tt.py -d GDELT --description gdelt_soft --max-epochs 30 --oracle-epochs 20 --valid-epochs 10 --alpha 0.2 --lambdax 2 --batch-size 1024 --lr 0.001 --oracle_lr 0.001 --oracle_mode soft --save_dir SAVE --eva_dir SAVE --k 15 --beta 0.6 --gamma 0.1 --dropout 0.2 --time_span 15 --timestamps 2976 --static False
```
## Acknowledge
The basic framework of our code is referenced from SANe, and the original datasets can be found here: https://github.com/codeofpaper/SANe.

## Citation
```
@article{zhang-etal-2024-lgre,
title = {Learning Granularity Representation for Temporal Knowledge Graph Completion},
year = {2024},
author = {Jinchuan Zhang, Ming Sun, Qian Huang, Ling Tian},
keywords = {Temporal knowledge graphs; Extrapolation; Representation learning; Contrastive learning}
}
```
