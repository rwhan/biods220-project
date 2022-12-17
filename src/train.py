import pretrain_utils
import train_utils

import pandas as pd
import argparse
import os

ROOT = "/home/ec2-user/biods220/project/assign2"
if not os.path.exists('output'):
    os.makedirs('output')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--forward', type=int, default = 0)
parser.add_argument('--trend', type=int, default = 0)
parser.add_argument('--dropout', type=float, default = 0.0)
parser.add_argument('--features', type=str, default = 'all')
parser.add_argument('--no_build', action='store_false')
parser.add_argument('--no_pretrain', action='store_false')

pretrain_target = vars(parser.parse_args())
build = pretrain_target['no_build']
pretrain_target.pop('no_build')

pretrain = pretrain_target['no_pretrain'] and ((pretrain_target['forward'] != 0) or (pretrain_target['trend'] != 0) or (pretrain_target['dropout'] != 0))
pretrain_target.pop('no_pretrain')

model, num_pretrain_labels, pretrain_features = pretrain_utils.run_pretrain(pretrain_target, build =build, run = pretrain)
if pretrain:
    model.save('output/' + str(pretrain_target) + '_pretrained.h5')

results = pd.DataFrame(columns = ['condition', 'label fraction', 'pretrain', 'seed', 'le_auroc', 'ft_auroc'])
for target in ['SEPSIS', 'VANCOMYCIN', 'MI']:
    for label_frac, num_seeds in [(0.1, 3), (0.25, 3), (1, 1)]:
        for seed in range(num_seeds):
            le_auroc, ft_auroc = train_utils.run_experiment(target, pretrain_target, pretrain_features, num_pretrain_labels, label_frac, pretrain, seed)
            new_results = {'condition': [target], 'label fraction': [label_frac], 'pretrain': [pretrain], 'seed': seed, 'le_auroc': [le_auroc], 'ft_auroc': [ft_auroc]}
            results = pd.concat([results, pd.DataFrame(new_results)])

results.to_csv('output/' + str(pretrain_target) + '_results.csv')