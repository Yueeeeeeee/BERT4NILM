from dataset import *
from dataloader import *
from trainer import *
from config import *
from utils import *
from model import BERT4NILM

import argparse
import torch


def train(args, export_root=None, resume=True):
    args.validation_size = 0.1
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [2, 3, 4, 5, 6]
        dataset = REDD_LF_Dataset(args)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [1, 3, 4, 5]
        dataset = UK_DALE_Dataset(args)

    x_mean, x_std = dataset.get_mean_std()
    stats = (x_mean, x_std)

    model = BERT4NILM(args)

    if export_root == None:
        folder_name = '-'.join(args.appliance_names)
        export_root = 'experiments/' + args.dataset_code + '/' + folder_name

    dataloader = NILMDataloader(args, dataset, bert=True)
    train_loader, val_loader = dataloader.get_dataloaders()

    trainer = Trainer(args, model, train_loader,
                    val_loader, stats, export_root)
    if args.num_epochs > 0:
        if resume:
            try:
                model.load_state_dict(torch.load(os.path.join(
                    export_root, 'best_acc_model.pth'), map_location='cpu'))
                print('Successfully loaded previous model, continue training...')
            except FileNotFoundError:
                print('Failed to load old model, continue training new model...')
        trainer.train()

    args.validation_size = 1.
    if args.dataset_code == 'redd_lf':
        args.house_indicies = [1]
        dataset = REDD_LF_Dataset(args, stats)
    elif args.dataset_code == 'uk_dale':
        args.house_indicies = [2]
        dataset = UK_DALE_Dataset(args, stats)

    dataloader = NILMDataloader(args, dataset)
    _, test_loader = dataloader.get_dataloaders()
    rel_err, abs_err, acc, prec, recall, f1 = trainer.test(test_loader)
    print('Mean Accuracy:', acc)
    print('Mean F1-Score:', f1)
    print('Mean Relative Error:', rel_err)
    print('Mean Absolute Error:', abs_err)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    

torch.set_default_tensor_type(torch.DoubleTensor)
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--dataset_code', type=str,
                    default='redd_lf', choices=['redd_lf', 'uk_dale'])
parser.add_argument('--validation_size', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--house_indicies', type=list, default=[1, 2, 3, 4, 5])
parser.add_argument('--appliance_names', type=list,
                    default=['microwave', 'dishwasher'])
parser.add_argument('--sampling', type=str, default='6s')
parser.add_argument('--cutoff', type=dict, default=None)
parser.add_argument('--threshold', type=dict, default=None)
parser.add_argument('--min_on', type=dict, default=None)
parser.add_argument('--min_off', type=dict, default=None)
parser.add_argument('--window_size', type=int, default=480)
parser.add_argument('--window_stride', type=int, default=120)
parser.add_argument('--normalize', type=str, default='mean',
                    choices=['mean', 'minmax'])
parser.add_argument('--denom', type=int, default=2000)
parser.add_argument('--model_size', type=str, default='gru',
                    choices=['gru', 'lstm', 'dae'])
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--mask_prob', type=float, default=0.25)
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
parser.add_argument('--optimizer', type=str,
                    default='adam', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--decay_step', type=int, default=100)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--c0', type=dict, default=None)

args = parser.parse_args()


if __name__ == "__main__":
    fix_random_seed_as(args.seed)
    get_user_input(args)
    set_template(args)
    train(args)
