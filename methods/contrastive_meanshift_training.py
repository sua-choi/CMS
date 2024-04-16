import argparse
import os

import torch
import numpy as np
import torch.nn as nn
import wandb

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from project_utils.cluster_utils import AverageMeter
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_datasets_with_gcdval, get_class_splits
from project_utils.cluster_and_log_utils import *
from project_utils.general_utils import init_experiment, str2bool

from models.dino import *
from methods.loss import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def train(model, train_loader, test_loader, train_loader_memory, args):

    optimizer = SGD(list(model.module.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * args.eta_min,
        )

    sup_con_crit = SupConLoss()
    unsup_con_crit = ConMeanShiftLoss(args)

    best_agglo_score = 0        
    best_agglo_img_score = 0
    best_img_old_score = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        with torch.no_grad():
            model.eval()
            all_feats = []
            for batch_idx, batch in enumerate(tqdm(train_loader_memory)):
                images, class_labels, uq_idxs, mask_lab = batch
                images = torch.cat(images, dim=0).to(device)

                features = model(images)
                all_feats.append(features.detach().cpu())
            all_feats = torch.cat(all_feats)

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, class_labels, uq_idxs, mask_lab = batch
            class_labels, mask_lab = class_labels.to(device), mask_lab[:, 0].to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            features = model(images)

            classwise_sim = torch.einsum('b d, n d -> b n', features.cpu(), all_feats)

            _, indices = classwise_sim.topk(k=args.k+1, dim=-1, largest=True, sorted=True)
            indices = indices[:, 1:]
            knn_emb = torch.mean(all_feats[indices, :].view(-1, args.k, args.feat_dim), dim=1).to(device)

            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
                f3, f4 = [f[~mask_lab] for f in knn_emb.chunk(2)]
                con_knn_emb = torch.cat([f3, f4], dim=0)
                con_uq_idxs = uq_idxs[~mask_lab]

            else:
                # Contrastive loss for all examples
                con_feats = features
                con_knn_emb = knn_emb
                con_uq_idxs = uq_idxs

            unsup_con_loss = unsup_con_crit(con_knn_emb, con_feats)
            
            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

            # Total loss
            loss = (1 - args.sup_con_weight) * unsup_con_loss + args.sup_con_weight * (sup_con_loss)
        
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        with torch.no_grad():
            model.eval()
            all_feats_val = []
            targets = np.array([])
            mask = np.array([])

            for batch_idx, batch in enumerate(tqdm(test_loader)):
                images, label, _ = batch[:3]
                images = images.cuda()
                
                features = model(images)
                all_feats_val.append(features.detach().cpu().numpy())
                targets = np.append(targets, label.cpu().numpy())
                mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                                else False for x in label]))

        # -----------------
        # Clustering
        # -----------------
        img_all_acc_test, img_old_acc_test, img_new_acc_test, img_agglo_score, estimated_k = test_agglo(epoch, all_feats_val, targets, mask, "Test/ACC", args)
        if args.wandb:
            wandb.log({ 'test/all': img_all_acc_test, 'test/base': img_old_acc_test, 'test/novel': img_new_acc_test,
                        'score/agglo': img_agglo_score, 'score/estimated_k': estimated_k}, step=epoch)

        print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(img_all_acc_test, img_old_acc_test, img_new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()
        torch.save(model.state_dict(), args.model_path)

        if img_agglo_score > best_agglo_img_score:
            torch.save({
                'k': estimated_k,
                'model_state_dict': model.state_dict()}
                , args.model_path[:-3] + f'_best.pt')
            best_agglo_img_score = img_agglo_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--exp_root', type=str, default='/home/suachoi/CMS/log')
    parser.add_argument('--pretrain_path', type=str, default='/home/suachoi/CMS/models')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eta_min', type=float, default=1e-3)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=bool, default=False)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--temperature', type=float, default=0.5)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--inductive', action='store_true')
    parser.add_argument('--wandb', action='store_true', help='Flag to log at wandb')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_class_splits(args)

    args.feat_dim = 768
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['cms'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    if args.wandb:
        wandb.init(project='CMS')
        wandb.config.update(args)


    # --------------------
    # MODEL
    # --------------------
    model = DINO(args)
    model = nn.DataParallel(model).to(device)

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    if args.inductive:
        train_dataset, test_dataset, unlabelled_train_examples_test, val_datasets, datasets = get_datasets_with_gcdval(args.dataset_name, train_transform, test_transform, args)
    else:
        train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)


    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True)
    train_loader_memory = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False)
    if args.inductive:
        test_loader_labelled = DataLoader(val_datasets, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    else:
        test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    

    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, train_loader_memory, args)