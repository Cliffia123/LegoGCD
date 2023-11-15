import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from tqdm import tqdm
import torch.nn.functional as F

from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, initial_qhat, update_qhat, causal_inference, WeightedEntropyLoss 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 2
np.random.seed(seed)
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
       
def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    
            
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
  
    qhat = initial_qhat(class_num=args.num_labeled_classes+args.num_unlabeled_classes)
    
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
                teacher_out = student_out.detach()
                
                #* clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                #* clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                #* represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                #* representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)
                
                #* Soft loss
                unsup_logits = torch.softmax(student_out/0.1, dim=-1)
                max_probs_, idx = torch.max(unsup_logits, dim=-1)
                mask_all = max_probs_.ge(args.thr)

                mask_lab_p = torch.cat([mask_lab, mask_lab], dim=0)
                labels_p = torch.cat([class_labels, class_labels], dim=0)
                mask_p_true = (mask_lab_p == False) & (mask_all == True)
                mask_p = (mask_lab_p == False) & (mask_all == True) & (labels_p == idx)

                mask_old = torch.zeros_like(mask_p_true)
                mask_old[(idx.lt(args.num_labeled_classes)) & (mask_p_true == True)] = 1 #大于0.7，unlabeled，预测正确，old
                mask_old_num = torch.sum(mask_old.int()).item()

                #* Unlabeled True
                mask_condidate_unlabeled = (mask_lab_p == False) & (mask_all == True)
                mask_p_unlabeled_true = (mask_lab_p == False) & (mask_all == True) &(labels_p == idx)
                mask_p_unlabeled_true_old = (mask_lab_p == False) & (mask_all == True) &(labels_p == idx) & (idx.lt(args.num_labeled_classes))
                mask_p_unlabeled_true_novel = (mask_lab_p == False) & (mask_all == True) &(labels_p == idx) & (idx.ge(args.num_labeled_classes))
                
                #* Unlabeled True old ro novel                                                                                            
                Unlabeled_true_to_old = torch.sum(mask_p_unlabeled_true_old.int()).item() #大于0.7且是unlabeled的个数
                Unlabeled_true_to_novel = torch.sum(mask_p_unlabeled_true_novel.int()).item() #大于0.7且是unlabeled的个数
                                                          
                mask_p_unlabeled_wrong = (mask_lab_p == False) & (mask_all == True) &(labels_p != idx)

                Unlabeled_condiate = torch.sum(mask_condidate_unlabeled.int()).item() #大于0.7且是unlabeled的个数
                Unlabeled_true = torch.sum(mask_p_unlabeled_true.int()).item() #大于0.7且是unlabeled的，且预测正确的个
                Unlabeled_wrong = torch.sum(mask_p_unlabeled_wrong.int()).item()
            
                #* Unlabeled Wrong
                idx_ = labels_p[mask_p_unlabeled_wrong]
                idx_pre = idx[mask_p_unlabeled_wrong]
                
                idx_num_old = idx_.lt(args.num_labeled_classes)
                idx_num_novel = idx_.ge(args.num_labeled_classes)
                
                idx_num_novel_to_old = idx_.ge(args.num_labeled_classes) & idx_pre.lt(args.num_labeled_classes)
                idx_num_novel_to_novel = idx_.ge(args.num_labeled_classes) & idx_pre.ge(args.num_labeled_classes)

                num_wrong_old = torch.sum(idx_num_old.int()).item()
                num_wrong_novel = torch.sum(idx_num_novel.int()).item()
                
                num_wrong_novel_to_old = torch.sum(idx_num_novel_to_old.int()).item()
                num_wrong_novel_to_novel = torch.sum(idx_num_novel_to_novel.int()).item()

                
                pseudo_label = torch.softmax((student_out/0.05), dim=-1)
                delta_logits = torch.log(qhat)
                #logits_u = student_out/0.05
                logits_u = student_out/0.05 + 0.4*delta_logits
                log_pred = F.log_softmax(logits_u, dim=-1)
                nll_loss = torch.sum(-pseudo_label*log_pred, dim=1)*mask_old
                
                qhat = update_qhat(torch.softmax(student_out.detach(), dim=-1), qhat, momentum=args.qhat_m)
                
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'nll_loss: {nll_loss.mean().item():.4f} '
               
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                loss += 2*nll_loss.mean()
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {} Unlabeled condidate {} Unlabeled pred Old {} Unabeled true {} Unlabeled true_to_old {} Unlabeled true_to_novel {} Unabeled wrong {} Wrong old {} Wrong novel {} Wrong novel_to_old {} Wrong novel_to_novel {}'
                    .format(epoch, batch_idx, len(train_loader), loss.item(), pstr, Unlabeled_condiate, mask_old_num, Unlabeled_true, Unlabeled_true_to_old, Unlabeled_true_to_novel, Unlabeled_wrong, num_wrong_old, num_wrong_novel,num_wrong_novel_to_old, num_wrong_novel_to_novel))
                
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
        
        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))
        
        if epoch==100:
            qhat_ = qhat.tolist()[0]
            info_marks = pd.DataFrame({'Qhat': qhat_})
            writer = pd.ExcelWriter(os.path.join(args.log_dir, 'qhat_'+args.dataset_name+'_count_100.xlsx'))
            info_marks.to_excel(writer)
            writer.save()
            print('DataFrame is written successfully to the Excel File.')
    qhat_ = qhat.tolist()[0]
    info_marks = pd.DataFrame({'Qhat': qhat_})
    writer = pd.ExcelWriter(os.path.join(args.log_dir, 'qhat_'+args.dataset_name+'_count.xlsx'))
    info_marks.to_excel(writer)
    writer.save()
    print('DataFrame is written successfully to the Excel File.')

def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--thr', type=float, default=0.7)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument('--masked-qhat', action='store_true', default=False,
                    help='update qhat with instances passing a threshold')
    parser.add_argument('--qhat_m', default=0.999, type=float,
                    help='momentum for updating q_hat')
    parser.add_argument('--e_cutoff', default=-5.4, type=int)
    parser.add_argument('--use_marginal_loss', default=False)
    parser.add_argument('--tau', default=0.4, type=float)


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['legogcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    # args.mlp_out_dim = args.num_labeled_classes
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, train_labelled = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    # train_unlabeled = unlabelled_train_examples_test
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
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    
    model = nn.Sequential(backbone, projector).to(device)
    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
