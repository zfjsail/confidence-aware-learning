import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import time

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from dataset.venue_loader import VenueCNNMatchDataset
from model.cnn_match import CNNMatchModel
from utils.data_utils import ChunkSampler
import utils_orig
import crl_utils

from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

parser = argparse.ArgumentParser(description='Confidence Aware Learning')
# parser.add_argument('--epochs', default=300, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--data', default='cifar10', type=str, help='Dataset name to use [cifar10, cifar100, svhn]')
parser.add_argument('--model', default='res', type=str, help='Models name to use [res, dense, vgg]')
parser.add_argument('--rank_target', default='softmax', type=str,
                    help='Rank_target name to use [softmax, margin, entropy]')
parser.add_argument('--rank_weight', default=0.0, type=float, help='Rank loss weight')
parser.add_argument('--data_path', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./test/', type=str, help='Savefiles directory')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
parser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
parser.add_argument('--mat1-channel1', type=int, default=8, help='Matrix1 number of channels1.')
parser.add_argument('--mat1-kernel-size1', type=int, default=3, help='Matrix1 kernel size1.')
parser.add_argument('--mat1-channel2', type=int, default=16, help='Matrix1 number of channel2.')
parser.add_argument('--mat1-kernel-size2', type=int, default=2, help='Matrix1 kernel size2.')
parser.add_argument('--mat1-hidden', type=int, default=512, help='Matrix1 hidden dim.')
parser.add_argument('--mat2-channel1', type=int, default=8, help='Matrix2 number of channels1.')
parser.add_argument('--mat2-kernel-size1', type=int, default=2, help='Matrix2 kernel size1.')
parser.add_argument('--mat2-hidden', type=int, default=512, help='Matrix2 hidden dim')
parser.add_argument('--build-index-window', type=int, default=5, help='Matrix2 hidden dim')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--initial-accumulator-value', type=float, default=0.01, help='Initial accumulator value.')
parser.add_argument('--weight-decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch', type=int, default=64, help="Batch size")
parser.add_argument('--dim', type=int, default=64, help="Embedding dimension")
parser.add_argument('--check-point', type=int, default=2, help="Check point")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--file-dir', type=str, default=settings.VENUE_DATA_DIR, help="Input file directory")
parser.add_argument('--train-ratio', type=float, default=60, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=10, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=False,
                    help="Adjust weights inversely proportional"
                         " to class frequencies in the input data")

args = parser.parse_args()


def evaluate(epoch, loader, model, thr=None, return_best_thr=False, args=args):
    model.eval()
    total = 0.
    loss = 0.
    y_true, y_pred, y_score = [], [], []

    for i_batch, batch in enumerate(loader):
        X_title, X_author, Y, Y_onehot, idx = batch
        bs = len(Y)

        if args.cuda:
            X_title = X_title.cuda()
            X_author = X_author.cuda()
            Y = Y.cuda()

        output, _ = model(X_title.float(), X_author.float())
        loss_batch = F.nll_loss(output, Y.long())
        loss += bs * loss_batch.item()

        y_true += Y.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                loss / total, auc, prec, rec, f1)

    if return_best_thr:  # valid
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr, [loss / total, auc, prec, rec, f1]
    else:
        return None, [loss / total, auc, prec, rec, f1]


def train(loader, valid_loader, test_loader, model, criterion_cls, criterion_ranking, optimizer, epoch, history, logger_w,
          args):
    batch_time = utils_orig.AverageMeter()
    data_time = utils_orig.AverageMeter()
    total_losses = utils_orig.AverageMeter()
    top1 = utils_orig.AverageMeter()
    cls_losses = utils_orig.AverageMeter()
    ranking_losses = utils_orig.AverageMeter()
    end = time.time()

    model.train()
    for i, batch in enumerate(loader):
        data_time.update(time.time() - end)
        x1, x2, labels, labels_onehot, idx = batch

        if args.cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
            labels = labels.cuda()
            labels_onehot = labels_onehot.cuda()
            idx = idx.cuda()

        output, logits = model(x1.float(), x2.float())

        # compute ranking target value normalize (0 ~ 1) range
        # max(softmax)
        if args.rank_target == 'softmax':
            conf = F.softmax(logits, dim=1)
            confidence, _ = conf.max(dim=1)

        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        # calc target, margin
        rank_target, rank_margin = history.get_target_margin(idx, idx2)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

        # ranking loss
        ranking_loss = criterion_ranking(rank_input1,
                                         rank_input2,
                                         rank_target)

        # total loss
        cls_loss = criterion_cls(output, labels)
        # print("cls loss", cls_loss)
        prec, correct = utils_orig.accuracy(output, labels)

        if epoch > 1:  # [4, 5]
            # sample_weight = torch.Tensor(4 - history.correctness[idx]/history.max_correctness)
            sample_weight = torch.Tensor(3 - correct.float())
        else:
            sample_weight = torch.ones_like(cls_loss)
        sample_weight = sample_weight / torch.sum(sample_weight)
        # print("sample weight", sample_weight)
        cls_loss = torch.sum(cls_loss * sample_weight, dim=0)

        ranking_loss = args.rank_weight * ranking_loss
        loss = cls_loss + ranking_loss

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss and accuracy
        total_losses.update(loss.item(), labels.size(0))
        cls_losses.update(cls_loss.item(), labels.size(0))
        ranking_losses.update(ranking_loss.item(), labels.size(0))
        top1.update(prec.item(), labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Rank Loss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'
                  'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=total_losses, cls_loss=cls_losses,
                rank_loss=ranking_losses, top1=top1))

        # correctness count update
        history.correctness_update(idx, correct, output)

    # max correctness update
    history.max_correctness_update(epoch)

    logger_w.write([epoch, total_losses.avg, cls_losses.avg, ranking_losses.avg, top1.avg])

    metrics_val = None
    metrics_test = None
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint! validation...", epoch)
        best_thr, metrics_val = evaluate(epoch, valid_loader, model, return_best_thr=True, args=args)
        logger.info('eval on test data!...')
        _, metrics_test = evaluate(epoch, test_loader, model, thr=best_thr, args=args)

    return metrics_val, metrics_test


def main():
    # set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # check save path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = VenueCNNMatchDataset(args.file_dir, args.matrix_size1, args.matrix_size2, args.seed, args.shuffle)
    N = len(dataset)
    train_start, valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(valid_start - train_start, 0))
    valid_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(test_start - valid_start, valid_start))
    test_loader = DataLoader(dataset, batch_size=args.batch,
                             sampler=ChunkSampler(N - test_start, test_start))

    model = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                          mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                          mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                          mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                          mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)

    # set criterion
    # cls_criterion = nn.CrossEntropyLoss().cuda()
    cls_criterion = nn.NLLLoss(reduction="none")
    ranking_criterion = nn.MarginRankingLoss(margin=0.0)

    if args.cuda:
        model = model.cuda()
        cls_criterion = cls_criterion.cuda()
        ranking_criterion = ranking_criterion.cuda()

    optimizer = optim.Adagrad(model.parameters(), lr=args.lr,
                              # initial_accumulator_value=args.initial_accumulator_value,
                              weight_decay=args.weight_decay)

    # make logger
    train_logger = utils_orig.Logger(os.path.join(save_path, 'train.log'))
    result_logger = utils_orig.Logger(os.path.join(save_path, 'result.log'))

    correctness_history = crl_utils.History(len(train_loader.dataset), args)

    loss_val_min = None
    best_test_metric = None
    best_model = None

    # start Train
    for epoch in range(1, args.epochs + 1):
        metrics = train(train_loader,
                        valid_loader,
                        test_loader,
                        model,
                        cls_criterion,
                        ranking_criterion,
                        optimizer,
                        epoch,
                        correctness_history,
                        train_logger,
                        args)

        metrics_val, metrics_test = metrics
        if metrics_val is not None:
            # result write
            result_logger.write(
                ["valid", metrics_val[0], metrics_val[2]*100, metrics_val[3]*100, metrics_val[4]*100,
                 "test", metrics_test[0], metrics_test[2]*100, metrics_test[3]*100, metrics_test[4]*100])
            if loss_val_min is None:
                loss_val_min = metrics_val[0]
                best_test_metric = metrics_test
                best_model = model
            elif metrics_val[0] < loss_val_min:
                loss_val_min = metrics_val[0]
                best_test_metric = metrics_test
                best_model = model

        # save model
        if epoch == args.epochs:
            torch.save(best_model.state_dict(),
                       os.path.join(save_path, 'model.pth'))

    print("min_val_loss", loss_val_min, "best test metrics", best_test_metric[2:])


if __name__ == "__main__":
    main()
    print("done")
