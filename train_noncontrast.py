import os
from dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import torch
from ResNet_3D import ResNet18_3D_3stream_LSTM
from loss_function.CB_Loss import CB_loss
import argparse
import random
import shutil
from torch.optim.lr_scheduler import MultiStepLR
from utils import calculate_all_metrics, plot_best_roc_curves

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=4, help='batch size')
parser.add_argument('--epoch', type=int, default=200, help='all_epochs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()  


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

lr_max = 0.004894475924162084
L2 = 0.0002026834528438926
data_dir = "./data/preprocessed_train/"
metadata_path = "./relevant_files/matadata_S1.csv"    #S1 or S4
train_split_path = "./relevant_files/train.txt"
val_split_path = "./relevant_files/inval.txt"
num_class = 2

save_dir = './noncontrast_trained_models/trained_models_S1/bs{}_epoch{}_seed{}'.format(args.bs, args.epoch, args.seed)    #S1 or S4

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, 'roc'), exist_ok=True)

train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
print(save_dir)

print('dataset loading')

train_data = Dataset(data_dir, train_split_path, metadata_path, augment=True)
val_data = Dataset(data_dir, val_split_path, metadata_path, augment=False)

train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, drop_last=False)

print('train_lenth: %i  val_lenth: %i  num_0: %i  num_1: %i' % (
    train_data.len, val_data.len, train_data.num_0, train_data.num_1))


net = ResNet18_3D_3stream_LSTM(in_channels=1, n_classes=num_class, pretrained=False, no_cuda=False)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
net = torch.nn.DataParallel(net)
net = net.cuda()

optimizer = optim.SGD(net.parameters(), lr=lr_max, weight_decay=L2)
lr_scheduler = MultiStepLR(optimizer, milestones=[int((6 / 10) * args.epoch), int((9 / 10) * args.epoch)], gamma=0.1, last_epoch=-1)

best_ACC_val = 0
print('training')

for epoch in range(args.epoch):
    net.train()
    train_epoch_loss = []
    train_epoch_one_hot_label = []
    train_epoch_pred_scores = []
    train_epoch_class_label = []
    train_epoch_pred_class = []
    for i, (T1W1_imgs, T2W2_imgs, DWI_imgs, labels) in enumerate(train_dataloader):
        T1W1_imgs = T1W1_imgs.cuda().float()
        T2W2_imgs = T2W2_imgs.cuda().float()
        DWI_imgs = DWI_imgs.cuda().float()
        labels = labels.cuda().long()

        labels_one_hot = torch.zeros((labels.size(0), num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
        optimizer.zero_grad()
        outputs = net(T1W1_imgs, T2W2_imgs, DWI_imgs)
        loss = CB_loss(labels, outputs, samples_per_cls=[train_data.num_0, train_data.num_1],
                       no_of_classes=num_class, loss_type='focal', beta=0.999, gamma=2)
        loss.backward()
        optimizer.step()
        outputs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
        train_epoch_pred_scores.append(outputs.detach().cpu())
        train_epoch_one_hot_label.append(labels_one_hot)
        train_epoch_loss.append(loss.item())
        train_epoch_class_label.append(labels.cpu().numpy())
        train_epoch_pred_class.append(predicted.cpu().numpy())
        print('[%d/%d, %d/%d] train_loss: %.3f' %
              (epoch + 1, args.epoch, i + 1, len(train_dataloader), loss.item()))
    lr_scheduler.step()
    with torch.no_grad():
        net.eval()
        val_epoch_loss = []
        val_epoch_label = []
        val_epoch_pred_scores = []
        val_epoch_class_label = []
        val_epoch_pred_class = []
        for i, (T1W1_imgs, T2W2_imgs, DWI_imgs, labels) in enumerate(val_dataloader):
            T1W1_imgs = T1W1_imgs.cuda().float()
            T2W2_imgs = T2W2_imgs.cuda().float()
            DWI_imgs = DWI_imgs.cuda().float()
            labels = labels.cuda().long()
            labels_one_hot = torch.zeros((labels.size(0), num_class)).cuda().scatter_(1, labels.unsqueeze(1), 1).float().cpu()
            outputs = net(T1W1_imgs, T2W2_imgs, DWI_imgs)
            loss = CB_loss(labels, outputs,
                           samples_per_cls=[train_data.num_0, train_data.num_1],
                           no_of_classes=num_class, loss_type='softmax', beta=0.7931640416902306, gamma=2.1728597586331582)
            outputs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1, keepdim=False).detach()
            val_epoch_pred_scores.append(outputs.detach().cpu())
            val_epoch_label.append(labels_one_hot)
            val_epoch_loss.append(loss.item())
            val_epoch_class_label.append(labels.cpu().numpy())
            val_epoch_pred_class.append(predicted.cpu().numpy())

    train_epoch_one_hot_label = torch.cat(train_epoch_one_hot_label, dim=0).numpy().astype(np.uint8)
    train_epoch_pred_scores = torch.cat(train_epoch_pred_scores, dim=0).numpy()
    val_epoch_label = torch.cat(val_epoch_label, dim=0).numpy().astype(np.uint8)
    val_epoch_pred_scores = torch.cat(val_epoch_pred_scores, dim=0).numpy()

    train_epoch_class_label = np.concatenate(train_epoch_class_label)
    train_epoch_pred_class = np.concatenate(train_epoch_pred_class)
    val_epoch_class_label = np.concatenate(val_epoch_class_label)
    val_epoch_pred_class = np.concatenate(val_epoch_pred_class)

    train_metrics = calculate_all_metrics(train_epoch_class_label, train_epoch_pred_class, train_epoch_pred_scores)
    val_metrics = calculate_all_metrics(val_epoch_class_label, val_epoch_pred_class, val_epoch_pred_scores)

    train_metrics['Loss'] = np.mean(train_epoch_loss)
    val_metrics['Loss'] = np.mean(val_epoch_loss)

    print(
        '[%d/%d]  train_AUC: %.3f val_AUC: %.3f train_ACC: %.3f val_ACC: %.3f' %
        (epoch+1, args.epoch, train_metrics['auc_result']['auc'], val_metrics['auc_result']['auc'], train_metrics['ACC'], val_metrics['ACC']))

    if val_metrics['ACC'] > best_ACC_val:
        best_ACC_val = val_metrics['ACC']
        best_metrics_at_best_acc = {
            'epoch': epoch + 1,
            'train_metrics': train_metrics.copy(),
            'val_metrics': val_metrics.copy()
        }
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_ACC_val.pth'))
        
    if epoch + 1 == args.epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

    train_writer.add_scalar('loss', train_metrics['Loss'], epoch)
    train_writer.add_scalar('AUC', train_metrics['auc_result']['auc'], epoch)
    train_writer.add_scalar('ACC', train_metrics['ACC'], epoch)

    val_writer.add_scalar('loss', val_metrics['Loss'], epoch)
    val_writer.add_scalar('AUC', val_metrics['auc_result']['auc'], epoch)
    val_writer.add_scalar('ACC', val_metrics['ACC'], epoch)

train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)
print('best_ACC_val:', best_ACC_val)

print('\n' + '='*40)
print(f"最佳验证ACC: {best_ACC_val:.4f} (第{best_metrics_at_best_acc['epoch']}轮)")
print('='*40)
print("验证集指标:")
for metric, value in best_metrics_at_best_acc['val_metrics'].items():
    print(f"  {metric}: {value}")
print("\n训练集指标:")
for metric, value in best_metrics_at_best_acc['train_metrics'].items():
    print(f"  {metric}: {value}")
print('='*40)

plot_best_roc_curves(best_metrics_at_best_acc['train_metrics'], os.path.join(save_dir, 'roc/train_roc_curve.pdf'))
plot_best_roc_curves(best_metrics_at_best_acc['val_metrics'], os.path.join(save_dir, 'roc/val_roc_curve.pdf'))

