import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import resnet34

from efficientnet_pytorch import EfficientNet

from dataset import cifar_loader
from vit import VisionTransformer
from utils import AverageMeter, calculate_accuracy, make_dirs, get_lr_scheduler, plot_metrics
from utils import init_weights_normal, init_weights_xavier, init_weights_kaiming

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(data_loader, model, optimizer, criterion, epoch, config):

    # Average Meter #
    top_loss = AverageMeter()
    top1_accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    # Switch to Train Mode #
    model.train()

    # Train #
    for i, (image, label) in enumerate(data_loader):

        # Prepare Data #
        image, label = image.to(device), label.to(device)

        # Initialize Optimizer #
        optimizer.zero_grad()

        # Forward Data and Calculate Loss#
        pred = model(image)
        loss = criterion(pred, label)

        # Back Propagation and Update #
        loss.backward()
        optimizer.step()

        # Record Data #
        top1_pred, top5_pred = calculate_accuracy(pred.data, label.data, topk=(1, 5))
        top_loss.update(loss.item(), image.size(0))
        top1_accuracy.update(top1_pred.item(), image.size(0))
        top5_accuracy.update(top5_pred.item(), image.size(0))

        # Print Statistics #
        if (i+1) % config.print_every == 0:
            print("Train | Epoch [{}/{}] | Iterations [{}/{}] | Loss {:.4f} | Top 1 Accuracy {:.2f}% | Top 5 Accuracy {:.2f}%"
                  .format(epoch+1, config.num_epochs, i+1, len(data_loader), top_loss.avg, top1_accuracy.avg, top5_accuracy.avg))

    return top_loss.avg, top1_accuracy.avg, top5_accuracy.avg


def validate(data_loader, model, criterion, epoch, config):

    # Average Meter #
    top_loss = AverageMeter()
    top1_accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    # Switch to Evaluation Mode #
    model.eval()

    for i, (image, label) in enumerate(data_loader):

        # Prepare Data #
        image, label = image.to(device), label.to(device)

        # For Inference #
        with torch.no_grad():
            pred = model(image)
            loss = criterion(pred, label)

        # Record Data #
        top1_pred, top5_pred = calculate_accuracy(pred.data, label.data, topk=(1, 5))
        top_loss.update(loss.item(), image.size(0))
        top1_accuracy.update(top1_pred.item(), image.size(0))
        top5_accuracy.update(top5_pred.item(), image.size(0))

        # Print Statistics #
        if (i+1) % config.print_every == 0:
            print("  Val | Epoch [{}/{}] | Iterations [{}/{}] | Loss {:.4f} | Top 1 Accuracy {:.2f}% | Top 5 Accuracy {:.2f}%"
                  .format(epoch+1, config.num_epochs, i+1, len(data_loader), top_loss.avg, top1_accuracy.avg, top5_accuracy.avg))

    return top_loss.avg, top1_accuracy.avg, top5_accuracy.avg


def test(data_loader, model, config):

    # Average Meter #
    top1_accuracy = AverageMeter()
    top5_accuracy = AverageMeter()

    # Load Pre-trained Model Weight #
    checkpoint = torch.load(os.path.join(config.weights_path, 'BEST_{}_{}_{}.pkl'.format(model.__class__.__name__, str(config.dataset).upper(), config.num_classes)))
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # Switch to Evaluation Mode #
    model.eval()

    for i, (image, label) in enumerate(data_loader):

        # Prepare Data #
        image, label = image.to(device), label.to(device)

        # For Inference #
        with torch.no_grad():
            pred = model(image)

            # Record Data #
            top1_pred, top5_pred = calculate_accuracy(pred.data, label.data, topk=(1, 5))
            top1_accuracy.update(top1_pred.item(), image.size(0))
            top5_accuracy.update(top5_pred.item(), image.size(0))

    # Print Statistics #
    print("Test | {} | Top 1 Accuracy {:.2f}% | Top 5 Accuracy {:.2f}%".format(model.__class__.__name__, top1_accuracy.avg, top5_accuracy.avg))


def main(config):

    # For Reproducibility #
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Weights and Plots Path #
    paths = [config.weights_path, config.plots_path]
    for path in paths:
        make_dirs(path)

    # Prepare Data Loader #
    if config.dataset == 'cifar':
        train_loader, val_loader, test_loader = cifar_loader(config.num_classes, config.batch_size)
        input_size = 32

    # Prepare Networks #
    if config.model == 'vit':
        model = VisionTransformer(
            in_channels=config.in_channels, 
            embed_dim=config.embed_dim, 
            patch_size=config.patch_size, 
            num_layers=config.num_layers, 
            num_heads=config.num_heads, mlp_dim=config.mlp_dim, 
            dropout=config.drop_out, 
            input_size=input_size, 
            num_classes=config.num_classes).to(device)

    elif config.model == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=config.num_classes).to(device)

    elif config.model == 'resnet':
        model = resnet34(pretrained=False).to(device)
        model.fc = nn.Linear(config.mlp_dim, config.num_classes).to(device)

    else:
        raise NotImplementedError
    
    # Weight Initialization #
    if not config.model == 'efficient':
        if config.init == 'normal':
            model.apply(init_weights_normal)
        elif config.init == 'xavier':
            model.apply(init_weights_xavier)
        elif config.init == 'he':
            model.apply(init_weights_kaiming)
        else:
            raise NotImplementedError
    
    # Train #
    if config.phase == 'train':

        # Loss Function #
        criterion = nn.CrossEntropyLoss()

        # Optimizers #
        if config.num_classes == 10:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.5, 0.999))
            optimizer_scheduler = get_lr_scheduler(config.lr_scheduler, optimizer)
        elif config.num_classes == 100:
            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
            optimizer_scheduler = get_lr_scheduler('step', optimizer)

        # Constants #
        best_top1_acc = 0

        # Lists #
        train_losses, val_losses = list(), list()
        train_top1_accs, train_top5_accs = list(), list()
        val_top1_accs, val_top5_accs = list(), list()

        # Train and Validation #
        print("Training {} has started.".format(model.__class__.__name__))
        for epoch in range(config.num_epochs):

            # Train #
            train_loss, train_top1_acc, train_top5_acc = train(train_loader, model, optimizer, criterion, epoch, config)

            # Validation #
            val_loss, val_top1_acc, val_top5_acc = validate(val_loader, model, criterion, epoch, config)

            # Add items to Lists #
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_top1_accs.append(train_top1_acc)
            train_top5_accs.append(train_top5_acc)

            val_top1_accs.append(val_top1_acc)
            val_top5_accs.append(val_top5_acc)

            # If Best Top 1 Accuracy #
            if val_top1_acc > best_top1_acc:
                best_top1_acc = max(val_top1_acc, best_top1_acc)

                # Save Models #
                print("The best model is saved!")
                torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_{}_{}.pkl'.format(model.__class__.__name__, str(config.dataset).upper(), config.num_classes)))

            print("Best Top 1 Accuracy {:.2f}%\n".format(best_top1_acc))

            # Optimizer Scheduler #
            optimizer_scheduler.step()

        # Plot Losses and Accuracies #
        losses = (train_losses, val_losses)
        accs = (train_top1_accs, train_top5_accs, val_top1_accs, val_top5_accs)
        plot_metrics(losses, accs, config.plots_path, model, config.dataset, config.num_classes)

        print("Training {} using {} {} finished.".format(model.__class__.__name__, str(config.dataset).upper(), config.num_classes))

    # Test #
    elif config.phase == 'test':

        test(test_loader, model, config)

    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    parser.add_argument('--dataset', type=str, default='cifar', help='which dataset to train')
    parser.add_argument('--num_classes', type=int, default=10, help='num_classes for cifar dataset', choices=[10, 100])
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')

    parser.add_argument('--model', type=str, default='vit', help='which model to train for benchmarking', choices=['vit', 'efficient', 'resnet'])
    parser.add_argument('--init', type=str, default='he', help='which initialization technique to apply', choices=['normal', 'xavier', 'he'])

    parser.add_argument('--in_channels', type=int, default=3, help='in channels')
    parser.add_argument('--embed_dim', type=int, default=512, help='embed dim')
    parser.add_argument('--patch_size', type=int, default=4, help='patch size')
    parser.add_argument('--num_layers', type=int, default=4, help='the number of layers')
    parser.add_argument('--num_heads', type=int, default=6, help='the number of heads')
    parser.add_argument('--mlp_dim', type=int, default=512, help='mlp dimension')
    parser.add_argument('--drop_out', type=float, default=0.1, help='probability for drop out')

    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')

    parser.add_argument('--num_epochs', type=int, default=40, help='total epoch')
    parser.add_argument('--print_every', type=int, default=100, help='print statistics for every n iteration')

    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)