
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import glob, os, re
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import random
from tqdm import tqdm
import torch.nn.functional as F

import  torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold, KFold
from esemble_model import MyEffientnet_b0, MyResnet50, MyResnet50_esemble, MyEffientnet_b0_esemble, MyEnsemble
from utils import change_contrast, ImageDataset, read_data, read_files

def model_train(skfold, model_name, trainloader, valloader, class_weight, args):

    scheduler = None

    if skfold == 0:
        '''고정!'''
        my_model = MyResnet50()
        my_model.cuda()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-5, weight_decay=1e-4)
        num_epoch = 10
    elif skfold == 1:
        '''고정! '''
        my_model = MyEffientnet_b0()
        my_model.cuda()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epoch = 10
    elif skfold == 2:
        '''고정! '''
        my_model = MyResnet50()
        my_model.cuda()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=7e-6, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 8], gamma=0.9)
        num_epoch = 5
    elif skfold == 3:
        '''고정!!!'''
        my_model = MyEffientnet_b0()
        my_model.cuda()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        num_epoch = 15
    else :
        '''고정!!!'''
        my_model = MyResnet50()
        my_model.cuda()
        optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        num_epoch = 5

    if scheduler is not None:
        print('scheduler will be applied to training')

    class_weight = class_weight.cuda()
    criterion = nn.CrossEntropyLoss(class_weight)


    val_auc_check = np.array([])
    for epoch in range(num_epoch):
        epoch_loss_train = 0.0
        epoch_train_acc = 0.0
        predicted_train_output = np.array([])
        train_real = np.array([])
        train_probability = np.array([]).reshape(0, 2)


        my_model.train()
        for enu, (train_x_batch, train_y_batch) in enumerate(tqdm(trainloader)):
            train_x = Variable(train_x_batch).cuda()
            train_y = Variable(train_y_batch).cuda()

            optimizer.zero_grad()

            train_output = my_model(train_x)
            train_epoch_loss = criterion(train_output, torch.max(train_y, 1)[1])

            train_epoch_loss.backward()
            optimizer.step()

            epoch_loss_train += (train_epoch_loss.data.item() * len(train_x_batch))

            train_pred = np.argmax(train_output.detach().data.cpu().numpy(), axis = 1)
            train_true = np.argmax(train_y.detach().data.cpu().numpy(), axis = 1)
            predicted_train_output = np.append(predicted_train_output, train_pred)
            train_real = np.append(train_real, train_true)
            train_probability = np.append(train_probability, train_output.detach().data.cpu().numpy(), axis = 0)

        del train_x_batch, train_y_batch, train_output
        torch.cuda.empty_cache()

        train_loss = epoch_loss_train / (len(trainloader)/16)
        train_acc = len(np.where(predicted_train_output == train_real)[0]) / len(predicted_train_output)
        train_auc_score = roc_auc_score(train_real, train_probability[:, 1])

        with torch.no_grad():
            epoch_loss_val = 0.0
            epoch_acc_val = 0.0
            predicted_val_output = np.array([])
            val_real = np.array([])
            val_probability = np.array([]).reshape(0, 2)

            my_model.eval()

            for enu, (validation_x_batch, validation_y_batch) in enumerate(tqdm(valloader)):
                validation_x = Variable(validation_x_batch).cuda()
                validation_y = Variable(validation_y_batch).cuda()

                validation_output = my_model(validation_x)
                validation_epoch_loss = criterion(validation_output, torch.max(validation_y, 1)[1])

                epoch_loss_val += (validation_epoch_loss.data.item() * len(validation_x_batch))

                pred_val = np.argmax(validation_output.data.cpu().numpy(), axis = 1)
                true_val = np.argmax(validation_y.data.cpu().numpy(), axis = 1)
                predicted_val_output = np.append(predicted_val_output, pred_val)
                val_real = np.append(val_real, true_val)
                val_probability = np.append(val_probability, validation_output.detach().data.cpu().numpy(), axis = 0)

            del validation_x_batch, validation_y_batch, validation_output
            torch.cuda.empty_cache()

            val_loss = epoch_loss_val / (len(valloader)/16)
            val_acc = len(np.where(predicted_val_output == val_real)[0]) / len(predicted_val_output)
            val_auc_score = roc_auc_score(val_real, val_probability[:, 1])
            val_auc_check = np.append(val_auc_check, val_auc_score)

        if val_auc_check[epoch] == val_auc_check.max():
            print('model saving ' + model_name + '--------- epoch : {}, validation_auc : {:.6f}'.format(epoch, val_auc_check[epoch]))
            torch.save(my_model.state_dict(), os.path.join(args.model, str(skfold) +'_checkpoint.pth'))
            for number in range(10) :
                je_path = args.model + '/esemble_' + str(number)
                if not os.path.exists(je_path):
                    os.makedirs(je_path)
                print('safety weights are saving')
                torch.save(my_model.state_dict(), os.path.join(je_path, str(skfold) +'_checkpoint.pth'))


        if scheduler is not None:
            scheduler.step()

        print('Epoch : [{}]/[{}]   \t'
              'train auc : {:.4f}\t'
             'train acc : {:.4f}\t'
             'train loss : {:.4f}\t'
              'val auc : {:.4f}\t'
             'val acc : {:.4f}\t'
             'val loss : {:.4f}\t'.format(epoch, num_epoch, train_auc_score, train_acc, train_loss,
                                          val_auc_score, val_acc, val_loss))



def ensemble_training(model, trainloader, valloader, num_epoch, class_weight, args):
    my_model = model
    my_model.cuda()

    class_weight = class_weight.cuda()
    criterion = nn.CrossEntropyLoss(class_weight)
    optimizer = torch.optim.Adam(my_model.classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    #1e-7 (x)

    val_auc_check = np.array([])

    for epoch in range(num_epoch):
        epoch_loss_train = 0.0
        epoch_train_acc = 0.0
        predicted_train_output = np.array([])

        train_real = np.array([])
        train_probability = np.array([]).reshape(0, 2)

        my_model.train()
        for enu, (train_x_batch, train_y_batch) in enumerate(tqdm(trainloader)):
            train_x = Variable(train_x_batch).cuda()
            train_y = Variable(train_y_batch).cuda()

            optimizer.zero_grad()

            train_output = my_model(train_x)
            train_epoch_loss = criterion(train_output, torch.max(train_y, 1)[1])

            train_epoch_loss.backward()
            optimizer.step()

            epoch_loss_train += (train_epoch_loss.data.item() * len(train_x_batch))

            train_pred = np.argmax(train_output.detach().data.cpu().numpy(), axis = 1)
            train_true = np.argmax(train_y.detach().data.cpu().numpy(), axis = 1)
            predicted_train_output = np.append(predicted_train_output, train_pred)
            train_real = np.append(train_real, train_true)
            train_probability = np.append(train_probability, train_output.detach().data.cpu().numpy(), axis = 0)

        del train_x_batch, train_y_batch, train_output
        torch.cuda.empty_cache()

        train_loss = epoch_loss_train / (len(trainloader)/16)
        train_acc = len(np.where(predicted_train_output == train_real)[0]) / len(predicted_train_output)
        train_auc_score = roc_auc_score(train_real, train_probability[:, 1])

        with torch.no_grad():
            epoch_loss_val = 0.0
            epoch_acc_val = 0.0
            predicted_val_output = np.array([])
            val_real = np.array([])
            val_probability = np.array([]).reshape(0, 2)

            my_model.eval()

            for enu, (validation_x_batch, validation_y_batch) in enumerate(tqdm(valloader)):
                validation_x = Variable(validation_x_batch).cuda()
                validation_y = Variable(validation_y_batch).cuda()

                validation_output = my_model(validation_x)
                validation_epoch_loss = criterion(validation_output, torch.max(validation_y, 1)[1])

                epoch_loss_val += (validation_epoch_loss.data.item() * len(validation_x_batch))

                pred_val = np.argmax(validation_output.data.cpu().numpy(), axis = 1)
                true_val = np.argmax(validation_y.data.cpu().numpy(), axis = 1)
                predicted_val_output = np.append(predicted_val_output, pred_val)
                val_real = np.append(val_real, true_val)
                val_probability = np.append(val_probability, validation_output.detach().data.cpu().numpy(), axis = 0)


            del validation_x_batch, validation_y_batch, validation_output
            torch.cuda.empty_cache()

            val_loss = epoch_loss_val / (len(valloader)/16)
            val_acc = len(np.where(predicted_val_output == val_real)[0]) / len(predicted_val_output)
            val_auc_score = roc_auc_score(val_real, val_probability[:, 1])
            val_auc_check = np.append(val_auc_check, val_auc_score)

        if val_auc_check[epoch] == val_auc_check.max():
            print('model saving "esemble"--------- epoch : {}, validation_auc : {:.6f}'.format(epoch, val_auc_check[epoch]))
            torch.save(my_model.state_dict(), os.path.join(args.model, str(epoch) + '_esemble_checkpoint.pth'))



        print('Epoch : [{}]/[{}]   \t'
              'train auc : {:.4f}\t'
             'train acc : {:.4f}\t'
             'train loss : {:.4f}\t'
              'val auc : {:.4f}\t'
             'val acc : {:.4f}\t'
             'val loss : {:.4f}\t'.format(epoch, num_epoch, train_auc_score, train_acc, train_loss,
                                          val_auc_score, val_acc, val_loss))


def main():

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random_seed = 1024

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data",
        default='/DATA/data_cancer/train/',
        type=str,
        required=True,
    )

    # Other parameters
    parser.add_argument(
        "--model",
        default="/USER/USER_WORKSPACE/jechoi/code/weights",
        type=str,
        help="model is saving in this directory"
    )

    args = parser.parse_args()


    num_epoch = 1
    batch_size = 16



    train_transforms = {
        'train_base': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train_flip' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train_aff' : transforms.Compose([
            transforms.Resize(256),
            transforms.RandomAffine(degrees=(-70,70), scale=(1.2, 1.2)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    esemble_transforms = {
        'train_base': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomPerspective(0.7, p=0.7),
            transforms.RandomRotation(50),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    test_transforms={
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    img_names, labels = read_files(args.data)

    zero_ratio = len(np.where(labels == 0)[0]) / len(labels)
    one_ratio = len(np.where(labels == 1)[0]) / len(labels)
    cancer_weights = round(zero_ratio, 3)
    else_weights = 1 - cancer_weights

    class_weight = torch.FloatTensor([else_weights, cancer_weights])

    # labels_list = labels.tolist()

    zero = np.where(labels == 0)[0].tolist()
    one = np.where(labels == 1)[0].tolist()

    # zero = random.shuffle(zero)
    # one = random.shuffle(one)

    train_size = len(labels)*0.9

    train_zero_idx = random.sample(zero, int(train_size*zero_ratio))
    train_one_idx = random.sample(one, int(train_size*one_ratio))

    val_zero_idx = [i for i in zero if i not in train_zero_idx]
    val_one_idx = [i for i in one if i not in train_one_idx]

    train_idx = train_zero_idx + train_one_idx
    val_idx = val_zero_idx + val_one_idx

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    sk_labels = labels[train_idx]
    sk_img = img_names[train_idx]

    esemble_val = labels[val_idx]

    sk_labels_list = sk_labels.tolist()

    skf = StratifiedKFold(n_splits=5)


    for skfold, (sk_train_idx, sk_val_idx) in enumerate(skf.split(sk_labels_list, sk_labels_list)):
        if skfold == 0:
            aug_list = ['train_base']
        elif skfold == 1:
            aug_list = ['train_base']
        elif skfold == 2:
            aug_list = ['train_flip']
        elif skfold == 3:
            aug_list = ['train_flip']
        else:
             aug_list = ['train_aff']


        train_dataset = ImageDataset(args.data, sk_img[sk_train_idx], transform=aug_list[0])
        val_dataset = ImageDataset(args.data, sk_img[sk_val_idx], transform=test_transforms['val'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=2, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=True)
        if skfold == 0:
            model_name = 'resnet50'
        elif skfold == 1:
            model_name = 'efficient_b0'
        elif skfold == 2:
            model_name = 'resnet50'
        elif skfold == 3:
            model_name = 'efficient_b0'
        else:
            model_name = 'resnet50'

        if str(skfold) +'_checkpoint.pth' not in os.listdir(args.model) :
            print('\n')
            print('\n')
            print('{} fold : model name {} is now training'.format(skfold, model_name))

            model_train(skfold, model_name, train_loader, val_loader, class_weight, args)
        else:
            print('\n')
            print('{} fold : {} already trained'.format(skfold, model_name))

    print('\n')
    print('now esemble model is training')

    train_transformed_list = list()
    for aug in esemble_transforms.keys():
        train_transformed_list.append(ImageDataset(args.data, img_names[train_idx], esemble_transforms[aug]))

    train_dataset = torch.utils.data.ConcatDataset(train_transformed_list)
    val_dataset = ImageDataset(args.data, img_names[val_idx], transform=test_transforms['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, pin_memory=True)

    A_fname = os.path.join(args.model, '0_checkpoint.pth')
    B_fname = os.path.join(args.model, '1_checkpoint.pth')
    C_fname = os.path.join(args.model, '2_checkpoint.pth')
    D_fname = os.path.join(args.model, '3_checkpoint.pth')
    E_fname = os.path.join(args.model, '4_checkpoint.pth')


    model_A = MyResnet50_esemble()
    model_B = MyEffientnet_b0_esemble()
    model_C = MyResnet50_esemble()
    model_D = MyEffientnet_b0_esemble()
    model_E = MyResnet50_esemble()

    model_A.cuda()
    model_B.cuda()
    model_C.cuda()
    model_D.cuda()
    model_E.cuda()

    model_A.load_state_dict(torch.load(A_fname))
    model_B.load_state_dict(torch.load(B_fname))
    model_C.load_state_dict(torch.load(C_fname))
    model_D.load_state_dict(torch.load(D_fname))
    model_E.load_state_dict(torch.load(E_fname))


    for param in model_A.parameters():
        param.requires_grad_(False)

    for param in model_B.parameters():
        param.requires_grad_(False)

    for param in model_C.parameters():
        param.requires_grad_(False)

    for param in model_D.parameters():
        param.requires_grad_(False)

    for param in model_E.parameters():
        param.requires_grad_(False)

    esemble = MyEnsemble(model_A, model_B, model_C, model_D, model_E)
    ensemble_training(esemble, train_loader, val_loader, 10, class_weight, args)

    print('training is finished')

if __name__ == "__main__":
    main()
