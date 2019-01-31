"""
Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""


from __future__ import print_function
from __future__ import division

import time
import os
import copy
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.model import initialize_model



######################################################################
# Inputs

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "../data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "inception"

# Number of classes in the dataset
num_classes = 101

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 200

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# use GPU ID please check `nvidia-smi` command
gpu_id = 0

# output dir log, graph etc
output_dir = "../outputs/{}_{}_{}".format(model_name, batch_size, num_epochs)
os.mkdir(output_dir)

# root rogger setting
logger = getLogger()
formatter = Formatter('%(asctime)s %(name)s %(funcName)s [%(levelname)s]: %(message)s')
handler = StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
file_handler = FileHandler(filename=output_dir+"/train.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(DEBUG)

# log torch versions
logger.info("PyTorch Version: {}".format(torch.__version__))
logger.info("Torchvision Version: {}".format(torchvision.__version__))

# Detect if we have a GPU available
device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def main():
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    logger.info(model_ft)

    ######################################################################
    # Load Data
    # ---------
    #
    # Now that we know what the input size must be, we can initialize the data
    # transforms, image datasets, and the dataloaders. Notice, the models were
    # pretrained with the hard-coded normalization values, as described
    # `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
    #

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    logger.info("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'test']}


    ######################################################################
    # Create the Optimizer
    # --------------------
    #
    # Now that the model structure is correct, the final step for finetuning
    # and feature extracting is to create an optimizer that only updates the
    # desired parameters. Recall that after loading the pretrained model, but
    # before reshaping, if ``feature_extract=True`` we manually set all of the
    # parameter’s ``.requires_grad`` attributes to False. Then the
    # reinitialized layer’s parameters have ``.requires_grad=True`` by
    # default. So now we know that *all parameters that have
    # .requires_grad=True should be optimized.* Next, we make a list of such
    # parameters and input this list to the SGD algorithm constructor.
    #
    # To verify this, check out the printed parameters to learn. When
    # finetuning, this list should be long and include all of the model
    # parameters. However, when feature extracting this list should be short
    # and only include the weights and biases of the reshaped layers.
    #

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    logger.info("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                logger.info("\t {}".format(name))
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                logger.info("\t {}".format(name))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.
    #

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))

    ######################################################################
    # Comparison with Model Trained from Scratch
    # ------------------------------------------
    #
    # Just for fun, lets see how the model learns if we do not use transfer
    # learning. The performance of finetuning vs. feature extracting depends
    # largely on the dataset but in general both transfer learning methods
    # produce favorable results in terms of training time and overall accuracy
    # versus a model trained from scratch.
    #

    # Initialize the non-pretrained version of the model used for this run
    scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _, scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer,
                                  num_epochs=num_epochs, is_inception=(model_name == "inception"))

    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    ohist = []
    shist = []

    ohist = [h.cpu().numpy() for h in hist]
    shist = [h.cpu().numpy() for h in scratch_hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    plt.plot(range(1, num_epochs + 1), shist, label="Scratch")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(output_dir + "/figure.png")
    # plt.show()

if __name__ == '__main__':
    main()
