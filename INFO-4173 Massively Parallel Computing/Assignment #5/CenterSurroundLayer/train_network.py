#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, List
import os
import numpy  as np
import matplotlib.pyplot as plt
import argparse
import urllib
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from tqdm import tqdm
from networks import (FullyConnectedNetwork,
                      TraditionalConvolutionalNetwork,
                      CenterSurroundConvolutionalNetwork)


def get_mnist_dataloaders(batch_size: int) -> Tuple[DataLoader]:
    """ Generates the dataflow for the MNIST dataset.
    Args:
        batch_size:
            Size of the mini-batches.
    Returns:
        Train- and TestDataloader for MNIST which generate randomized examples
        of shape [batch_size, 1, 28, 28].
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    root_path = "/tmp/datasets"

    train_set = torchvision.datasets.MNIST(
        root=root_path,
        train=True,
        download=True,
        transform=transform)
    test_set = torchvision.datasets.MNIST(
        root=root_path,
        train=False,
        download=True,
        transform=transform)

    proc_count = multiprocessing.cpu_count()
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=proc_count,
        shuffle=True,
        pin_memory=True)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True)
    return train_loader, test_loader


def matplotlib_imshow(img: torch.Tensor, one_channel: bool = False) -> None:
    """ Show the given image into the active matplotlib figure.
    Args:
        img:
            The image as pytorch tensor of shape [channels, height, width].
        one_channel:
            If true the img is reduced to one channel by computing the mean
            over the cannel dimension.
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net: torch.nn.Module,
                    images: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images.
    Args:
        net:
            A pytorch.nn.Module that acts on the images.
        images:
            The input images for the net.
    Returns:
        Tuple of predictions and predicted probabilities for the prediction.
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item()
                   for i, el in zip(preds, output)]


def plot_classes_preds(net: torch.nn.Module, images: torch.Tensor,
                       labels: torch.Tensor) -> plt.Figure:
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    Args:
        net:
            The model to make predictions.
        images:
            The input images.
        labels:
            The ground truth labels.
    Returns:
        The filled matplotlib figure.
    """
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 2))
    for idx in np.arange(8):
        ax = fig.add_subplot(1, 8, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig

def test(model, test_loader, device):
    # tell the model that it is evaluating (important for Dropout)
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        progress = tqdm(test_loader,
                        desc="Testing",
                        ascii=True)
        for input_images, labels in progress:
            # copy input to gpu
            input_images, labels = input_images.to(device), labels.to(device)
            # run model
            output = model(input_images)
            test_loss += torch.nn.functional.nll_loss(output, labels,
                                                      reduction="sum").item()
            # get index of the maximum log probability
            pred = output.argmax(dim=1, keepdim=True)
            # compare with label and add up number of correct predictions
            correct += pred.eq(labels.view_as(pred)).sum().item()
    # compute average loss
    example_count = len(test_loader.dataset)
    test_loss /= example_count
    print("Evaluation on Test-Set:")
    print("Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
        test_loss, correct, example_count,
        100.0 * correct / example_count))
    return test_loss


def train(model: torch.nn.Module, train_loader: DataLoader,
          test_loader: DataLoader, save_path: str, epochs: int = 10,
          profile: bool = False):
    """Train the given model on the given data and evaluate the test set after
    every epoch.
    Args:
        model:
            The nn model to train.
        train_loader:
            Data loader with training data.
        test_loader:
            Data loader with test data.
        save_path:
            Path where to save the trained model and the training log.
        epochs:
            Number of training epochs.
        profile:
            If true train only for one step (usefull for profiling).
    """
    # create folder for saving
    os.makedirs(save_path, exist_ok=True)
    # Writer for tensorboard log
    writer = SummaryWriter(os.path.join(save_path, "log"))
    # We train on GPU
    device = torch.device("cuda")
    model.to(device)
    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # show model graph in tensorboard
    dataiter = iter(train_loader)
    images, label = next(dataiter)
    writer.add_graph(model, images.to(device))

    # print number of trainable variables in model
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Info: model has", param_count, "trainable parameters.")
    print("-" * 80)

    running_loss = 0.0
    step = 0
    for epoch in range(epochs):
        print("Starting Epoch", epoch + 1)
        progress = tqdm(train_loader,
                        ascii=True,
                        desc="Training")
        # Tell the model it is training (important for dropout)
        model.train()
        for input_images, labels in progress:
            # copy input to gpu
            input_images, labels = input_images.to(device), labels.to(device)
            # reset gradients
            optimizer.zero_grad()
            # evaluate model
            output = model(input_images)
            # Compute a loss (negative log likelihood)
            loss = torch.nn.functional.nll_loss(output, labels)
            # run back propagation
            loss.backward()
            # Update weights
            optimizer.step()

            # Show some gradient statistics - usefull for debugging
            for name, param in model.named_parameters():
                name = name.replace(".", "/")
                abs_grad = param.grad.abs().mean()
                abs_range = (param.min(), param.max())
                writer.add_scalar(name + "/grad-abs-mean",
                                  abs_grad,
                                  step)
                writer.add_scalar(name + "/min",
                                  abs_range[0],
                                  step)
                writer.add_scalar(name + "/max",
                                  abs_range[1],
                                  step)
            step += 1

            # update running loss for tensorboard
            running_loss += loss.item()

            # update progress bar
            progress.set_postfix({"loss": loss.item()})
            if profile:
                print("Stopping to reduce profiling time")
                return

        path = os.path.join(save_path,
                            "mnist_model_epoch{0:04d}".format(epoch))
        print("Saving Model to", path)
        torch.save(model.state_dict(), path)

        # create a loss plot
        writer.add_scalar("training loss",
                          running_loss / len(train_loader),
                          (1+epoch) * len(train_loader))
        running_loss = 0.0
        # add prediction images to tensorboard
        writer.add_figure("Prediction vs. Actuals",
                          plot_classes_preds(model, input_images, labels),
                          global_step=(1+epoch) * len(train_loader))

        print("Evaluating model on test data")
        test_loss = test(model, test_loader, device)
        writer.add_scalar("test loss",
                          test_loss,
                          (1+epoch) * len(train_loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Index of GPU to use for training")
    parser.add_argument("--model", type=str, default="csc",
                        help="Model to use. One of ['fc', 'conv', 'csc'] "
                        "for 'Linear Fully Connected', "
                        "'Traditional 2d Convolution' and 'Center Surround "
                        "Convolution'.")
    parser.add_argument("--save", type=str, default="/tmp/mnist",
                        help="Path where to save the trained model.")
    parser.add_argument("--profile", action="store_true",
                        help="Run program in profiling mode. "
                        "(viewer iterations)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    train_loader, test_loader = get_mnist_dataloaders(args.batch)
    if args.model == "fc":
        print("I am using a linear fully connected model")
        model = FullyConnectedNetwork()
        save_path = os.path.join(args.save, "fully_connected")
    elif args.model == "conv":
        print("I am using traditional 2d convolutions")
        model = TraditionalConvolutionalNetwork()
        save_path = os.path.join(args.save, "traditional")
    elif args.model == "csc":
        print("I am using center surround convolutions")
        model = CenterSurroundConvolutionalNetwork()
        save_path = os.path.join(args.save, "center_surround")
    else:
        raise ValueError("Argument model must be one of "
                         "['fc', 'conv', 'csc'].")
    train(model, train_loader, test_loader, save_path, epochs=args.epochs,
          profile=args.profile)


if __name__ == '__main__':
    main()


