# Rotation-Invariant Coordinate Convolutional Neural Network

Hanlin Mo and Guoying Zhao. "RIC-CNN: Rotation-Invariant Coordinate Convolutional Neural Network".

Instruction

In this paper, we propose a Rotation-Invariant Coordinate Convolution (RIC-C), which achieves natural invariance to arbitrary rotations around the input center without additional trainable parameters or data augmentation. We evaluate the rotational invariance of RIC-C using the MNIST dataset and compare its performance with previous rotation-invariant CNN models. RIC-C achieves state-of-the-art classification on the MNIST-rot test set without data augmentation and with lower computational costs. We further demonstrate the interchangeability of RIC-C with traditional convolution operations by integrating it into common CNN models such as VGG, ResNet, and DenseNet. We conduct image classification and matching experiments on the NWPU VHR-10, MTARSI, AID, and UBC benchmark datasets, showing that RIC-C significantly enhances the performance of CNN models across different applications.

The papper can be downloaded from: https://arxiv.org/abs/2211.11812

Usage

The code is tested under Pytorch 2.0.1, Python 3.10, and CUDA12.2 on Ubuntu16.04. 

Questions

If you have any questions, please do not hesitate to contact hanlin.mo@oulu.fi.   