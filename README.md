# VEN
Official implementation of "Interactive Hepatic Vessel Segmentation Framework via Multi-Task Learning and Vessel Geometric Prior Compression based on Run-Length Encoding".

Train base on nnUnetv2:nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation

This repository provides a lightweight GUI implementation of VEN, along with the pre-trained weights of the VEN-Lite model trained on Re-MSD8. For details, please refer to the paper.

We provide one example from Re-MSD8 that was not used during training. See in case/*

Run gui.py to start.

See 3D Mask RLE in sp_blocks.py

# Overview

Auto seg -> Edit wrong slice -> VEN refines whole seg

<img width="1080" height="586" alt="overview_low" src="https://github.com/user-attachments/assets/579c84ab-8fd8-4bc5-a715-3b65853e5a73" />



