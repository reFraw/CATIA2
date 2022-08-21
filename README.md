# CATIA2

<p align="center" width="100%">
   <img width="80%" src="https://raw.githubusercontent.com/reFraw/CATIA2/main/images/CATIA2.png">
</p>

### Description
This tool allows you to use convolutional neural networks for image analysis in a guided way and therefore suitable for beginners (like me).
Actually this project was born as a training tool for my programming skills.

### Dataset structure
To add a personal dataset, remember that it must have two folders, called 'training' and 'test', where in each one there are subfolders related to the classes.

### Usage
CATIA2 has two modes:
1) Normal mode >>> Enter 'python CATIA2.py'
2) One-line mode >>> Enter 'python CATIA2.py one-line'

To see the parameters to be entered in the one-line mode use 'python CATIA2.py one-line -h'.

Starting the program for the first time allows the creation of folders:
1) <</DATASET>>: The datasets to be analyzed must be entered in this folder.
2) <</models_save>>: The trained models are saved in this folder.
3) <</results>>: The reports of the training and testing phases are saved in this folder, together with the plots of the main metrics.

### Notes
1) The inspiration for the writing of this project comes from the TAMI tool (https://github.com/Djack1010/tami) used during my internship and thesis work.
