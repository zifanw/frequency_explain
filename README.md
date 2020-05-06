# Frequency_explain
10708 Group Project 2020 Spring

To get access of the robust data, please use this link: https://drive.google.com/drive/folders/1QTLckP-G7bTR8e31xqW2rZYV7iPsF7H7?usp=sharing

To get the simple black box attacked images, please refer to: https://drive.google.com/drive/folders/1teGoSbrjen4tHlOpvx3rBNfRuNUxxx-1?usp=sharing



### Architecture

| Architecture \ Task | Attribution                                  | Robust Dataset | Adversarial Attack |
| ------------------- | -------------------------------------------- | -------------- | ------------------ |
| Model One           | ResNet (Madry pertained)                   |  cifar_test_non_robust              |   NA                 |
| Model Two           | ResNet ($\delta_2=0.25$, Madry pertained )   |  cifar_test_robust              |       PGD             |
| Model Three         | ResNet ($\delta_2=0.5$, Madry pertained )    |                |          PGD          |
| Model Four          | ResNet ($\delta_\infin=8$, Madry pertained ) |                |          PGD          |


The simple-black-box-attack is copied from https://github.com/cg563/simple-blackbox-attack, with some modifications to adapt the CIFAR-10 dataset.
To run the simple-black-box-attack, some additional files are required, which can be download from: https://drive.google.com/file/d/1XdjmSu7jzcez1p7M_nl6jaCrYMNNDJW8/view?usp=sharing. Simply unzip the downloaded .tar file before running.

Robustness package is need to conduct the simple-black-box-attack, which can be installed by:

`pip install robustness`

The command to run the attack on cifar10 is:

`python run_simba.py --data_root cifar10 --num_runs 10000 --num_iters 10000 --pixel_attack  --freq_dims 32`

Change --num_runs to attack different number of images.
