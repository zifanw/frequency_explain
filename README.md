# Frequency_explain
10708 Group Project 2020 Spring

To get access of the robust data, please use this link: https://drive.google.com/drive/folders/1QTLckP-G7bTR8e31xqW2rZYV7iPsF7H7?usp=sharing



### Architecture

| Architecture \ Task | Attribution                                  | Robust Dataset | Adversarial Attack |
| ------------------- | -------------------------------------------- | -------------- | ------------------ |
| Model One           | ResNet (pytorch pertained)                   |  cifar_test_non_robust              |   NA                 |
| Model Two           | ResNet ($\delta_2=0.25$, Madry pertained )   |  cifar_test_robust              |       PGD             |
| Model Three         | ResNet ($\delta_2=0.5$, Madry pertained )    |                |          PGD          |
| Model Four          | ResNet ($\delta_\infin=8$, Madry pertained ) |                |          PGD          |

