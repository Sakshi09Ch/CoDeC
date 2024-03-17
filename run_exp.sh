#!/bin/bash

#--devices=4 by default, and it specifies the number of GPU cards used. Change to 1/2/3 based on the availability.

#Split CIFAR-100 experiments
#CoDeC(f)
python trainer_singlegpm_alexnet.py --world_size=4 --batch-size=88 
python trainer_singlegpm_alexnet.py --world_size=8 --batch-size=176

#CoDeC
python trainer_singlegpm_alexnet.py --world_size=4 --batch-size=88 --compress=True
python trainer_singlegpm_alexnet.py --world_size=8 --batch-size=176 --compress=True

#D-EWC
python trainer_gatherewc_alexnet.py --world_size=4 --batch-size=88 --lr=0.05 
python trainer_gatherewc_alexnet.py --world_size=8 --batch-size=176 --lr=0.05 

#D-SI
python trainer_gatherSI_alexnet.py --world_size=4 --batch-size=88 --epochs=100 --ewc_lamb=0.1 --lr=0.001
python trainer_gatherSI_alexnet.py --world_size=8 --batch-size=176 --epochs=100 --ewc_lamb=0.1 --lr=0.001

#Split miniImageNet experiments
#CoDeC(f)
python trainer_singlegpm_resnet18.py --dataset=miniimagenet --num_tasks=20 --classes=100 --threshold=0.985 --increment_th=0.0003 --world_size=4 --batch-size=40 --epochs=10
python trainer_singlegpm_resnet18.py --dataset=miniimagenet --num_tasks=20 --classes=100 --threshold=0.985 --increment_th=0.0003 --world_size=8 --batch-size=80 --epochs=10

#CoDeC
python trainer_singlegpm_resnet18.py --dataset=miniimagenet --num_tasks=20 --classes=100 --threshold=0.985 --increment_th=0.0003 --world_size=4 --batch-size=40 --epochs=10 --compress=True
python trainer_singlegpm_resnet18.py --dataset=miniimagenet --num_tasks=20 --classes=100 --threshold=0.985 --increment_th=0.0003 --world_size=8 --batch-size=80 --epochs=10 --compress=True

#D-EWC
python trainer_gatherewc_resnet.py --dataset=miniimagenet --num_tasks=20 --classes=100 --world_size=4 --batch-size=40 --epochs=10 --lr=0.03
python trainer_gatherewc_resnet.py --dataset=miniimagenet --num_tasks=20 --classes=100 --world_size=8 --batch-size=80 --epochs=10 --lr=0.03

#D-SI
python trainer_gatherSI_resnet.py --dataset=miniimagenet --world_size=4 --batch-size=40 --num_tasks=20 --ewc_lamb=0.3 --epochs=10 --lr=0.001
python trainer_gatherSI_resnet.py --dataset=miniimagenet --world_size=8 --batch-size=80 --num_tasks=20 --ewc_lamb=0.3 --epochs=10 --lr=0.001

#CoDeC with scaled gradient updates
python trainer_singlesgp_alexnet.py --world_size=4 --batch-size=88
python trainer_singlesgp_alexnet.py --world_size=4 --batch-size=88 --compress=True

#MedMNIST-5 experiments
python trainer_singlegpm_resnet18.py --lr=0.001 --dataset=medmnist --num_tasks=5 --threshold=0.99 --world_size=16 --batch-size=512 --epochs=50 --print-freq=200
python trainer_gatherSI_resnet.py --lr=0.001 --dataset=medmnist --num_tasks=5 --ewc_lamb=0.3 --epochs=50 --world_size=16 --batch-size=512 --print-freq=200
python trainer_gatherewc_resnet.py --lr=0.001 --dataset=medmnist --num_tasks=5 --world_size=16 --batch-size=512 --epochs=50 --print-freq=200