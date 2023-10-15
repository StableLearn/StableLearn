#This code is for paper Covariate-Shift Generalization via Random Sample Weighting (AAAI2023)
#Thanks for https://github.com/facebookresearch/InvariantRiskMinimization and https://github.com/LJSthu/HRM, that this code refers to.

#Description for the data in Data file.
#Simulation data (experimental setting simulation1, see the description in paper for details）
data_10_1000_2_6: p=10, |Vb|=2, Scale=6, r=2
data_10_1000_3_6: p=10, |Vb|=2, Scale=6, r=3

#Simulation data (experimental setting simulation2, see the description in paper for details）
data_10_1000_2_7: p=10, |Vb|=2, Scale=7, r=2
data_10_1000_2_8: p=10, |Vb|=2, Scale=8, r=2

#Simulation data (experimental setting simulation3, see the description in paper for details）
data_20_1000_0.2: p=20, |Vb|/|V|=0.1, Scale=6, r=0.2
data_30_1000_0.2: p=30, |Vb|/|V|=0.1, Scale=6, r=0.2

#Real data (house price dataset, see the description in paper for details)
The dataset house.npy comes from https://www.kaggle.com/datasets/harlfoxem/housesalesprediction.

#Real data (adult dataset, see the description in paper for details)
The dataset adult_raceandsex.npy comes from https://archive.ics.uci.edu/ml/datasets/adult

#Real data (image dataset, see the description in paper for details)
CS-Colored MNIST dataset comes from paper Ahuja, K. and et al. Empirical or Invariant Risk Minimization? A Sample Complexity Perspective (ICLR2021)，which would be downloaded automatically when running the code.

#Metric description
For regression tasks (simulation dataset, house price dataset), the smaller the MSE or RMSE, the better performance; and the smaller the STD, the more stable;
For classification tasks (adult dataset, image dataset), the higher the accuracy, the better performance; and the smaller the STD, the more stable.

#running instruction
#For details about how to configure the code runtime environment, see the Requirement file.
For simulation dataset, house price dataset and adult dataset, it can get the results in the Table 1 and Figure 1 in paper (repeated experiments), by running main.py in Code file.
The input --data_path of main.py is the path of  dataset，--data_type is the type of dataset (1:simulation data, 2:house price data, 3:adult data). See the Hyper-parameter file for other model parameters in each dataset.
Example: for data_10_1000_2_6 dataset, run "python main.py --data_path data_10_1000_2_6 --lambda1 0.01 --lambda2 10. --lambda3 1e5 --num_env 3 --bias --data_type 1".

For image dataset, it can get the results in the Figure 3 in paper, by running main.py in Code/InvariantRiskMinimizationR/code/colored_mnist file.
set the ratio of label flip (0.2/0.25) at line 71 "labels = torch_xor(labels, torch_bernoulli(0.2, len(labels)))";
set the ratio of 2 colors (0.1/0.2) at line 82 and 83 "make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1),
                                                       make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),".
Example: for case (ratio of label flip = 0.25, ratio of 2 colors = 0.1), run "python main.py --penalty_weight 43".

