#Machine Learning
LINEARREGRESSION = 0
RANDOMFOREST = 1
SVRLINEAR = 2
SVRPOLY = 3
SVRGAUSS = 4
CGA2M = 5
LIGHTGBM = 6
NEURALNETWORK = 7
QUADRATICREGRESSION = 8
WEIGHTEDLINEARREGRESSION = 9
ANNLINEARREGRESSION = 10
KNNLINEARREGRESSION = 11
POLYNOMIALREGRESSION = 12
GAUSSIANPROCESSREGRESSION = 13

#Problem
NONCONSTRAINT = 0
BOXCONSTRAINT = 1
BOXFEATURECONSTRAINT = 2
DISKCONSTRAINT = 3
INEQUALITYCONSTRAINT = 4
REALCONSTRAINT = 5

#Data Type
COMPLEX7 = 0
SINX0 = 1
SINX0MOUNT2 = 2
ROSENBROCK = 3
ACKELY = 4
XSQUARE = 5
SINX01MOUNT2 = 6
MOUNT2 = 7
REAL = 8
RASTRIGIN = 9
LOGX = 10
LOGX2 = 11

#Opt Method
MATHMATICALOPTIMIZATION = 0
SIMULATEDANNEALING = 1
TRUSTREGION = 2
FRANKWOLFE = 3
STEPDISTANCE = 4
BAYESIANOPTIMIZATIONMU = 5
BAYESIANOPTIMIZATIONLCB = 6
SELECTBESTDATA = 7
FRANKWOLFE2 = 8

#Formalization
LP = 0
MIP = 1
NLP = 2

#See Status
SEEALL = 0
SEELAST = 1
DONTSEE = 2

#ANN Library
ANNOY = "annoy"
NMSLIB = "nmslib"
FAISS = "faiss"

#Environment
TSUBAME = "tsubame"
LOCAL = "local"
DOCKER = "docker"

#Names
mlmodel_names = {LINEARREGRESSION: "lr", RANDOMFOREST: "rf", SVRLINEAR: "sl", SVRPOLY: "sp", SVRGAUSS: "sg", CGA2M:"c2", LIGHTGBM: "lg", NEURALNETWORK: "nn", WEIGHTEDLINEARREGRESSION: "wl",
                 ANNLINEARREGRESSION: "al", KNNLINEARREGRESSION: "kl", POLYNOMIALREGRESSION: "pr", GAUSSIANPROCESSREGRESSION: "gp"}
problem_names = {NONCONSTRAINT: "non-constraint", BOXCONSTRAINT: "box-constraint", BOXFEATURECONSTRAINT: "box-feature-constraint", DISKCONSTRAINT: "disk-constraint", INEQUALITYCONSTRAINT: "inequality-constraint"}
data_type_names = {COMPLEX7: "complex7", SINX0: "sinx0", SINX0MOUNT2: "sinx0mount2", ROSENBROCK: "rosenbrock", ACKELY: "ackely", XSQUARE: "xsquare", 
                   SINX01MOUNT2: "sinx01mount2", MOUNT2: "mount2", RASTRIGIN: "rastrigin", LOGX: "logx", LOGX2: "logx2"}
solvers_names = {SIMULATEDANNEALING: "SimulatedAnnealing", TRUSTREGION: "TrustRegion", MATHMATICALOPTIMIZATION: "MathmaticalOptimization", FRANKWOLFE: "FrankWolfe",
                              STEPDISTANCE: "StepDistance", BAYESIANOPTIMIZATIONMU: "BayesianOptimizationMu", BAYESIANOPTIMIZATIONLCB: "BayesianOptimizationLcb",
                              SELECTBESTDATA: "SelectBestData", FRANKWOLFE2: "FrankWolfe2"}

#The index of Tree's feature
node_info = {"node_index": 2, "left_child": 3, "right_child": 4, "split_feature": 6, "threshold": 8, "value": 12}


#Color
color_dic = {
             (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[POLYNOMIALREGRESSION]): "c",
             (solvers_names[SELECTBESTDATA], mlmodel_names[KNNLINEARREGRESSION]): "m",     
             (solvers_names[SIMULATEDANNEALING], mlmodel_names[LIGHTGBM]): "red",
             (solvers_names[FRANKWOLFE], mlmodel_names[KNNLINEARREGRESSION]): "blue",
             (solvers_names[FRANKWOLFE2], mlmodel_names[KNNLINEARREGRESSION]): "green",

             (solvers_names[SIMULATEDANNEALING], mlmodel_names[NEURALNETWORK]): "c",
             (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[LINEARREGRESSION]): "black",
             (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[SVRGAUSS]): "m",
             (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[SVRPOLY]): "yellow",

             (solvers_names[BAYESIANOPTIMIZATIONMU], mlmodel_names[GAUSSIANPROCESSREGRESSION]): "blue",
             (solvers_names[BAYESIANOPTIMIZATIONLCB], mlmodel_names[GAUSSIANPROCESSREGRESSION]): "blue",
       

             (solvers_names[FRANKWOLFE], mlmodel_names[WEIGHTEDLINEARREGRESSION]): "red",
             (solvers_names[FRANKWOLFE], mlmodel_names[ANNLINEARREGRESSION]): "red",
             (solvers_names[STEPDISTANCE], mlmodel_names[ANNLINEARREGRESSION]): "blue",
             (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[CGA2M]): "blue",
}

#Plot Coordinate
plot_coordinate_dic = {
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[POLYNOMIALREGRESSION]): 0.1,
            (solvers_names[SELECTBESTDATA], mlmodel_names[KNNLINEARREGRESSION]): -0.2,  
            (solvers_names[SIMULATEDANNEALING], mlmodel_names[LIGHTGBM]): -0.1,
            (solvers_names[FRANKWOLFE], mlmodel_names[KNNLINEARREGRESSION]): 0,
            (solvers_names[FRANKWOLFE2], mlmodel_names[KNNLINEARREGRESSION]): 0.2,


            (solvers_names[SIMULATEDANNEALING], mlmodel_names[NEURALNETWORK]): -0.2,
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[LINEARREGRESSION]): 0.2,
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[SVRGAUSS]): 0.1,
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[SVRPOLY]): 0.3, 

            (solvers_names[BAYESIANOPTIMIZATIONMU], mlmodel_names[GAUSSIANPROCESSREGRESSION]): 0.2,
            (solvers_names[BAYESIANOPTIMIZATIONLCB], mlmodel_names[GAUSSIANPROCESSREGRESSION]): 0.25,

            (solvers_names[FRANKWOLFE], mlmodel_names[WEIGHTEDLINEARREGRESSION]): 0,
            (solvers_names[FRANKWOLFE], mlmodel_names[ANNLINEARREGRESSION]): 0,
            (solvers_names[STEPDISTANCE], mlmodel_names[ANNLINEARREGRESSION]): 0.3,
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[CGA2M]): 0.3,
}

plot_name_dic = {
            (solvers_names[SIMULATEDANNEALING], mlmodel_names[LIGHTGBM]): "LightGBM+SA",
            (solvers_names[SELECTBESTDATA], mlmodel_names[KNNLINEARREGRESSION]): "過去の最良解",  
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[POLYNOMIALREGRESSION]): "多項式回帰+IPOPT",
            (solvers_names[FRANKWOLFE], mlmodel_names[KNNLINEARREGRESSION]): r"提案手法($\gamma=\frac{2}{k+2}$)",
            (solvers_names[FRANKWOLFE2], mlmodel_names[KNNLINEARREGRESSION]): r"提案手法($\gamma=\frac{1}{1000+k}$)",

            (solvers_names[SIMULATEDANNEALING], mlmodel_names[NEURALNETWORK]): "NN+SA",
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[LINEARREGRESSION]): "LinearRegression+CBC",
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[SVRGAUSS]): "SVR+IPOPT",
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[SVRPOLY]): "SVR(POLY)+IPOPT",

            (solvers_names[BAYESIANOPTIMIZATIONMU], mlmodel_names[GAUSSIANPROCESSREGRESSION]): "ベイズ最適化(Mu)",
            (solvers_names[BAYESIANOPTIMIZATIONLCB], mlmodel_names[GAUSSIANPROCESSREGRESSION]): "ベイズ最適化(LCB)",

            (solvers_names[FRANKWOLFE], mlmodel_names[WEIGHTEDLINEARREGRESSION]): "proposed method",
            (solvers_names[FRANKWOLFE], mlmodel_names[ANNLINEARREGRESSION]): "proposed method",
            (solvers_names[STEPDISTANCE], mlmodel_names[ANNLINEARREGRESSION]): "proposed method",
            (solvers_names[MATHMATICALOPTIMIZATION], mlmodel_names[CGA2M]): "proposed method",
}

# plot rate
plot_colormap_rate_dic = {SINX0: 10, SINX0MOUNT2: 10, SINX01MOUNT2: 10, ROSENBROCK: 10, ACKELY: 1.5, XSQUARE: 10, MOUNT2: 10}