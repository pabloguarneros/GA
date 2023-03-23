import os
from copy import deepcopy
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from scipy.special import kl_div
from scipy.special import softmax
from scipy.spatial import distance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hyperparameters = {
    'generations': 1,
    'population': 10,
    'rou_switch': 0,
    'hid_nodes': 1,
    'host_mut_rate': 0.5,
    'host_mut_amount': 0.005,
    'parasite_mut_rate': 0.5,
    'parasite_mut_amount': 0.005,
    'visualize': False,
    'visualize_per': 0,
}

def load_data():
    """
    Helper function to load MNIST handwritten digits (Deng, 2012).
    
    This function has two purposes:
    1. Load the data
    2. Divide the datasaet into two parts: training and test dataset.

    Training dataset  is used to train the model and test dataset is used to check the accuracy.
    The dataset comes with images and correct labels of the digits.
    
    Deng, L., 2012. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), pp. 141–142.
    """

    train_csv = "data/mnist_train.csv"
    test_csv = "data/mnist_test.csv"

    data_train = pd.read_csv(train_csv) # load MNIST training data in
    data_train = np.array(data_train)   # turn into array

    np.random.shuffle(data_train)
    y_train = data_train[1:, 0]
    X_train = data_train[1:, 1:]

    data_test = pd.read_csv(test_csv)   # validating data loaded in
    data_test = np.array(data_test)     # turned to array and transposed
    np.random.shuffle(data_test)

    y_test = data_test[1:, 0]    # first row of data
    X_test = data_test[1:, 1:]  # rest of data

    #next two lines are taking 10,000 samples from MNIST
    X_train, X_val = X_train[:50000], X_train[50000:60000]
    y_train, y_val = y_train[:50000], y_train[50000:60000]

    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test}
    
    return data

class NonSpatial_Coev_GA:
    def __init__(self, hyperparameters, data):
        """
        Initialize key attributes for the model:
        NNs: host algorithm, content is initialized in the birth method.
        NNS_copy: copy of the host. Referenced during selection, and mutation for repopulation purpose.  
        all_train_score: keeps global max train accuracy
        all_val_score: keeps global max validation accuracy
        all_parasite_score: keeps global MNIST score
        cos_sim: keeps global genotype score
        entropy: keeps global phenotype score
        """
        self.hyperparameters = hyperparameters
        self.data = data
        self.NNs = {} # set of models for evolution. Swaps based on training score during evolution. 
        self.NNs_copy = {}  # for reference during evolution. Does not change during swaps.
        self.all_train_score = []
        self.all_val_score = []
        self.all_parasite_score = []
        self.neighbors = []
        self.cos_sim = []
        self.entropy = []
        self.results_path = self.make_results_path()

    def make_results_path(self):
        now = datetime.now().strftime("%y%m%d%H%M%S")
        results_path = f"{os.getcwd()}/results/nonspatial_coevolution_{now}"
        os.mkdir(results_path)
        os.mkdir(f"{results_path}/visualization")
        return results_path

    ############## 1. Initialize host and parasite ##############
    def birth(self):
        """
        Produce host and parasite. 

        The pair is pacakged into a dictionary containing:
            - MLPClassifier model
            - training score 
            - validation score 
            - parasites (10 digits from each class)

        """
        for ind in range(self.hyperparameters["population"]):
            self.NNs[ind] = {"model": MLPClassifier(hidden_layer_sizes=(self.hyperparameters["hid_nodes"],), 
                                                    max_iter=1, 
                                                    alpha=1e-4,
                                                    solver='sgd', 
                                                    verbose=False, 
                                                    learning_rate_init=.1),
                            "train_score": 0,
                            "val_score": 0,
                            "parasite_X_train": None,
                            "parasite_y_train": None,
                            "parasite_score": 0}

            ### 1.1 populate host (Neural Network Classifiers) ###
            
            # fit the network to initialize W and b
            self.NNs[ind]["model"].fit(self.data["X_train"], self.data["y_train"])
            
            # randomly initialize weights and biases // genome of neural networks as genotype
            self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1, high=1, size=(784, self.hyperparameters["hid_nodes"])) 
            self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1, high=1, size=(self.hyperparameters["hid_nodes"], 10))

            ### 1.2 populate parasite (MNIST handwritten digits) ###
            indices = []
            counter = 0
            for num in np.unique(self.data["y_train"], return_counts = True)[1]:
                # for each class of digit, randomly pick 1000 images with replacement
                indices.extend(np.random.randint(counter, counter + num, 50))
                counter += num
            self.NNs[ind]["parasite_X_train"] = self.data["X_train"][indices]
            self.NNs[ind]["parasite_y_train"] = self.data["y_train"][indices]

    ############## 2.Run Non-spatial coevolution ##############
    
    # 2.1 calculate fitness for host
    def calculate_fitness(self):
        """
        Calculate max score for the host and paraste in each generation.

        Fitness for host NN algorithm (line 166 and line 169): correct classification percentage
        Fitness for host MNIST algoirithm (line 180 and line 181): misclassifiation percentage
        """
        train_score = []
        val_score = []
        parasite_score = []
        cf_matrix = []

        for ind in self.NNs:
            current_mlp = self.NNs[ind]
            # calculate training score
            current_mlp["train_score"]= current_mlp["model"].score(current_mlp["parasite_X_train"], 
                                                                   current_mlp["parasite_y_train"]) 
            # calculate validation score
            current_mlp["val_score"]= current_mlp["model"].score(self.data["X_val"], self.data["y_val"])

            train_score.append(current_mlp["train_score"])
            val_score.append(current_mlp["val_score"])
            
            ## output confusion matrix
            y_train_pred = current_mlp["model"].predict(current_mlp["parasite_X_train"])
            cf_matrix.append(confusion_matrix(current_mlp["parasite_y_train"], y_train_pred))

            ### compute parasite score ###
            true_result = current_mlp["parasite_y_train"] == y_train_pred
            current_mlp["parasite_score"]  = 1 - (sum(true_result) / self.hyperparameters["population"])

            parasite_score.append(current_mlp["parasite_score"])

        self.NNs_copy = deepcopy(self.NNs) # clone the population
        print("Max training score: ", np.amax(train_score))
        print("Max validation score: ", np.amax(val_score))
        print("Max parasite score: ", np.amax(parasite_score))
        self.all_train_score.append(np.amax(train_score))
        self.all_val_score.append(np.amax(val_score))
        self.all_parasite_score.append(np.amax(parasite_score))
        
        # updates the value score and confusion matrix
        self.entropy_calculator(cf_matrix)
        self.cosine_sim()

    # 2.2 select the best performing host and parasite
    def select_top_performing(self):
        self.calculate_fitness()
        self.select("val_score", ["model"])
        self.select("parasite_score", ["parasite_X_train", "parasite_y_train"])

    def select(self, host_par_score, host_par_model):
        """
        Select top perfoming individuals.

        This is a probabilistic replacement where the top performing individuals are proportionately selected.
        The elitist strategy is from Mitchell (2006), howerver recent suggestions consider diveristy Mouret, J. B. (2020). 
        """
        # make an array w prob. distribution
        all_scores = []
        # extract score from the model
        for ind in self.NNs:
            all_scores.append(self.NNs[ind][host_par_score])
        all_probs = softmax(all_scores)

        # loop through each cell and chose the new one
        for idx in self.NNs:
            new_idx = np.random.choice(a=self.hyperparameters["population"], size=1, p=all_probs)[0]
            for i in host_par_model:
                self.NNs[idx][i] = deepcopy(self.NNs_copy[new_idx][i])

    # 2.3 mutation both host and parasite
    def mutate_system(self):
        for idx in range(self.hyperparameters["population"]):
            self.mutate_host(idx, hyperparameters["host_mut_rate"], hyperparameters["host_mut_amount"])
            self.mutate_parasite(idx, hyperparameters["parasite_mut_rate"], hyperparameters["parasite_mut_amount"])

    def mutate_host(self, idx, mut_rate=0.5, mut_amount=0.005):
        """
        In each layer, mutate weights at random cites with probability "mut_rate", Mitchell (2006).
        ---
        idx: int
        """

        for i in range(2):
            # randomly select mutation sites
            coef_size = self.NNs[idx]["model"].coefs_[i].size
            shape = self.NNs[idx]["model"].coefs_[i].shape
            mutation_sites = np.random.choice(coef_size, size=int(mut_rate * coef_size))

            # mutate mut_amount sampled from a normal distribution
            for loci in mutation_sites:
                self.NNs[idx]["model"].coefs_[i].flat[loci] += np.random.normal(loc=mut_amount)
            self.NNs[idx]["model"].coefs_[i].reshape(shape)
    
    def mutate_parasite(self, idx, mut_rate=0.5, mut_amount=10):
        """
        Mutate weights at random cites with probability "mut_rate", Mitchell (2006).
        ---
        idx: int
        """
        for image in self.NNs[idx]["parasite_X_train"]:
            # randomly select mutation sites
            shape = image.shape[0]
            mutation_sites = np.random.choice(shape, size=int(mut_rate * shape))

            # mutate mut_amount sampled from a normal distribution
            for loci in mutation_sites:
                image[loci] += np.random.normal(loc=mut_amount)
        return None

    ############## 3.Measure phenotype, genotype and store result ##############
    def entropy_calculator(self, cf_matrix):
        """
        Compute KL-divergence (distance between metrices) to characterize phenotype Mitchell 2006.
        normalize each row of the confusion matrix for KL-Divergence calculation
        """
        entropy_periter = []

        for i in range(len(cf_matrix)):
            for j in range(len(cf_matrix)):
                if i >= j:
                    continue
                for row in range(10):
                    # add 1e-4 to avoid overflow (division by 0 for log)
                    entropy_periter.append(kl_div(normalize(1e-4+cf_matrix[i], axis=1, norm='l1')[row],
                                                normalize(1e-4+cf_matrix[j], axis=1, norm='l1')[row]))

        mean_KL = np.average(entropy_periter)
        self.entropy.append(mean_KL)
        print("KL-Divergence: ", mean_KL)

    def cosine_sim(self):
        """
        Vectorize weights of each host NN into a single genome and measure the angular difference to capture genotype diveristy.
        """
        current_div = []
        for ind1 in self.NNs:
            for ind2 in self.NNs:
                if ind1 >= ind2:
                    continue
                dist = distance.cosine(np.concatenate((np.ravel([self.NNs[ind1]["model"].coefs_[0]]),
                                                        np.ravel([self.NNs[ind1]["model"].coefs_[1]]))), 
                                        np.concatenate((np.ravel([self.NNs[ind2]["model"].coefs_[0]]), 
                                                        np.ravel([self.NNs[ind2]["model"].coefs_[1]]))))
                current_div.append(dist)
        div_score = np.mean(current_div)
        self.cos_sim.append(div_score)
        print("Cosine Similarity: ", div_score,"\n")

    def store_result(self):
        
        results = {
            "train_score":self.all_train_score,
            "val_score":self.all_val_score,
            "parasite_score":self.all_parasite_score,
            "cos_sim":self.cos_sim,
            "rel_ent":self.entropy,
            "hyp_params":self.hyperparameters}
        
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in results.items() ]))
        df.to_csv(f"{self.results_path}/results.csv")

    def capture_mnist_state(self, i):

        if i % hyperparameters["visualize_per"] == 0:
            _, axes = plt.subplots(2, 5)
            for i in range(10):
                image = self.NNs[0]['parasite_X_train'][i]
                label = self.NNs[0]['parasite_y_train'][i]
                ax = axes[i//5, i%5]
                ax.imshow(np.reshape(image,(28,28)), cmap='gray')
                ax.set_title('Label: {}'.format(label))

            plt.tight_layout()
            plt.savefig(f"{self.results_path}/visualization/{i}.png")

def run():
    print("Generations: ", hyperparameters["generations"])
    print("Population: ", hyperparameters["population"])
    print("Host mutation rate: ", hyperparameters["host_mut_rate"])
    print("Host mutation amount: ", hyperparameters["host_mut_amount"])
    print("Parasite mutation rate: ", hyperparameters["parasite_mut_rate"])
    print("Parasite mutation amount: ", hyperparameters["parasite_mut_amount"])

    ######### 1. Load Data (host, and parasite) #########

    print("Data is loading.")
    data = load_data()
    print("Data has loaded.")

    ######### 2. Initialize Population (host, and parasite) #########
    
    model = NonSpatial_Coev_GA(hyperparameters, data)
    model.birth()
    
    ######### 2. Run Co-Evolution #########

    for i in range(hyperparameters["generations"]):
        print("\ncurrent generation: ", i)
        # coevolve
        model.select_top_performing()
        model.mutate_system()

        # capture state
        if hyperparameters["visualize"]:
            model.capture_mnist_state(i)
    model.store_result()

if __name__=="__main__":
    run()