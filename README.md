# DL-IS800-Project-UMBC
Code implementing IS800 course project, titled as "AE-NCF: An Autoencoder-informed Neural Collaborative Filtering Model".

## Team Members
* Rashidul Islam (email: dr97531@umbc.edu)
* Kamrun Naher Keya (email: kkeya1@umbc.edu)

## Required Libraries
* NumPy
* pandas
* scikit-learn
* PyTorch

## Instructions
* To replicate results for our proposed AE-NCF model, simply run the following python file: "run_our_ae_ncf.py". At first, this code will find out the best hyperparameters for the model via grid search on the development set. Finally, the code will evaluate the model on the test set and generate the experimental results in the "results" folder as "proposed_ae_ncf_model_results.txt" file. 
* Similarly, to replicate results for the 3 baseline models, run the following python files: "run_ncf.py", "run_user_autoencoder.py", and "run_item_autoencoder.py". Each code will conduct hyperparameter selection on the development set at first and evaluate the model on the test set. Generated results from each model will be saved in the "results" folder as "ncf_results.txt", "user_based_ae_results.txt", and "item_based_ae_results.txt", respectively. 
