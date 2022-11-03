# NN_from_scratch
Simple dense neural network application using mostly Numpy in Python, and the base functions in R. Kaggle's Dogs vs Cats dataset is used as a test.

There are two main notebooks with the same content. One focuses on building the application in Python, and the other one in R. Both attempt to use as
little external packages and libraries as possible.

## The data

The set consists of 25000 images balancedly divided into cats and dogs. For the purpose of this project I used 85% of them for the train set.

## The code
The notebooks generate several other .py, .R, or ,jl files which are then imported/sourced in each respective notebook. These were done in this way just a way
to keep things organized, but to launch the application one needs only to call the `dense_nn()` function. If the data set is not ready one might also want 
to call the `create_dataset()` function too.

There are shallow explanations to the different steps in a dense neural network in the [Python notebook](https://github.com/williantleite/NN_from_scratch/blob/main/Python-NN%20from%20Scratch.ipynb). The [R notebook](https://github.com/williantleite/NN_from_scratch/blob/main/R-NN%20from%20Scratch.ipynb) and [Julia notebook](https://github.com/williantleite/NN_from_scratch/blob/main/Julia-NN%20from%20Scratch.ipynb) contain in principle the same code as the python version but adapted to the specific syntax of these languages.

## Future improvements
Should I ever have the energy to go back to this project I would like to make a class object in both languages to facilitate using all the different functions and 
retrieving cached information from them. It would be nice to also introduce other optimization algorithms, or new functions like regularization, data augmentation, 
and convolutional layers.
