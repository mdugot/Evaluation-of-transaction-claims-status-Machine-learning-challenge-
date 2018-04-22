# transaction_claims_prediction-Machine_learning_challenge

Using neural network, estimate if an online transaction will be subject to a claim from the user or not and if so, which kind of claim.  
The neural network is built on Tensorflow and can be used in the Python3 console as follows :  


>  \>\>\> from data import Data  
>  \>\>\> from nn import NeuralNetwork  
>  \>\>\> data = Data()  
>  \>\>\> nn = NeuralNetwork(data, [100, 100])  # Parameters : hidden layers  
>  \>\>\> nn.start()  
>  \>\>\> nn.train(1000, 100, 0.001, 0.5)  # Parameters : batch size, iterations, learning rate, dropout rate  
>  \>\>\> nn.save("result.csv")  
