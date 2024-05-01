from param_choice import search_hyperparameters

layer_size_options = [[784, 256, 64, 10], [784, 128, 64, 10]]
learning_rate_options = [0.01, 0.001, 0.005]
reg_lambda_options = [0.01, 0.001]
search_hyperparameters(layer_size_options, learning_rate_options, reg_lambda_options, epochs=5, batch_size=100)
