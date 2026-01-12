# Let's do simple vs complex word classification!
# 1 is complex, 0 is simple
# Don't worry about the hyperparameters, we will take a look at them later.
# training will take around 5 minutes
word_train_losses, word_val_losses = run(train_data = word_train_data,
                                         val_data = word_val_data,
                                         hidden_size = 50,
                                         n_epochs = 50000,
                                         learning_rate = 0.005,
                                         loss_func = nn.NLLLoss(),
                                         print_every = 5000,
                                         plot_every = 250,
                                         model_name = "./word_RNN"
                            )

word_train_data[0] # the words
word_train_data[1] # label for complex or simple

# len(word_val_losses) and len(word_train_losses) is 200, because we run n_epochs is 50000, plot every 250, 
# 50000/250 = 200


