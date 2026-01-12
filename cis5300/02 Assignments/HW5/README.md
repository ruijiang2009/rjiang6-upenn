"/content/glove.6B.50d.magnitude",
"/content/glove.6B.100d.magnitude",
"/content/glove.6B.200d.magnitude",
"/content/glove.6B.300d.magnitude",
"/content/glove.840B.300d.magnitude",


Development Data: The development data consists of two files:
1. words file (input)
2. clusters file (output)

dev_input.txt

k input to clustering algorithm

target.pos :: k :: paraphrase1 paraphrase2 paraphrase3 ...


The clusters file dev_output.txt contains the ground truth clusters for each target word’s paraphrase set, split over k lines:

target.pos :: 1 :: paraphrase2 paraphrase6
target.pos :: 2 :: paraphrase3 paraphrase4 paraphrase5
    .
    .
    .
target.pos :: k :: paraphrase1 paraphrase9

k - number of ground truth clusters and their paraphrase sets

3.1-3.3 only test_input.txt

3,4 test_nok_input.txt 
containing. the test tyarget words and their paraphrases sets.


The general idea behind paired F-score is to treat clustering prediction like a classification problem; given a target word and its paraphrase set, we call a positive instance any pair of paraphrases that appear together in a ground-truth cluster. Once we predict a clustering solution for the paraphrase set, we similarly generate the set of word pairs such that both words in the pair appear in the same predicted cluster. We can then evaluate our set of predicted pairs against the ground truth pairs using precision, recall, and F-score.

V-Measure is another metric that is used to evaluate clustering solutions, however we will not be using it in this Assignment.

4 functions

Tasks: Your task is to fill in 4 functions: 
1. cluster_random,
2. cluster_with_sparse_representation,
3. cluster_with_dense_representation,
4. cluster_with_no_k.

We provided 5 utility functions for you to use:

1. load_input_file(file_path) that converts the input data (the words file) into 2 dictionaries. The first dictionary is a mapping between a target word and a list of paraphrases. The second dictionary is a mapping between a target word and a number of clusters for a given target word.

2. load_output_file(file_path) that converts the output data (the clusters file) into a dictionary, where a key is a target word and a value is it’s list of list of paraphrases. Each list of paraphrases is a cluster. Remember that Neither order of senses, nor order of words in a cluster matter.

3. get_paired_f_score(gold_clustering, predicted_clustering) that calculates paired F-score given a gold and predicted clustering for a target word.

4. evaluate_clusterings(gold_clusterings, predicted_clusterings) that calculates paired F-score for all target words present in the data and prints the final F-Score weighted by the number of senses that a target word has.

5. write_to_output_file(file_path, clusterings) that writes the result of the clustering for each target word into the output file (clusters file) Full points will be awarded for each of the tasks if your implementation gets above a certain threshold on the test dataset. Please submit to autograder to see thresholds. Note that thresholds are based on the scores from the previous year and might be lowered depending on the average performance.


Sample usage
```
word_to_paraphrases_dict, word_to_k_dict = load_input_file('dev_input.txt')
gold_clusterings = load_output_file('dev_output.txt')
predicted_clusterings = cluster_random(word_to_paraphrases_dict, word_to_k_dict)
evaluate_clusterings(gold_clusterings, predicted_clusterings)
```