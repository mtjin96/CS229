import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    mes = message.lower()
    words = mes.split(' ')
    return words
    # *** END CODE HERE ***

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    word_dict = {}
    for mes in messages:
        word_lst = get_words(mes)
        word_lst_unq = set(word_lst)
        # check the word appears in how many messages
        for item in word_lst_unq:
            if item in word_dict:
                word_dict[item] = word_dict[item] + 1
            else:
                word_dict[item] = 1

    for item in word_dict.copy():
        if word_dict[item] < 5:
            del word_dict[item]
    return word_dict
    # *** END CODE HERE ***

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    row = len(messages)
    col = len(word_dictionary)
    s = (row, col)
    word_freq_array = np.zeros(s, dtype=int)

    word_lst = []
    for i in range(len(messages)):
        word_lst.append(get_words(messages[i]))
    c = -1
    for word in word_dictionary:
        c = c + 1
        # check the occurances of a word in the whole messages
        for r in range(len(word_lst)):
            for item in word_lst[r]:
                if item == word:
                    word_freq_array[r, c] = word_freq_array[r, c] + 1
    return word_freq_array
    # *** END CODE HERE ***

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    sample_sz = len(labels)
    word_num = matrix.shape[1]
    I0, I1 = [], []
    for i in range(sample_sz):
        if labels[i] == 0:
            I0.append(i)
        if labels[i] == 1:
            I1.append(i)
    spam = matrix[I1]
    ham = matrix[I0]

    y1 = len(I1)
    y0 = len(I0)
    # MLE
    phi_y = np.log(y1 / sample_sz)
    phi_y_0 = np.log(y0 / sample_sz)

    # find phi_k conditioning on y is 1 (i.e.it's spam)
    phi_k_1 = [0] * word_num
    deno = sum(sum(spam))
    for k in range(word_num):
        num = 0
        for i in range(spam.shape[0]):
            num += spam[i, k]
        phi_k_1[k] = np.log((num + 1) / (deno + word_num))

    # find phi_k conditioning on y is 0 (i.e.it's ham)
    phi_k_0 = [0] * word_num
    deno2 = sum(sum(ham))
    for k in range(word_num):
        num2 = 0
        for i in range(ham.shape[0]):
            num2 += ham[i, k]
        phi_k_0[k] = np.log((num2 + 1) / (deno2 + word_num))

    return phi_y, phi_y_0, phi_k_1, phi_k_0

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: The trained model
    """
    # *** START CODE HERE ***
    word_num = matrix.shape[1]
    phi_y = 0
    phi_y_0 = 0
    phi_k_0 = [0] * word_num
    phi_k_1 = [0] * word_num
    phi_y, phi_y_0, phi_k_1, phi_k_0 = model


    # predict given x
    size = matrix.shape[0]
    y_pred = [0] * size

    for i in range(size):
        sum_1 = 0
        sum_0 = 0
        for k in range(word_num):
            freq = matrix[i, k]
            sum_1 = sum_1 + (phi_k_1[k] * freq)
            sum_0 = sum_0 + (phi_k_0[k] * freq)
        pos = sum_1 + phi_y
        neg = sum_0 + phi_y_0

        if neg > pos:
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    return y_pred
    # *** END CODE HERE ***

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    word_num = len(dictionary)
    phi_y = 0
    phi_y_0 = 0
    phi_k_0 = [0] * word_num
    phi_k_1 = [0] * word_num
    phi_y, phi_y_0, phi_k_1, phi_k_0 = model
    ind_scr = dict()
    i = -1
    for key in dictionary:
        i += 1
        ind_scr[key] = phi_k_1[i] - phi_k_0[i]
    scr_sorted_keys = sorted(ind_scr, key=ind_scr.get, reverse=True)
    return scr_sorted_keys[0:5]
    # *** END CODE HERE ***

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        eval_matrix: The word counts for the validation data
        eval_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accuracy_old = 0
    rad_max = radius_to_consider[0]
    for rad in radius_to_consider:
        pred = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, rad)
        accuracy = np.mean(pred == val_labels)
        if accuracy > accuracy_old:
            accuracy_old = accuracy
            rad_max = rad
    return rad_max
    # *** END CODE HERE ***

def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')

    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)

    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

if __name__ == "__main__":
    main()