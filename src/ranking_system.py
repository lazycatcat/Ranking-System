import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as LA


# Data Preprocess and Markov chains modeling
def process_M():
    scores = pd.read_csv('data/CFB2016_scores.csv', header=None)
    team_names = pd.read_csv('data/TeamNames.txt', header=None)
    scores = pd.DataFrame.as_matrix(self=scores)
    team_names = np.ravel(pd.DataFrame.as_matrix(self=team_names))
    M = np.zeros((len(team_names), len(team_names)))
    for i in range(len(scores)):
        team_i = scores[i][0] - 1
        team_j = scores[i][2] - 1
        sum_score = np.float(scores[i][1] + scores[i][3])
        if (scores[i][1] > scores[i][3]):
            M[team_i, team_i] += 1 + scores[i][1] / sum_score
            M[team_j, team_j] += scores[i][3] / sum_score
            M[team_i, team_j] += scores[i][3] / sum_score
            M[team_j, team_i] += 1 + scores[i][1] / sum_score
        else:
            M[team_i, team_i] += scores[i][1] / sum_score
            M[team_j, team_j] += 1 + scores[i][3] / sum_score
            M[team_i, team_j] += 1 + scores[i][3] / sum_score
            M[team_j, team_i] += scores[i][1] / sum_score

    norm_M = [M[i, :] / sum(M[i, :]) for i in range(len(M))]
    return (norm_M, team_names)


def rank_top_teams(iteration_times, M, team_names):
    w_t = np.ones(len(M)) / np.float(len(M))
    rank = {i / 1000000.0: 0 for i in range(25)}
    for i in range(iteration_times):
        w_t = np.dot(w_t, M)
    for i, score in enumerate(w_t):
        if score > min(rank.keys()):
            rank.pop(min(rank.keys()))
            rank[score] = i
    print("iteration times" + ":" + str(iteration_times))
    print(" |     Score     |     Team     ")
    print(" |---------------|--------------")
    for key in sorted(rank.keys(), reverse=True):
        print(" | ", key, " | ", team_names[rank[key]])


def error(M, times):
    eigenvalues, eigenvectors = LA.eig(np.transpose(M))
    w_inf = np.transpose(eigenvectors[:, 4]) / np.sum(eigenvectors[:, 4])
    w_t = np.ones(len(M)) / np.float(len(M))
    error = []
    for i in range(times):
        w_t = np.dot(w_t, M)
        error.append(np.sum(np.abs(w_t - w_inf)))

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    iterations = np.linspace(1, times, times)

    plt.plot(iterations, error)
    axes.set_ylabel("Error")
    axes.set_xlabel("Iterations")
    plt.show()


# Data Preprocess and Nonnegative Matrix Factorization Modeling
def process():
    nyt_data = pd.read_table('data/nyt_data.txt', header=None, error_bad_lines=False, sep=" ")
    nyt_vocab = pd.read_csv('data/nyt_vocab.dat', header=None, error_bad_lines=False)
    nyt_vocab = np.ravel(pd.DataFrame.as_matrix(self=nyt_vocab))
    list_nyt_data = nyt_data.values.T.tolist()[0]
    X = np.zeros((3012, 8447))
    W = np.zeros((3012, 25))
    H = np.zeros((25, 8447))

    for i in range(len(W)):
        for j in range((len(W[0]))):
            W[i, j] = np.random.uniform(1, 2)

    for i in range(len(H)):
        for j in range((len(H[0]))):
            H[i, j] = np.random.uniform(1, 2)

    for i, row in enumerate(list_nyt_data):
        for word in row.split(","):
            tuple = word.split(":")
            X[int(tuple[0]) - 1, i] = int(tuple[1])

    for i in range(len(X[0])):
        indice = np.where(X[:, i] == 0)[0]
        X[indice, i] = 1.0 / 10000000000000000

    return X, W, H, nyt_vocab


def topics(X, W, H, times):
    D = []
    for i in range(times):
        purple = np.divide(X, np.dot(W, H))
        pink = W
        blue = H
        # Update H
        for j in range(len(W[0])):
            pink[:, j] = W[:, j] / np.sum(W[:, j])
        H = np.multiply(H, np.dot(np.transpose(pink), purple))

        purple = np.divide(X, np.dot(W, H))
        # Update W
        for k in range(len(H)):
            blue[k, :] = H[k, :] / np.sum(H[k, :])
        W = np.multiply(W, np.dot(purple, np.transpose(blue)))

        # Caculate Objective Function
        D.append(-np.sum(np.multiply(X, np.log(np.dot(W, H))) - np.dot(W, H)))

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    iterations = np.linspace(1, times, times)
    plt.plot(iterations, D)
    axes.set_ylabel("Objective function")
    axes.set_xlabel("Iterations")
    print(D)
    plt.show()

    return W


def words(W, nyt_vocab):
    for j in range(len(W[0])):
        W[:, j] = W[:, j] / np.sum(W[:, j])

    rank = []
    for i in range(len(W[0])):
        rank.append({i / 100000000.0: 0 for i in range(10)})
        for j in range(len(W)):
            if W[j, i] > min(rank[i].keys()):
                rank[i].pop(min(rank[i].keys()))
                rank[i][W[j, i]] = j

    # print("Column" + ":" + str(i + 1))
    for i in range(5):
        for j in range(5):
            print("|     Weight     |     Word     ", end='')
        print("|")
        for j in range(5):
            print("|----------------|--------------", end='')
        print("|")
        ordered_key_temp = [sorted(rank[j + i * 5].keys(), reverse=True) for j in range(5)]
        for k in range(10):
            for j in range(5):
                print(" | ", ordered_key_temp[j][k], " | ", nyt_vocab[rank[j + i * 5][ordered_key_temp[j][k]]], end='')
            print("|")
        for j in range(5):
            print("|----------------|--------------", end='')
        print("|")

if __name__ == "__main__":
    M,team_names =process_M()
    iteration_times = [10,100,1000,10000]
    for i in iteration_times:
        rank_top_teams(i, M,team_names)
    error(M,iteration_times[3])
    X, W, H, nyt_vocab = process()
    new_W = topics(X, W, H, 100)
    words(new_W, nyt_vocab)
