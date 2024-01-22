import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class LinearRegression:
    def __init__(self):
        pass

    def get_params(self):
        return self.w, self.b

    def set_params(self, w_inp, b_inp):
        self.w = w_inp.reshape(self.n, 1)
        self.b = b_inp

    def predict(self, X):
        yhat = np.matmul(X, self.w) + self.b
        return yhat

    def mean_square_cost(self, X, Y, lambda_):
        # X-->(m,n) Y-->(m,1) w-->(n) b-->scalar
        cost = 0
        # Dot product for each example and then mean square.
        f_wb = np.matmul(X, self.w) + self.b  # f_wb-->(m,1)
        cost = np.sum((f_wb - Y)**2)
        cost = cost / 2
        # Regularization Term
        reg_cost = np.sum(np.power(self.w, 2))
        reg_cost = reg_cost * (lambda_/2)
        total_cost = cost + reg_cost

        return total_cost

    def get_gradients(self, X, Y, lambda_):
        # X-->(m,n) Y-->(m,1) w-->(n)1D array b-->scalar
        mini_batch_length = X.shape[0]
        dj_dw = np.zeros((self.n, 1))
        dj_db = 0

        f_wb = np.matmul(X, self.w) + self.b  # f_wb-->(m,1)
        err = f_wb - Y  # err-->(m,1)
        dj_db = np.sum(err)
        dj_db = dj_db / mini_batch_length

        # err-->(m,1) and X-->(m,n)1D
        dj_dw = np.matmul(X.T, err)  # dj_dw --> (n,1)
        dj_dw = dj_dw / mini_batch_length

        # Regularization term
        dj_dw += self.w * (lambda_ / mini_batch_length)

        return dj_db, dj_dw

    def fit(self, x, y, alpha=0.01, epochs=10000, lambda_=0, plot_cost=False):
        # x-->(m,n) y-->(m,1) w-->(n,1) b-->scalar, alpha, iterations-->scalar
        self.m, self.n = x.shape
        self.cost_history = []
        self.w = np.zeros((self.n, 1))  # w-->(n,1)
        self.b = 0
        batch_size = 30

        for i in range(1, epochs+1):
            zipped_data = np.hstack((x, y))
            np.random.shuffle(zipped_data)
            mini_batches = [zipped_data[k:k+batch_size]
                            for k in range(0, self.m, batch_size)]
            cost = 0
            # Using Mini Batch gradient Descent.
            for mini_batch in mini_batches:
                X = mini_batch[:, :self.n]
                Y = mini_batch[:, self.n:]

                dj_db, dj_dw = self.get_gradients(X, Y, lambda_)
                # Update values of w and b at each iteration
                self.w = self.w - alpha * dj_dw
                self.b = self.b - alpha * dj_db

                cost += self.mean_square_cost(X, Y, lambda_)
            cost /= self.m
            # Save cost J at each iteration
            self.cost_history.append(cost)
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i == 1:
                # Initiate the spacing for alignment of output.
                width = len(f'{self.cost_history[-1]:.3f}')

            if i % math.ceil(epochs / 10) == 0:
                print(
                    f'Epoch {i:3d} ⫸ Cost {self.cost_history[-1]:^{width}.3f} ⫷')

        if plot_cost:
            plt.plot(np.arange(epochs), self.cost_history)
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.title('Cost vs Epochs')
            plt.show()
        self.cost_history = np.array(self.cost_history)
        return self.w, self.b, self.cost_history


class CreatePolynomialFeatures:
    def __init__(self):
        pass

    @staticmethod
    def combinations(degree):
        pows = []
        for x1 in range(degree+1):
            for x2 in range(degree+1):
                x3 = degree - (x1 + x2)
                if x3 < 0:
                    break
                else:
                    pows.append((x1, x2, x3))
        return pows

    @staticmethod
    def poly_transform(X, degree):
        if degree == 1:
            result = X
            return result

        elif degree > 1:
            m, n = X.shape

            new_result = np.empty((m, 0))
            prev_result = CreatePolynomialFeatures.poly_transform(X, degree-1)
            new_result = np.concatenate((new_result, prev_result), axis=1)

            temp_result = np.empty((m, 0))
            powers = CreatePolynomialFeatures.combinations(degree)
            for pow in powers:
                result_col = X**pow
                result_col = np.prod(result_col, axis=1)
                temp_result = np.concatenate(
                    (temp_result, result_col.reshape(-1, 1)), axis=1)

            new_result = np.concatenate((new_result, temp_result), axis=1)
            return new_result


class LogisticRegression:
    def __init__(self):
        pass

    def get_params(self):
        return self.w, self.b

    def set_params(self, w_inp, b_inp):
        self.w = w_inp.reshape(self.n, self.c)
        self.b = b_inp.reshape(1, self.c)

    def predict(self, X):
        z = np.matmul(X, self.w) + self.b
        yhat = Utilities.sigmoid(z)  # yhat-->(m,c)
        if yhat.shape[1] > 1:
            predictions = np.argmax(yhat, axis=1)
        else:
            predictions = np.where(yhat >= 0.5, 1, 0)
        return predictions.reshape(-1, 1)

    def logistic_cost(self, X, Y, lambda_):
        # X-->(m,n) Y-->(m,c) w-->(n,c) b-->(1,c)
        cost = 0

        z = np.matmul(X, self.w) + self.b  # f_wb-->(m,c)
        f_wb = Utilities.sigmoid(z)
        cost = np.sum(-Y * np.log(f_wb + 0.0000001) - (1-Y)
                      * np.log(1 - f_wb + 0.0000001), axis=0)  # (c,)1D array
        # Regularization Term
        reg_cost = np.sum(np.power(self.w, 2), axis=0)  # (c,)1D array
        reg_cost = reg_cost * (lambda_/2)
        total_cost = cost + reg_cost

        return total_cost

    def get_gradients(self, X, Y, lambda_):
        # X-->(m,n) Y-->(m,c) w-->(n,c) b-->(1,c)
        mini_batch_length = X.shape[0]
        dj_dw = np.zeros((self.n, self.c))
        dj_db = np.zeros((1, self.c))

        z = np.matmul(X, self.w) + self.b  # f_wb-->(m,c)
        f_wb = Utilities.sigmoid(z)
        err = f_wb - Y  # err-->(m,c)
        dj_db = np.sum(err, axis=0)
        dj_db = dj_db / mini_batch_length
        # err-->(m,c) and X.T-->(n,m)
        # Dot product between each feture of X and each class in err.
        dj_dw = np.matmul(X.T, err)  # dj_dw-->(n,c)
        dj_dw = dj_dw / mini_batch_length
        # Regularization term
        dj_dw += self.w * (lambda_ / mini_batch_length)

        return dj_db, dj_dw

    def fit(self, x, y, alpha=0.01, epochs=10000, lambda_=0, plot_cost=False):
        # x-->(m,n) y-->(m,c) w-->(n,c), b-->(1,c), alpha, iterations-->scalar, c-->no. of classes
        # One hot encode the target matrix if multi class classification.
        if len(np.unique(y)) > 2:
            y = Utilities.one_hot_encode(y)

        self.m, self.n = x.shape
        self.cost_history = []
        self.c = y.shape[1]
        self.w = np.zeros((self.n, self.c))  # W-->(n,1)
        self.b = np.zeros((1, self.c))+3

        batch_size = 30

        for i in range(1, epochs+1):
            zipped_data = np.hstack((x, y))
            np.random.shuffle(zipped_data)
            mini_batches = [zipped_data[k:k+batch_size]
                            for k in range(0, self.m, batch_size)]
            # Using Mini Batch gradient Descent.
            cost = 0
            avg_cost = 0
            for mini_batch in mini_batches:
                X = mini_batch[:, :self.n]
                Y = mini_batch[:, self.n:]

                dj_db, dj_dw = self.get_gradients(X, Y, lambda_)
                # Update values of w and b at each iteration
                self.w = self.w - alpha * dj_dw
                self.b = self.b - alpha * dj_db

                cost += self.logistic_cost(X, Y, lambda_)
            avg_cost += np.average(cost)
            avg_cost /= self.m

            self.cost_history.append(avg_cost)
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i == 1:
                # Code made for alignment of output prints.
                width = len(f'{self.cost_history[-1]:.3f}')
            if i % math.ceil(epochs / 10) == 0:
                print(
                    f'Epoch {i:3d} ⫸ Cost {self.cost_history[-1]:^{width}.3f} ⫷')

        self.cost_history = np.array(self.cost_history)

        if plot_cost:
            plt.plot(np.arange(epochs), self.cost_history)
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.title('Cost vs Epochs')
            plt.show()

        return self.w, self.b, self.cost_history


class KNearestNeighbourClassifier:
    def __init__(self, K):
        self.K = K

    def euclidiean_distance(self, X_test, X_train):
        # X_test-->(n,)1D Array X_train-->(m,n)
        d = np.sqrt(np.sum(np.square(X_train - X_test), axis=1))
        return d  # (m,) 1D Array

    def fit(self, X_train, Y_train):
        self.m, self.n = X_train.shape
        self.X_train = X_train
        self.Y_train = Y_train.reshape(self.m)  # (m,) 1D Array

    def search_neighbours(self, X):
        distances = self.euclidiean_distance(X, self.X_train)  # (m,) 1D Array
        indices = np.argsort(distances)
        Y_train_sorted = self.Y_train[indices]
        return Y_train_sorted[:self.K]

    def predict(self, X_test):
        # Get number of test examples and features
        m_test, _ = X_test.shape
        predictions = np.zeros(m_test)

        for i in range(m_test):
            X = X_test[i]
            # Store the K nearest neighbours. (m_test,)
            neighbours = self.search_neighbours(X)

            values, counts = np.unique(neighbours, return_counts=True)
            indices = np.argmax(counts)
            predictions[i] = values[indices]

        return predictions.reshape(-1, 1)


class KMeansClustering:
    def __init__(self, X, iterations, K):
        self.iterations = iterations
        self.K = K
        self.m, self.n = X.shape
        self.X = X
        self.fit()

    def initialize_centroids(self):
        # centroids-->(K,n) contains coordinated of K centroids
        self.new_centroids = np.zeros((self.K, self.n))
        for k in range(self.K):
            self.new_centroids[k] = self.X[np.random.choice(self.m)]

    def find_closest_centroid(self):
        # Create a matrix to save the distances of all examples from all the centroids.
        distances = np.zeros((self.m, self.K))
        for k, centroid in enumerate(self.new_centroids):
            distances[np.arange(self.m), k] = np.sqrt(
                np.sum((self.X - centroid)**2, axis=1))
        # At axis=1, choose the centroid nearest to the corresponding example.
        indices = np.argmin(distances, axis=1)
        return indices

    def move_centroid(self, ind):
        self.previous_centroids = self.new_centroids

        self.new_centroids = np.zeros((self.K, self.n))
        # For each centroid calculate the new mean point of the cluster.
        # If a centroid has no points assigned, then randomly reassign the centroid.
        for k in range(self.K):
            centroid_points = self.X[ind == k]
            if centroid_points.shape[0] == 0:
                self.new_centroids[k] = self.X[np.random.choice(self.m)]
            else:
                self.new_centroids[k] = np.mean(centroid_points, axis=0)

    def cost_function(self, ind):
        centroids = self.new_centroids[ind]
        distances = np.sqrt(np.sum((self.X - centroids)**2, axis=1))
        cost = np.mean(distances)
        return cost

    def fit(self):
        self.initialize_centroids()
        self.cost_history = []
        for i in range(1, self.iterations+1):
            indices = self.find_closest_centroid()
            self.cost_history.append(self.cost_function(indices))
            self.move_centroid(indices)
            print(
                f'Iteration: {i} | Mean Sum Distance: {self.cost_history[i-1]}')
        self.cost_history = np.array(self.cost_history)
        return self.cost_history

    def predict(self, x):
        # X should be a matrix element.
        predictions = np.empty(x.shape[0])
        for i, example in enumerate(x):
            distances = np.zeros(self.K)

            for k, centroid in enumerate(self.new_centroids):
                distances[k] = np.sqrt(np.sum((example - centroid)**2))
            predictions[i] = np.argmin(distances)

        return predictions

# ============================Neural Network========================


class NeuralNet():
    def __init__(self, layers):
        self.layers = layers
        self.num_inputs = layers[0].num_inputs
        layers.pop(0)
        self.layers = layers
        self.set_model_params()

    def set_model_params(self):
        prev_units = self.num_inputs

        for layer in self.layers:
            # For W, cols-->number of units in that layer
            # rows-->number of input from previous layer.
            # For b, length is equal to number of units in that layer.
            curr_units = layer.units
            W = np.random.randn(prev_units, curr_units)/np.sqrt(prev_units)
            b = np.random.randn(curr_units)
            layer.set_weights(W, b)
            prev_units = curr_units
            layer.initialize_gradients()

    def get_layers(self):
        return self.layers

    def feed_forward(self, X):
        a = X
        for layer in self.layers:
            z = np.matmul(a, layer.W) + layer.b
            a = layer.activation(z)
        return a

    def predict(self, X):
        a_out = self.feed_forward(X)
        return a_out

    def fit(self, x, y, cost_function, epochs=100, alpha=0.01, lambda_=0, batch_size=50, plot_cost=False):
        self.m, self.n = x.shape
        self.lambda_ = lambda_
        self.alpha = alpha
        cost_function = cost_function(self)
        # Check if number of output neurons is more than 1.
        if self.layers[-1].units != 1:
            y = Utilities.one_hot_encode(y)

        self.cost_history = []

        for epoch in range(1, epochs+1):
            zipped_data = np.hstack((x, y))
            np.random.shuffle(zipped_data)
            mini_batches = [zipped_data[k:k+batch_size]
                            for k in range(0, self.m, batch_size)]
            cost = 0
            for mini_batch in mini_batches:
                X = mini_batch[:, :self.n]
                Y = mini_batch[:, self.n:]

                for i in range(X.shape[0]):
                    a = X[i].reshape(1, -1)

                    for layer in self.layers:
                        a = layer.forward(a)
                    yhat = a
                    cost += cost_function.loss(yhat, Y[i])

                    cost_gradient = cost_function.gradient()
                    dC_dY = self.layers[-1].backward_output(cost_gradient)
                    for layer in self.layers[-2::-1]:
                        dC_dY = layer.backward(dC_dY)

                for layer in self.layers:
                    layer.update_weights(X.shape[0], self)
            cost /= self.m
            self.cost_history.append(cost)
            # Print cost every at intervals 10 times or as many iterations if < 10
            if epoch == 1:
                # Code made for alignment of output prints.
                width = len(f'{self.cost_history[-1]:.3f}')
            print(
                f'Epoch {epoch:3d} ⫸ Cost {self.cost_history[-1]:^{width}.5f} ⫷')

        if plot_cost:
            plt.plot(np.arange(epochs), self.cost_history)
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.title('Cost vs Epochs')
            plt.show()
        return self.cost_history


class Inputs():
    def __init__(self, inputs):
        self.num_inputs = inputs


class Layer():
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def initialize_gradients(self):
        self.dC_dW = np.zeros_like(self.W)
        self.dC_dB = np.zeros_like(self.b)

    def get_weights(self):
        return self.W, self.b

    def forward(self, x):
        self.x = x
        z = np.matmul(self.x, self.W) + self.b
        self.a = self.activation(z)
        return self.a

    def backward(self, dC_dY):
        a_prime = self.activation(self.a, derivative=True).reshape(-1, 1)
        dC_dZ = np.multiply(dC_dY, a_prime)
        self.dC_dW += np.matmul(self.x.reshape(-1, 1), dC_dZ.T)
        self.dC_dB += dC_dZ.reshape(self.units)
        dC_dX = np.matmul(self.W, dC_dZ)
        return dC_dX

    def backward_output(self, dC_dZ):
        self.dC_dW += np.matmul(self.x.reshape(-1, 1), dC_dZ.T)
        self.dC_dB += dC_dZ.reshape(self.units)
        dC_dX = np.matmul(self.W, dC_dZ)
        return dC_dX

    def update_weights(self, mini_batch_length, NeuralNet):
        if NeuralNet.lambda_ == 0:
            self.W = self.W - NeuralNet.alpha * \
                (self.dC_dW / mini_batch_length)
        else:
            self.W = ((1 - (NeuralNet.alpha * NeuralNet.lambda_) / NeuralNet.m)
                      * self.W)-(NeuralNet.alpha * (self.dC_dW / mini_batch_length))
        self.b = self.b - NeuralNet.alpha * (self.dC_dB / mini_batch_length)
        self.initialize_gradients()


class MeanSquaredLoss():
    def __init__(self, NeuralNet):
        self.NeuralNet = NeuralNet

    def loss(self, yhat, y):
        self.yhat = yhat.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        reg_term = 0
        if self.NeuralNet.lambda_ != 0:
            for layer in self.NeuralNet.layers:
                reg_term += np.sum(np.square(layer.W)) * self.NeuralNet.lambda_
                reg_term /= 2

        loss = np.sum((yhat-y)**2) / 2
        total_loss = loss + reg_term
        return total_loss

    def gradient(self):
        return self.yhat - self.y


class BinaryCrossEntropy():
    def __init__(self, NeuralNet):
        self.NeuralNet = NeuralNet

    def loss(self, yhat, y):
        self.yhat = yhat.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        reg_term = 0
        if self.NeuralNet.lambda_ != 0:
            for layer in self.NeuralNet.layers:
                reg_term += np.sum(np.square(layer.W)) * self.NeuralNet.lambda_

        loss = np.sum(-self.y * np.log(self.yhat + 0.000001) -
                      (1-self.y) * np.log(1 - self.yhat + 0.000001))
        total_loss = loss + reg_term
        return total_loss

    def gradient(self):
        return self.yhat - self.y


class CategoricalCrossEntropy():
    def __init__(self, NeuralNet):
        self.NeuralNet = NeuralNet

    def loss(self, yhat, y):
        self.yhat = yhat.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        reg_term = 0
        if self.NeuralNet.lambda_ != 0:
            for layer in self.NeuralNet.layers:
                reg_term += np.sum(np.square(layer.W)) * self.NeuralNet.lambda_

        loss = np.sum(-y * np.log(yhat + 0.000001))
        total_loss = loss + reg_term
        return total_loss

    def gradient(self):
        return self.yhat - self.y


class Activations():
    def __init__(self) -> None:
        pass

    @staticmethod
    def linear(z, derivative=False):
        if derivative:
            return 1
        else:
            return z

    @staticmethod
    def relu(z, derivative=False):
        if derivative:
            return np.where(z > 0, 1, 0)
        else:
            a = np.where(z > 0, z, 0)
            return a

    @staticmethod
    def sigmoid(z, derivative=False):
        if derivative:
            return Activations.sigmoid(z) * (1 - Activations.sigmoid(z))
        else:
            a = 1/(1 + np.exp(-z))
            return a

    @staticmethod
    def softmax(z, derivative=False):
        if derivative:
            raise Exception("Derivative of softmax is not defined.")
        else:
            a = np.exp(z) / np.sum(np.exp(z), axis=1).reshape((-1, 1))
            return a

# =============================UTILITIES=========================================


class Utilities:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_data(path, start_x=None, end_x=None, start_y=None, end_y=None):
        df = pd.read_csv(path)
        x_train = df.iloc[:, start_x:end_x]
        x_train = x_train.to_numpy()

        # If nothing about y has been mentioned, then return x_train, otherwise return both.
        if not (start_y == None and end_y == None):
            y_train = df.iloc[:, start_y:end_y]
            y_train = y_train.to_numpy()
            return x_train, y_train

        return x_train

    @staticmethod
    def split_data(percent, x_data, y_data=np.empty(0)):
        # Taking percentage of elements to split the data
        index_x = int(len(x_data) * (percent/100))
        # Assigning first set
        x_1 = x_data[:index_x]
        # Assgning second set
        x_2 = x_data[index_x:]

        if y_data.size != 0:
            index_y = int(len(y_data) * (percent/100))
            y_1 = y_data[:index_y]
            y_2 = y_data[index_y:]
            return x_1, y_1, x_2, y_2
        else:
            return x_1, x_2

    @staticmethod
    def one_hot_encode(Y):
        Y = Y.reshape(-1)
        Y_encd = np.zeros((Y.size, len(np.unique(Y))), dtype=int)
        Y_encd[np.arange(Y.size), Y] = 1
        return Y_encd

    @staticmethod
    def sigmoid(z):
        sgm = 1/(1 + np.exp(-z))
        return sgm

# ====================FEATURE SCALING=============================================


class StandardizationScale:
    def __init__(self) -> None:
        pass

    def fit_transform(self, X):
        # Calculating mean and SD along each column(axis 0)
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.sigma != 0, (X - self.mu) / self.sigma, 0)
        return X_result

    def transform(self, X):
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.sigma != 0, (X - self.mu) / self.sigma, 0)
        return X_result


class NormalizationScale:
    def __init__(self) -> None:
        pass

    def fit_transform(self, X):
        # Calculating min value, max vlaue, mean, along each column(axis 0)
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        self.mean = np.mean(X, axis=0)
        range = (self.max - self.min)
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(range != 0, (X - self.mean)/range, 0)
        return X_result

    def transform(self, X):
        range = (self.max - self.min)
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(range != 0, (X - self.mean)/range, 0)
        return X_result


class RobustScale:
    def __init__(self) -> None:
        pass

    def fit_transform(self, X):
        # Calculating median and iqr i.e difference b/w 25th and 75th percentile.
        self.median = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25])
        self.iqr = q75 - q25
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.iqr != 0, (X - self.median)/self.iqr, 0)
        return X_result

    def transform(self, X):
        with np.errstate(divide='ignore', invalid="ignore"):
            X_result = np.where(self.iqr != 0, (X - self.median)/self.iqr, 0)
        return X_result

# ================================METRICS=============================================


class Metrics():
    def __init__(self) -> None:
        pass
    # yhat and y should be (m,1) matrices where m can be number of test examples.

    @staticmethod
    def classifier_accuracy(yhat, y):
        acc = np.mean(yhat == y) * 100
        return acc

    @staticmethod
    def mean_square_error(yhat, y):
        mse = np.mean((yhat - y)**2)
        return mse/2

    @staticmethod
    def confusion_matrix(yhat, y, metrics=False):
        y = y.reshape(-1)
        yhat = yhat.reshape(-1)
        K = len(np.unique(y))  # Number of classes
        confusion = np.zeros((K, K))

        for i in range(len(y)):
            confusion[y[i], yhat[i]] += 1

        confusion = confusion.astype('int')
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(confusion, cmap="summer")
        for i in range(K):
            for j in range(K):
                text = ax.text(j, i, confusion[i, j],
                               ha="center", va="center", color="black")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.xlabel('Actual')
        plt.xticks(np.arange(K))
        plt.yticks(np.arange(K))
        plt.ylabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

        precision = 0
        recall = 0
        for cls in range(K):
            precision += confusion[cls, cls] / np.sum(confusion[cls])
            recall += confusion[cls, cls] / np.sum(confusion[:, cls])
        precision /= K
        recall /= K

        if metrics:
            return confusion, precision, recall
        else:
            return confusion

    @staticmethod
    def r2_score(yhat, y):
        ybar = np.average(y)
        ssr = np.sum((y-yhat)**2, dtype=np.float64)
        sst = np.sum((y - ybar)**2, dtype=np.float64)

        r2_score = 1 - (ssr/sst)
        return r2_score
