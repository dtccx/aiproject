import numpy as np

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k
		self.train_X = []
		self.train_Y = []

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		# store X of train and test
		self.train_X=X
		self.train_Y=y

	#input is X_test, testdatat's X
	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		res=[]
		for i in range(len(X)):
			neighbors=self.getneighbors(X[i])
			result=self.majorityvote(neighbors, self.train_Y)
			res.append(result)
		res=np.asarray(res)
		return res

	def getneighbors(self, X):
		distance=[self.distance(traindata, X) for traindata in self.train_X]
		distance=np.asarray(distance)
		neighbors=np.argsort(distance)[:self.k]
		return neighbors

	def majorityvote(self, neighbors, X):
		count=0 # count label----1
		for index in neighbors:
			if X[index] == 1:
				count+=1
		if count > self.k - count:
			return 1
		elif count == self.k - count:
			return np.random.randint(2)
		else:
			return 0


class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		self.train_X=X
		self.train_Y=y
		categorical_data = self.preprocess(X)
		row_num, feature_num = X.shape
		for i in range(feature_num):
			type = np.unique(categorical_data[:,i])
			entropy_temp = 0
			# for j in range(len(type)):


		print(feature_num)




	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)

		return None

class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		# preround=0
		for i in range(steps):
			index=i%y.size
			# round=i/y.size
			# if round == preround:
			# 	# np.random.shuffle(X)
			# 	preround=round
			# print(X[index].shape)
			# print(self.w.shape)
			predict_y=np.matmul(X[index], self.w)+self.b
			# print(predict_y)
			if predict_y > 0:
				predict_y=1
			else:
				predict_y=0
			if predict_y != y[index]:
				# we change y[index] from 0 to -1
				y_change = 1
				if y[index] == 0:
					y_change = -1
				self.w=self.w+X[index]*y_change*self.lr
				self.b=self.b+y_change*self.lr

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		result=[]
		for data in X:
			res=np.matmul(data, self.w)+self.b
			if res > 0:
				res = 1
			else:
				res = 0
			result.append(res)
		result=np.asarray(result)
		return result

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b
		self.x = 0

	def forward(self, input):
		#Write forward pass here
		# self.y = 1/(1+np.exp(-input))
		self.x = input
		return input.dot(self.w)+self.b

	def backward(self, gradients):
		#Write backward pass here
		w_diff = self.x.T.dot(gradients)
		x_diff = gradients.dot(self.w.T)
		self.w -= self.lr * w_diff
		self.b -= self.lr * gradients
		return x_diff

class Sigmoid:

	def __init__(self):
		self.y = 0

	def forward(self, input):
		#Write forward pass here
		self.y = 1/(1+np.exp(-input))
		return 1/(1+np.exp(-input))

	def backward(self, gradients):
		#Write backward pass here
		return gradients*(1-self.y)*self.y
