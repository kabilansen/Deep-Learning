import numpy as np
import sys

#################
### Read data ###

f = open(sys.argv[1])
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0],1))
train = np.append(train,onearray,axis=1)

#print("train=",train)
# print("train shape=",train.shape)

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

#print("test=", test)

rows = train.shape[0]
cols = train.shape[1]

#hidden_nodes = int(sys.argv[3])

hidden_nodes = 3

##############################
### Initialize all weights ###

#w = np.random.rand(1,hidden_nodes)
w = np.random.rand(hidden_nodes)
# print("w=",w)

#check this command
#W = np.zeros((hidden_nodes, cols), dtype=float)
# W = np.ones((hidden_nodes, cols), dtype=float)
W = np.random.rand(hidden_nodes, cols)
# print("W=",W)

epochs = 1000
eta = .001
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
# print("hidden_layer_dim=",np.shape(hidden_layer))
# print("hidden_layer shape=",hidden_layer.shape)
def relu(arr):
	for i in range(len(arr)):
		if(arr[i] > 0):
			arr[i] = arr[i]
		else:
			arr[i] = 0
	return arr
	

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
# print("hidden_layer=",hidden_layer)
# print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
# print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
# print("obj=",obj)

#obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))

# print("Obj=",obj)

###############################
### Begin gradient descent ####

while(prevobj - obj > 0.001 or i < epochs ):
#while(prevobj - obj > 0):

	#Update previous objective
	prevobj = obj

	#Calculate gradient update for final layer (w)
	#dellw is the same dimension as w

	# print(hidden_layer[0,:].shape, w.shape)

	dellw = (np.dot(hidden_layer[0,:],w)-trainlabels[0])*hidden_layer[0,:]
	for j in range(1, rows):
		dellw += (np.dot(hidden_layer[j,:],np.transpose(w))-trainlabels[j])*hidden_layer[j,:]

	#Update w
	w = w - eta*dellw

#	print("dellf=",dellf)
	
	#Calculate gradient update for hidden layer weights (W)
	#dellW has to be of same dimension as W

	#Let's first calculate dells. After that we do dellu and dellv.
	#Here s, u, and v are the three hidden nodes
	#dells = df/dz1 * (dz1/ds1, dz1,ds2)
	dells = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[0] * (hidden_layer[0,0])*(1-hidden_layer[0,0])*train[0]
	dellu = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[1] * (hidden_layer[0,1])*(1-hidden_layer[0,1])*train[0]
	dellv = np.sum(np.dot(hidden_layer[0,:],w)-trainlabels[0])*w[2] * (hidden_layer[0,2])*(1-hidden_layer[0,2])*train[0]
	for j in range(1, rows):
		dells += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[0] * (hidden_layer[j,0])*(1-hidden_layer[j,0])*train[j]
		dellu += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[1] * (hidden_layer[j,1])*(1-hidden_layer[j,1])*train[j]
		dellv += np.sum(np.dot(hidden_layer[j,:],w)-trainlabels[j])*w[2] * (hidden_layer[j,2])*(1-hidden_layer[j,2])*train[j]

	# exit()

	#TODO: dellu = ?

	#TODO: dellv = ?

	#TODO: Put dells, dellu, and dellv as rows of dellW
	dellW = np.array([dells, dellu, dellv])

	#Update W
	for k in range(3):
		W[k] = W[k] - eta*dellW[k]
	# print(W)

	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))
	# print("hidden_layer=",hidden_layer)

	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
	# print("hidden_layer=",hidden_layer)

	output_layer = (np.matmul(hidden_layer, np.transpose(w)))
	# print("output_layer=",output_layer)
	sum_w =0
	sum_fr=0
	sum_r=0
	for o in range(len(w)):
		sum_w += w[o]**2
		for p in range(len(W)):
			sum_fr += W[o][p]**2	
	sumr = sum_w + sum_fr

	obj = np.sum(np.square(output_layer - trainlabels))
	# print("obj=",obj)
	
	# if(prevobj - obj < 0.1):
	# 	eta = 0.1
	# if(prevobj - obj < 0.01):
	# 	eta = 0.01
	# if(prevobj - obj < 0.001):
	# 	eta = 0.001
	# if(prevobj - obj < 0.0001):
	# 	eta = 0.0001
	# elif(prevobj - obj < 0.00001):
	# 	eta = 0.00001
	# if(obj < 4.01):
	# 	eta = 0.001

	i = i + 1
	# print(i)
	# print("Objective=",obj)
	

# predictions = np.sign(np.dot([[11,11,1]], np.transpose(w)))
predict_hidden = sigmoid(np.matmul(test, np.transpose(W)))
predictions = np.sign(np.matmul(predict_hidden,np.transpose(w)))
print(predictions)
# print(W)
# if(predictions < .5):
# 	print(predictions, -1)
# else:
# 	print(predictions,1)

# print(np.shape(w))
# print(np.shape(W))
# print(w)
