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

f = open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

onearray = np.ones((test.shape[0],1))
test = np.append(test,onearray,axis=1)

rows = train.shape[0]
cols = train.shape[1]

#hidden_nodes = int(sys.argv[3])
mb_size = int(sys.argv[3])

hidden_nodes = 3

##############################
### Initialize all weights ###

w = np.random.rand(hidden_nodes)
#w = np.ones(hidden_nodes)
print("w=",w)

#check this command
#W = np.zeros((hidden_nodes, cols), dtype=float)
#W = np.ones((hidden_nodes, cols), dtype=float)
W = np.random.rand(hidden_nodes, cols)
print("W=",W)

epochs = 100
eta = .001
prevobj = np.inf
i=0

###########################
### Calculate objective ###

hidden_layer = np.matmul(train, np.transpose(W))
#print("hidden_layer=",hidden_layer)
#print("hidden_layer shape=",hidden_layer.shape)

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#print("hidden_layer=",hidden_layer)
#print("hidden_layer shape=",hidden_layer.shape)

output_layer = np.matmul(hidden_layer, np.transpose(w))
#output_layer = np.array([sigmoid(xi) for xi in output_layer])
#print("output_layer=",output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
#print("obj=",obj)

#obj = np.sum(np.square(np.matmul(train, np.transpose(w)) - trainlabels))
#print("Obj=",obj)

###############################
### Begin gradient descent ####

stop=0.00001
rowindices = np.array([i for i in range(rows)])

while(i < epochs):
#while(prevobj - obj > stop and i < epochs):
#while(prevobj - obj > 0):

	#Update previous objective
	prevobj = obj

	#Calculate gradient update for final layer (w)
	#dellw is the same dimension as w

#	print(hidden_layer[0,:].shape, w.shape)

	for k in range(0, rows, 1):
#	for k in range(0, int(rows/10), 1):

		#Create mini-batch: randomly select k training points
		#Calculate gradients below using just the k points
		np.random.shuffle(rowindices)	
		
		index = rowindices[0]
		dellw = (np.dot(hidden_layer[index,:],np.transpose(w))-trainlabels[index])*hidden_layer[index,:]
		for j in range(1, mb_size, 1):		
			index = rowindices[j]
			dellw += (np.dot(hidden_layer[index,:],np.transpose(w))-trainlabels[index])*hidden_layer[index,:]

		#Update w
		w = w - eta*dellw
#		print("w=",w)
#		print("dellf=",dellf)
	
		#Calculate gradient update for hidden layer weights (W)
		#dellW has to be of same dimension as W
		#Let's first calculate dells. After that we do dellu and dellv.
		#Here s, u, and v are the three hidden nodes
		#dells = df/dz1 * (dz1/ds1, dz1,ds2)
	
		index = rowindices[0]
		dells = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[0] * (hidden_layer[index,0])*(1-hidden_layer[index,0])*train[index]
		for j in range(1, mb_size):
			index = rowindices[j]
			dells += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[0] * (hidden_layer[index,0])*(1-hidden_layer[index,0])*train[index]

		index = rowindices[0]
		dellu = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[1] * (hidden_layer[index,1])*(1-hidden_layer[index,1])*train[index]
		for j in range(1, mb_size):
			index = rowindices[j]
			dellu += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[1] * (hidden_layer[index,1])*(1-hidden_layer[index,1])*train[index]

		index = rowindices[0]
		dellv = np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[2] * (hidden_layer[index,2])*(1-hidden_layer[index,2])*train[index]
		for j in range(1, mb_size):
			index = rowindices[j]
			dellv += np.sum(np.dot(hidden_layer[index,:],w)-trainlabels[index])*w[2] * (hidden_layer[index,2])*(1-hidden_layer[index,2])*train[index]

		dellW = np.array(dells, ndmin=2)
		dellW = np.append(dellW, np.array(dellu,ndmin=2), axis=0)
		dellW = np.append(dellW, np.array(dellv,ndmin=2), axis=0)
#		print(dellW)

		#Update W
		W = W - eta*dellW

	#Recalculate objective
	hidden_layer = np.matmul(train, np.transpose(W))
#	print("hidden_layer=",hidden_layer)

	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#	print("hidden_layer=",hidden_layer)

	output_layer = np.matmul(hidden_layer, np.transpose(w))
#	print("output_layer=",output_layer)

	obj = np.sum(np.square(output_layer - trainlabels))
#	print("obj=",obj)
	
	i = i + 1
	print("Objective=",obj)

### Do final predictions ###

#Recalculate objective
hidden_layer = np.matmul(test, np.transpose(W))
#print("hidden_layer=",hidden_layer)

hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#print("hidden_layer=",hidden_layer)

output_layer = np.matmul(hidden_layer, np.transpose(w))
#print("output_layer=",output_layer)
print(np.sign(output_layer))

prediction = np.sign(output_layer)
error = 1 - np.mean(prediction == testlabels)
print("Error=",error)
