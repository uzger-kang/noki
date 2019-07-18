import numpy
import matplotlib.pyplot
import scipy.special
import imageio


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        #self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        #self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.onodes, self.hnodes))
        self.activation_function = lambda x:scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        return
    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        targets = numpy.array(target_list, ndmin=2).T
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), (numpy.transpose(hidden_outputs)))
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)
        return
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def backquery(self, targets_list):
        final_outputs = numpy.array(targets_list, ndmin=2).T
        final_inputs = self.inverse_activation_function(final_outputs)
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs


inputnodes=784
outputnodes=10
hiddennodes=100
learning_rate=0.1
n=neuralNetwork(inputnodes, hiddennodes, outputnodes, learning_rate)

with open( 'C:\\my_ml\\writed\\Write\\Write\\mnist_train.csv') as training_data_file:
    training_data_list = training_data_file.readlines()
    
for record in training_data_list:
    all_values = record.split(',')
    inputs = numpy.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
    targets = numpy.zeros(outputnodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)           # 按照行循环训练
              
with open('C:\\my_ml\\writed\\Write\\Write\\mnist_test.csv') as test_data_file:
    test_data_list = test_data_file.readlines()

# Single line
all_values = test_data_list[70].split(',')
inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01

print(n.query(inputs))

image_array = inputs.reshape(28,28)
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

'''
#Statisc
score = 0
for record in test_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    out_puts = n.query(inputs)
    label = numpy.argmax(out_puts)
    #print 'Result:' + str(label) + '\t' + 'Answer:' + str(all_values[0])
    if str(label)  == all_values[0]:
        score = score + 1
print score
print len(test_data_list)
print float(score)/len(test_data_list)
'''
'''
#query image
img_array = imageio.imread('C:\\Python27\\machinelearninginaction\\write\\self2.png', as_gray = True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data/255.0 * 0.99) + 0.01
out_puts = n.query( img_data)
print numpy.argmax(out_puts)
'''
'''
#back query
label = 0
targets = numpy.zeros(outputnodes) + 0.01
targets[label] = 0.99
print targets
image_data = n.backquery(targets)
matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

'''
