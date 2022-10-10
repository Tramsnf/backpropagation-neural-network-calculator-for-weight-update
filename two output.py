# iu1 represent the input training data to input unit 1
# iu2 represent the input training data to input unit 2
# w11 represent the weight between input unit 1 and hidden unit 1
# w12 represent the weight between input unit 1 and hidden unit 2
# w21 represent the weight between input unit 2 and hidden unit 1
# w22 represent the weight between input unit 2 and hidden unit 2
#
# v11 represent the weight between hidden unit 1 and output unit 1
# v12 represent the weight between hidden unit 1 and output unit 2
# v13 represent the weight between hidden unit 1 and output unit 3
# v21 represent the weight between hidden unit 2 and output unit 1
# v22 represent the weight between hidden unit 2 and output unit 2
# v23 represent the weight between hidden unit 2 and output unit 3

import numpy as np
import math
def sigmoid(z):
    round(math.pi, 4)
    return 1./(1. + np.exp(-z))
    

lc = 0.9
iu1 = 0 
iu2 = 1
w11 = 0.2 
w12 = -0.2
w21 = 0.3
w22 = -0.4
v11 = 0.5
v12 = 0.3
v21 = 0.4
v22 = 0.6
##
#
#equation 
h1_in = w11*iu1 + w21*iu2
h2_in = w12*iu1 + w22*iu2
h1_out, h2_out = sigmoid(h1_in), sigmoid(h2_in)

y1_in = v11*h1_out + v21*h2_out
y2_in = v12*h1_out + v22*h2_out
y1_out, y2_out = sigmoid(y1_in), sigmoid(y2_in)

##targeted output are !!
t1, t2 = 0.1, 0.9
#----------------------
e1, e2 = (t1-y1_out), (t2-y2_out)
#print(e1, e2, e3)

grad_y1_out = 1*e1
grad_y2_out = 1*e2
#print(grad_y1_out, grad_y2_out, grad_y3_out)
##
##
#to calculate the new weight 
#nw_1 =
##
# backprop through sigmoid, simply multiply by sigmoid(z) * (1-sigmoid(z))
grad_y1_in = (y1_out * (1-y1_out)) * grad_y1_out
grad_y2_in = (y2_out * (1-y2_out)) * grad_y2_out


grad_v21 = grad_y1_in * h2_out
grad_v22 = grad_y2_in * h2_out 
#print(grad_v21, grad_v22, grad_v23)

grad_v11 = grad_y1_in * h1_out
grad_v12 = grad_y2_in * h1_out 

#print(grad_v11, grad_v12, grad_v13)

grad_h1_out = grad_y1_in*v11 + grad_y2_in*v12 
grad_h2_out = grad_y1_in*v21 + grad_y2_in*v22 
#print(grad_h1_out, grad_h2_out)

# backprop through sigmoid, simply multiply by sigmoid(z) * (1-sigmoid(z))
grad_h1_in = (h1_out * (1-h1_out)) * grad_h1_out
grad_h2_in = (h2_out * (1-h2_out)) * grad_h2_out

#xx1 = v13+lc*grad_y3_in*h1_in
#xx2 = grad_h1_in*w21 + grad_h2_in*w22

dl = "======================================================================="

print(" ")
print(" ")
print(dl)
print("learning coeffienct =", lc)
print("input training data to input unit 1 =", iu1)
print("input training data to input unit 2 =", iu2)
print("weight between input unit 1 and hidden unit 1 =", w11)
print("weight between input unit 1 and hidden unit 2 =", w12)
print("weight between input unit 2 and hidden unit 1 =", w21)
print("weight between input unit 2 and hidden unit 2 =", w22)
print("the weight between hidden unit 1 and output unit 1 =", v11)
print("the weight between hidden unit 1 and output unit 2 =", v12)

print("the weight between hidden unit 2 and output unit 1 =", v21)
print("the weight between hidden unit 2 and output unit 2 =", v22)

print(dl)
print(" ")

print('Solution')
print(dl)
print("Net hidden unit 1 = ",h1_in)
print("Actual output of the hidden unit 1 = ",h1_out)
print(dl)
print("Net hidden unit 2 = ",h2_in)
print("Actual output of the hidden unit 2 = ",h2_out)
print(dl)

# next section (from the hidden side to the output side) 
print(dl)
#

print("Net output unit 1 = ", y1_in)
print("Actual output of output unit 1 = ",y1_out)
print(dl)
print("Net output unit 2 = ",y2_in)
print("Actual output of output unit 2 = ", y2_out)
print(dl)

print(dl)

print(dl)
print("Delta of output unit 1 = ", grad_y1_in)
print("Delta of output unit 2 = ", grad_y2_in)

print(dl)
print("")

print(dl)
print("Delta of hidden unit 1 = ", grad_h1_in)
print("Delta of hidden unit 2 = ", grad_h2_in)
print(dl)

#only work on the line below if you want to get the new weight
# get the gradients for the inputs (could be ignored in this case)


print(dl)
print(dl)
#print("New weight", xx1)
print(dl)
print(dl)

# get the gradients for the weights
#grad_w21 = grad_h1_in * iu2
#grad_w22 = grad_h2_in * iu2 
#print(grad_w21, grad_w22)

#grad_w11 = grad_h1_in * iu1
#grad_w12 = grad_h2_in * iu1 
#print(grad_w11, grad_w12)

# get the gradients for the inputs (could be ignored in this case)
#grad_iu1 = grad_h1_in*w11 + grad_h2_in*w12
#grad_iu2 = grad_h1_in*w21 + grad_h2_in*w22
#print(grad_iu1, grad_iu2)