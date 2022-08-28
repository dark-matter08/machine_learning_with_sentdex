""" 
X' - X prime
K(X, X') = Z . Z'
Z is a fxn applied to it's x counterpart
Z = f(X)
Z' = f(X')
K = Z . Z'
the fxn defination has to be desame
Kernel usualy denoted as ø (phi)


consider feature set = [X1, X2]
to take ourselves out to some sort of Z - space convert to a 2nd order polynomial

X = [X1, X2]
Z = [1, X1, X2, X1.X1, X2.X2, X1.X2]
Z' = [1, X'1, X'2, X'1.X'1, X'2.X'2, X'1.X'2]

K(X, X') = Z . Z'

=> [1 + X1.X'1 + X2.X'2 + (X1**2 . X'1**2) + (X2**2 . X'2**2) + X1.X'1.X2.X'2]

========== So can we get the kernel without having to visit the kernel space ===========

K(X, X') = (1 + X.X')^p ========================== the above example with the z space assumes p=2 and n=2

Kn(X, X') = (1 + X.X' + ... + Xn.X'n)^p

where n is the number of dimensions and p i sthe polynomial order

RBF = exp(x) = e^x


======================== introducing the soft margin support vector machine ==============================

in a real dataset if more than 10% of the data is a support vector, then you have done an over fitment => try a different kernel

Slack{E}
introducitn slack in svm
E >= 0

total slack = sum to i of Ei

Yi(Xi.W + b) >= 1 - E

nominaly, we want to min ||W||
minimize 1/2(||W||^2) + C[sum to i(Ei)]

The smaller C is the less the slack matters




"""