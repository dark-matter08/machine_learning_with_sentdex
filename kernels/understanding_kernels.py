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

k(X, X') = (1 + X.X' + ... + Xn.X'n)^p



"""