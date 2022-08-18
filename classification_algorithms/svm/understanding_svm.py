# magnitude of a vector
# A = [3, 4]
# || A|| = sqrt(A[0]**2 + A[1]**2)

# dot product
# A = [1, 3]
# B = [4, 1]

# A . B = (A[0] * B[0]) + (A[1] * B[1])

# dot product and magnitude are essentioal to the building and operation of an svm


# when an unknown point is found by the svm
# U is the unknown point
# W is ???????????
# calc = U . W + b(bias) 

# if calc > 0 it falls on the oposite side of where W is found
# if calc = 0 the U falls on the boundary line


# ========================================= Known formulae ==========================================
# X(-sv) . W + b = -1
# X(+sv) . W + b = 1

# Yi is class of the features

# if Yi is of + class then Yi = +1
# if Yi is of - class then Yi = -1

# formula to get class support vector

# + class = Xi . W + b = 1 => multiply by Yi => Yi (Xi . W + b) = 1
# Yi = 1

# - class = Xi . W + b = -1 => multiply by Yi => Yi (Xi . W + b) = 1
# Yi = -1

# subtract 1 from both sides on equation for both classes

# + class : Yi (Xi . W + b) - 1 = 0
# - class : Yi (Xi . W + b) - 1 = 0


# width = (X+ - X-) . (W/ ||W||)
# width = ( Yi (Xi . W + b) - 1 - Yi (Xi . W + b) - 1) ) . (W / ||W||)
# width = 2 / ||W||

# we wanna minimize 1/2(||W|| **2)
# applying lagrange
#  L(W, b) = 1/2(||W|| **2) - sum(ALPi(Yi (Xi . W + b) - 1))

# we wanna minimize W and maximize b
# Equation for a hyperplane
# (W . X) + b

# differentiate L wrt to W : W - sum(ALPi(Yi.Xi))
# differentiate L wrt to b : -sum(ALPi.Yi) = 0

# L = sum to i of (ALPi) - 1/2(sum to i & j of(ALPi . ALPj . Yi . Yj. (Xi . Xj))) 

# ============================================ clearing things ================================================================

# Equation for hyperplane X . W + b
# hyperplane for + class = Xi . W + b = 1 (support vectors)
# hyperplane for - class = Xi . W + b = -1 (support vectors)

# ================== The support vector is that member of the class that is closes to the hyperplane of the svm ==================
#  The decision boundy hyperplane value X . W + b = 0

# ========================================= optimizing for W and b ==============================================================
# The optimization objective  is 
# -> minimize ||W||
# -> maximize b

# The constraints
# Yi(Xi . W + b) >= 1
# class(known_featues . W + b) >= 1
# finding the lowest vector of W and the largest b

# convex problem
# vector W is a [1 x 2] matrix

# dot product of 2 vectors gives a scalar

# optimization is a major field in machine learning
# The Support vector machine is a convex optimization problem

# Libraries for optimization
# => cvxopt
# => libsvm