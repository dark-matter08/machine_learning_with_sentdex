# euclidean distance
# sqrt of the sum to n starting at i = 1 of ((Ai - Pi) ** 2)
# example two poins 
# a = (1,3)
# p = (2,5)

# euc_dist = sqrt((1 - 2)**2 + (3 - 5)**2)
#============ in code ==============

from math import sqrt

plot1 = [1, 3]
plot2 = [2, 5]

euclidean_distance = sqrt( (plot1[0] - plot2[0])**2 + (plot1[1] - plot2[1])**2 )

print("Distance: ", euclidean_distance)