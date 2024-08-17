import numpy as np


my_vector = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])

print("my_vector:\n")
print(my_vector)

## REASSIGNMENTS ##
# first_2_elmts = my_vector[0:2]
# first_2_elmts = np.array([[10],[20]]) # Guess: This is a reassignment and does not affect my_vector

# REFERENCES
# first_2_elmts = my_vector[0:2]
# first_2_elmts[0:2] = np.array([[10],[20]]) # This is not a reassignment but changes the value of my_vector

# print("first_2_elmts:\n")
# print(first_2_elmts)

# print("my_vector after change to first_2_elmts:\n")
# print(my_vector)

# first_1_elmt = my_vector[0]

# first_1_elmt = 13 # This is a reassignment and has nothing to do with my_vector anymore.

# print("first_1_elmts:\n")
# print(first_1_elmt)

# print("my_vector after change to first_1_elmt:\n")
# print(my_vector)

# my_number = 10
# my_vector[0][0] = my_number 

# my_number = 12

# print("my_vector after change to first_2_elmts:\n")
# print(my_vector)

## SINGLE ELEMENT POINTERS

# first_element = my_vector[0] # This is not a copy but a pointer
# first_element[0] = 10 # This changes the value of my_vector.

# print("my_vector after changes:\n")
# print(my_vector)


## 2-LEVEL POINTER
# first_2_elmts = my_vector[0:2]
# first_1_elmt = first_2_elmts[0]

# first_1_elmt[0] = 10 # This should affect both first_2_elmts and my_vector

# print("first_2_elmts:\n")
# print(first_2_elmts)

# print("my_vector after changes:\n")
# print(my_vector)

# POINTER ARRAY REASSIGNMENT
# first_2_elmts = my_vector[0:2]

# first_2_elmts[0:2] = np.array([[10],[20]]) # This is a not reassignment and does affect my_vector
# first_2_elmts = np.array([[10],[20]]) # This is a reassignment and does not affect my_vector


# print("my_vector after changes:\n")
# print(my_vector)


# THE [:] OPERATOR 

# The [:] operator can be used to adress all elements of an array
# my_vector = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
# my_pointer = my_vector[:] # This also creates a pointer rather than a copy
# my_pointer[0][0] = 10
# print("my_pointer after changes:\n")
# print(my_pointer)
# print("my_vector after changes:\n")
# print(my_vector)

# Similarly, the [:] operator behaves similarly for regular arrays
# my_vector = [[1], [2], [3], [4], [5], [6], [7], [8]] # This is a list of sublists.
# my_pointer = my_vector[:] # This creates a shallow copy, but the lists elements reference the same sublists.
# my_pointer[0][0] = 10
# print("my_pointer after changes:\n")
# print(my_pointer)
# print("my_vector after changes:\n")
# print(my_vector)

# Here another sublist is created, but the copy is appended with elements while the original list is not. the first elements remain identical and altering them in one array affects the other.
# my_vector = [[1, 2], [2, 4]]
# my_copy = my_vector[:] ## shallow copy
# my_copy.append([3, 6])

# print("my_copy after changes:\n")
# print(my_copy)
# print("my_vector after changes:\n")
# print(my_vector)