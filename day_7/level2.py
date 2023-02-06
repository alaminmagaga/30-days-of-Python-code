
A = {19, 22, 24, 20, 25, 26}
B = {19, 22, 20, 25, 26, 24, 28, 27}


# Join A and B
C = A.union(B)
print(C)

# Find A intersection B
D = A.intersection(B)
print(D)

# Is A subset of B
print(A.issubset(B))

# Are A and B disjoint sets
print(A.isdisjoint(B))

# Join A with B and B with A
E = A.union(B)
F = B.union(A)
print(E)
print(F)

# What is the symmetric difference between A and B
G = A.symmetric_difference(B)
print(G)

# Delete the sets completely
del it_companies
del A
del B
del C
del D
del E
del F
del G
