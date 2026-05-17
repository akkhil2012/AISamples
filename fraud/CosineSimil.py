import numpy as np

A = np.array([1, 2, 3])
B = np.array([2, 4, 6])

# cosine similarity
cosine = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# euclidean distance
euclidean = np.linalg.norm(A - B)

print(f"cosine:    {cosine:.4f}")     # 1.0000
print(f"euclidean: {euclidean:.4f}")  