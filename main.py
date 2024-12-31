import numpy as np

# constants
matrixA = [[4, 2, 0], [2, 10, 4], [0, 4, 5]]
vectorB = [2, 6, 5]


# helper functions
def check_diagonally_dominant(matrix):
    is_diagonally_dominant = True
    size = len(matrix)
    for i in range(size):
        if abs(matrix[i][i]) <= sum(abs(matrix[i][j]) for j in range(size) if j != i):
            is_diagonally_dominant = False
    if is_diagonally_dominant:
        print("Matrix is diagonally dominant")
        return
    print("Matrix is not diagonally dominant")


# yaakobi method
def yaakobi(matrix, vector_b, epsilon=0.00001, max_iter=20):
    check_diagonally_dominant(matrix)
    matrix = np.array(matrix)
    vector_b = np.array(vector_b)
    size = len(matrix)
    x_vector = np.zeros(size)  # initial guess
    d_inv_matrix = np.diag(1 / np.diag(matrix))
    l_matrix = np.tril(matrix, k=-1)
    u_matrix = np.triu(matrix, k=1)
    l_plus_u_matrix = l_matrix + u_matrix

    for _ in range(max_iter):
        print(_, x_vector)  # DEBUG
        x_new_vector = np.dot(d_inv_matrix, vector_b - np.dot(l_plus_u_matrix, x_vector))
        if np.linalg.norm(x_vector - x_new_vector) < epsilon:
            return x_new_vector
        x_vector = x_new_vector
    print("Maximum iterations reached. Result may not be accurate.")
    return x_vector


# gaus-seidel method
def gaus(matrix, vector_b, epsilon=0.00001, max_iter=20):
    check_diagonally_dominant(matrix)
    size = len(matrix)
    matrix = np.array(matrix)
    vector_b = np.array(vector_b)
    x_vector = np.zeros(size)  # initial guess

    u_matrix = np.triu(matrix, k=1)
    l_plus_d_inv_matrix = np.linalg.inv(np.tril(matrix))

    for _ in range(max_iter):
        print(_, x_vector) # DEBUG
        u_x_vector = np.dot(u_matrix, x_vector)
        x_new = -np.dot(l_plus_d_inv_matrix, u_x_vector) + np.dot(l_plus_d_inv_matrix, vector_b)
        if np.linalg.norm(x_vector - x_new) < epsilon:
            return x_new
        x_vector = x_new
    print("Maximum iterations reached. Result may not be accurate.")
    return x_vector


# main function
def main():
    method = input("Enter the method you want to use (yaakobi/gaus): ")
    if method == "yaakobi":
        result = yaakobi(matrixA, vectorB)
    elif method == "gaus":
        result = gaus(matrixA, vectorB)
    else:
        print("Invalid method")
        return
    print("Solution:", result)


# entrypoint
if __name__ == '__main__':
    main()
