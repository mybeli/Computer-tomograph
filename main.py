
import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not

    # TODO: Perform gaussian elimination
    z,s = np.shape(A)
    if(z != s) :
        raise ValueError
        
    n = len(b)
    if(z != n) :
        raise ValueError
    
    """Forward Elimination"""
    for j in range(0,s) :   # column
        for i in range(0,z) :   # row
            if(i == j) :
                if(use_pivoting == True) : # if pivoting used
                    maxIndex = np.argmax(abs(A[i:,j])) + i # find index of the max value
                    if(abs(A[maxIndex,j]) <= 1e-17) :   # if biggest value is zero
                        raise ValueError
                    if(maxIndex != i) : # if biggest value in another row, then swap 
                        temp = A[maxIndex,:].copy() # swap rows in A
                        A[maxIndex,:],A[i,:] = A[i,:],temp
                        tempV = b[maxIndex].copy() # swap rows in b
                        b[maxIndex],b[i] = b[i],tempV
                else : # if no pivoting used
                    if(abs(A[i,j]) <= 1e-15) : # rows must be swapped
                        swapIndex = i+1 # index of row to be swapped
                        while(swapIndex < z and abs(A[swapIndex,j]) <= 1e-15) :  # search viable row to be swapped
                            swapIndex = swapIndex + 1
                    
                        if(swapIndex >= z) :
                            raise ValueError
                        
                        temp = A[swapIndex,:].copy() # swap rows in A
                        A[swapIndex,:],A[i,:] = A[i,:],temp
                        tempV = b[swapIndex].copy() # swap rows in b
                        b[swapIndex],b[i] = b[i],tempV
                        
                pivot = A[i,j]
                p = 1   # p is a counter
                while(i+p <= z-1) : # turn all values below pivot to zero
                    div = A[i+p,j] / pivot
                    A[i+p,:] = A[i+p,:] - (A[i,:] * div)
                    if (abs(A[i+p,j]) <= 1e-15) : A[i+p,j] = 0  # für Schönheit
                    b[i+p] = b[i+p] - (b[i] * div)
                    if (abs(b[i+p]) <= 1e-15) : b[i+p] = 0
                    p = p + 1
        for i in range (z-1, 0, -1):
            if (A[i,i] == 0) :
                raise ValueError


    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    z,s = np.shape(A)
    if(z != s) :
        raise ValueError
        
    n = len(b)
    if(z != n) :
        raise ValueError
    # TODO: Initialize solution vector with proper size
    x = np.zeros(1)

    
    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
 
    for k in range(0,n): 
        if (A[k,k] == 0):
            raise ValueError
    
   

    for k in range(n-1,-1,-1): # start from bottom row
        b[k] = (b[k] - np.dot(A[k,k+1:s],b[k+1:n]))/A[k,k]
    
    
    return b

def forward_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()

    # Test if shape of matrix and vector is compatible
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b not compatible!")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A isn't square!")
    for i in range(0, A.shape[0]):
        for j in range(0, i):
            if not np.isclose(A[j, i], 0):
                raise ValueError("A isn't lower triangular!")

    # Initialize solution vector with proper size
    x = np.zeros(A.shape[1])

    # Run forwardsubstitution and fill solution vector
    for i in range(0, A.shape[0]):
        if np.isclose(A[i, i], 0):
            raise ValueError("Infinite solutions exist!")

        b[i] /= A[i, i]
        A[i, i] = 1
        x[i] = b[i]

        for j in range(i+1, A.shape[0]):
            b[j] -= A[j, i] * x[i]
            A[j, i] = 0

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if not np.allclose(M, np.transpose(M)):
        raise ValueError("Matrix is not symmetric!")

    # build the factorization and raise a ValueError in case of a non-positive definite input matrix
    L = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                sum = 0
                for k in range(0, i):
                    sum += L[i, k] ** 2
                if M[i, i]-sum <= 0:
                    raise ValueError("Matrix it not positive semi-definite!")
                L[i, j] = np.sqrt(M[i, i]-sum)
            elif i > j:
                sum = 0
                for k in range(0, j):
                    sum += L[i, k] * L[j, k]
                L[i, j] = (M[i, j] - sum) / L[j, j]

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    if n != m:
        raise ValueError("L isn't square!")
    for i in range(0, n):
        for j in range(0, i):
            if not np.isclose(L[j, i], 0):
                raise ValueError("L isn't lower triangular!")
    if n != b.shape[0]:
        raise ValueError("L and x are not compatible!")

    y = forward_substitution(L, b)
    x = back_substitution(np.transpose(L), y)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different directions
    n_rays   : number of parallel rays
    n_grid   : number of cells of grid in each direction

    Return:
    A : system matrix
    v : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # Initialize system matrix with proper size
    A = np.zeros((n_shots*n_rays, n_grid*n_grid))
    # Initialize sinogram for measurements for each shot in one column
    S = np.zeros((n_rays, n_shots))

    # Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    for i in range(0, n_shots):
        theta = np.pi * (i / n_shots)
        # Take a measurement from direction theta. Return values are
        # ints : measured intensities for parallel rays (ndarray)
        # idx  : indices of rays (ndarray)
        # idx_isect : indices of intersected cells (ndarray)
        # dt : lengths of segments in intersected cells (ndarray)
        ints, idxs, idxs_isects, dt = tomograph.take_measurement(n_grid, n_rays, theta)

        for j in range(0, n_rays):
            S[j, i] = ints[j]

        for j in range(0, len(dt)):
            A[i * n_rays + idxs[j], idxs_isects[j]] = dt[j]

    # Convert per shot measurements in sinogram to a 1D np.array so that columns
    # in the sinogram become consecutive elements in the array
    v = np.zeros(n_shots * n_rays)
    for i in range(0, n_shots):
        for j in range(0, n_rays):
            v[i * n_rays + j] = S[j, i]

    return [A, v]


def compute_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of shots from different directions
    n_rays   : number of parallel rays
    r_theta  : number of cells in the grid in each direction

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, v] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)
    densities = np.linalg.solve(np.dot(np.transpose(L), L), np.dot(np.transpose(L), v))

    # Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))
    for i in range(0, n_grid):
        for j in range(0, n_grid):
            tim[i, j] = densities[i * n_grid + j]

    return tim


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
