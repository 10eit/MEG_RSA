import numpy as np
import scipy
from scipy.stats import spearmanr, kendalltau
from scipy.spatial import procrustes
from scipy.stats import wasserstein_distance

"""
This File intended for calculating similarity
between different representation dissimilarity matrices.
"""

### Utilis Function Starts Here

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

### Similarity Function Starts Here

def norm_similarity(rdm1, rdm2, l=2):
    """
    Compute the l_p norm similarity between two RDMs.

    The similarity is calculated as the l_p norm of the difference between the two matrices.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.
    l : int, optional
        The order of the norm (default is 2, Euclidean norm).

    Returns
    -------
    float
        The l_p norm of the difference between the two RDMs.

    Raises
    ------
    ValueError
        If rdm1 and rdm2 do not have the same shape.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    diff = rdm1 - rdm2
    return np.linalg.norm(diff.flatten(), ord=l)

def frobenius_similarity(rdm1, rdm2):
    """
    Compute the Frobenius norm similarity between two RDMs.

    The Frobenius norm is the square root of the sum of the squares of the elements of the difference matrix.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.

    Returns
    -------
    float
        The Frobenius norm of the difference between the two RDMs.

    Raises
    ------
    ValueError
        If rdm1 and rdm2 do not have the same shape.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    diff = rdm1 - rdm2
    return np.linalg.norm(diff, ord='fro')

def cosine_similarity(rdm1, rdm2):
    """
    Compute the cosine similarity between two RDMs.

    The cosine similarity is the dot product of the flattened matrices divided by the product of their Euclidean norms.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.

    Returns
    -------
    float
        Cosine similarity between the two RDMs, ranging from -1 to 1.

    Raises
    ------
    ValueError
        If rdm1 and rdm2 do not have the same shape or if any RDM has zero norm.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    flat1 = rdm1.flatten()
    flat2 = rdm2.flatten()
    dot_product = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One or both RDMs have zero norm")
    return dot_product / (norm1 * norm2)

def correlation_similarity(rdm1, rdm2):
    """
    Compute the Pearson correlation coefficient between two RDMs.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.

    Returns
    -------
    float
        Pearson correlation coefficient between the two RDMs, ranging from -1 to 1.

    Raises
    ------
    ValueError
        If rdm1 and rdm2 do not have the same shape or if correlation is undefined.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    flat1 = rdm1.flatten()
    flat2 = rdm2.flatten()
    corr = np.corrcoef(flat1, flat2)[0, 1]
    if np.isnan(corr):
        raise ValueError("Correlation is undefined for constant vectors")
    return corr

def spearman_rho_similarity(rdm1, rdm2):
    """
    Compute Spearman's rank correlation coefficient between two RDMs.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.

    Returns
    -------
    float
        Spearman's rho, ranging from -1 to 1.

    Raises
    ------
    ValueError
        If rdm1 and rdm2 do not have the same shape or if rho is undefined.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    flat1 = rdm1.flatten()
    flat2 = rdm2.flatten()
    rho, _ = spearmanr(flat1, flat2)
    if np.isnan(rho):
        raise ValueError("Spearman's rho is undefined for constant vectors")
    return rho

def kendall_tau_similarity(rdm1, rdm2, type='beta'):
    """
    Compute Kendall's tau similarity between two RDMs.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.
    type : str, optional
        Type of Kendall's tau to compute. Must be 'beta' (default).

    Returns
    -------
    float
        Kendall's tau statistic.

    Raises
    ------
    ValueError
        If rdm1 and rdm2 do not have the same shape, type is invalid, or tau is undefined.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    if type not in ['beta']:
        raise ValueError("type must be 'beta'")
    flat1 = rdm1.flatten()
    flat2 = rdm2.flatten()
    tau, _ = kendalltau(flat1, flat2, variant='b')
    if np.isnan(tau):
        raise ValueError("Kendall's tau is undefined for constant vectors")
    return tau

def bures_similarity(rdm1, rdm2):
    """
    Compute the Bures distance between two positive definite RDMs.

    References
    ----------
    1. 

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix (positive definite).
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix (positive definite).

    Returns
    -------
    float
        Bures distance between the two RDMs.

    Raises
    ------
    ValueError
        If matrices are not square, have different shapes, or are not positive definite.
    LinAlgError
        If matrix operations fail.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
        raise ValueError("rdm1 must be a square matrix")
    if rdm2.shape[0] != rdm2.shape[1]:
        raise ValueError("rdm2 must be a square matrix")
    if not np.all(np.linalg.eigvalsh(rdm1) > 0):
        raise ValueError("rdm1 is not positive definite")
    if not np.all(np.linalg.eigvalsh(rdm2) > 0):
        raise ValueError("rdm2 is not positive definite")
    
    sqrt_rdm1 = scipy.linalg.sqrtm(rdm1)
    product = sqrt_rdm1 @ rdm2 @ sqrt_rdm1
    sqrt_product = scipy.linalg.sqrtm(product)
    trace_term = np.trace(rdm1 + rdm2 - 2 * sqrt_product)
    return np.sqrt(np.maximum(trace_term, 0))

def procruste_shape_similarity(rdm1, rdm2):
    """
    Compute the Procrustes shape similarity between two RDMs.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.

    Returns
    -------
    float
        Procrustes disparity (sum of squared differences after alignment).

    Raises
    ------
    ValueError
        If rdm1 and rdm2 do not have the same shape.
    """
    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    mtx1 = np.array(rdm1, dtype=np.float64)
    mtx2 = np.array(rdm2, dtype=np.float64)
    _, _, disparity = procrustes(mtx1, mtx2)
    return disparity

def wasserstein_similarity(rdm1, rdm2, normalized_method='Softmax'):
    """
    Compute the Wasserstein distance between two RDMs after normalization.

    References
    ----------
    https://www.biorxiv.org/content/10.1101/2020.11.25.398511v1

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.
    normalized_method : str|callable function, optional
        Normalization method for calculating Wasserstein distance
        if `str`, available probability normalization method is 'Softmax'
        if callable, it should pass parameters like `your_custom_normalizer(rdm1, rdm2)`

    Returns
    -------
    float
        Wasserstein distance between the normalized RDMs.

    Raises
    ------
    ValueError
        If shapes differ or normalization method is invalid.
    """

    ### Error Process

    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    
    if isinstance(normalized_method,str):
        if normalized_method != 'Softmax':
            raise ValueError("Only supported built-in normalizer is Softmax Normalizer.")
        else:
            flat1 = rdm1.flatten()
            flat2 = rdm2.flatten()
    
    if normalized_method == 'Softmax':
        
        prob1 = softmax(flat1)
        prob2 = softmax(flat2)
    else:
        prob1, prob2 = flat1, flat2  # Fallback, though already validated
    
    return wasserstein_distance(prob1, prob2)

def riemann_similarity(rdm1, rdm2):
    """
    Compute the Riemannian distance between two positive definite RDMs.

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix (positive definite).
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix (positive definite).

    Returns
    -------
    float
        Riemannian distance between the RDMs.

    Raises
    ------
    ValueError
        If matrices are not square, have different shapes, or are not positive definite.
    LinAlgError
        If matrix operations fail.
    """

    ### Error Process

    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
        raise ValueError("rdm1 must be a square matrix")
    if rdm2.ndim != 2 or rdm2.shape[0] != rdm2.shape[1]:
        raise ValueError("rdm2 must be a square matrix")
    if not np.all(np.linalg.eigvalsh(rdm1) > 0):
        raise ValueError("rdm1 is not positive definite")
    if not np.all(np.linalg.eigvalsh(rdm2) > 0):
        raise ValueError("rdm2 is not positive definite")
    
    ### Formula: Riemann(X,Y) = \sqrt{\sum \log \sigma_{i}}
    ### where sigma_i is eignevalue of X^{-1}Y.

    inv_rdm1 = np.linalg.inv(rdm1)
    mul_product = inv_rdm1 @ rdm2
    eigenval, _, _ = scipy.linalg.eig(mul_product)
    log_eigenval = np.log(eigenval)
    return np.linalg.norm(log_eigenval)

def ckn_similarity(rdm1, rdm2):
    """
    Compute CKA (Centered Kernel Alignment) similarity between two RDMs.

    References
    ----------
    1. https://www.biorxiv.org/content/10.1101/2020.11.25.398511v1
    2. https://arxiv.org/abs/1905.00414

    Parameters
    ----------
    rdm1 : numpy.ndarray
        First representational dissimilarity matrix.
    rdm2 : numpy.ndarray
        Second representational dissimilarity matrix.

    Returns
    -------
    float
        CKA similarity score.

    Raises
    ------
    ValueError
        If matrices have incompatible shapes or normalization fails.
    """

    ### Error Process

    if rdm1.shape != rdm2.shape:
        raise ValueError("rdm1 and rdm2 must have the same shape")
    
    ### Formula: CKN(X,Y) = HSIC(X,Y) / [HSIC(X,X) HSIC(Y,Y)]
    ### where HSIC(X,Y) = vec(G_X)^{\top} vec(G_Y) vec(·) is flatten operation
    ### and G_X is defined as HX(HX)^{\top} / p, p is the column number and H is the data centerizer.
    ### H = I_p - (11^{\top}) / p
    
    p, _ = rdm1.shape[0]
    centerizer = np.identity(p) - (np.ones_like(p) @ np.ones_like(p).T / p)

    rdm1_centered = centerizer @ rdm1 ## HX
    rdm2_centered = centerizer @ rdm2 ## HY

    G1_mat = rdm1_centered @ rdm1_centered.T ## G1
    G2_mat = rdm2_centered @ rdm2_centered.T ## G2

    hsic = G1_mat.flatten().T @ G2_mat.flatten()  ## vec(G1)^{\top} vec(G2)
    
    # Compute normalization terms
    norm1 = G1_mat.flatten().T @ G1_mat.flatten() ## HSIC(X,X)
    norm2 = G2_mat.flatten().T @ G2_mat.flatten() ## HSIC(Y,Y)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Normalization term is zero")
    
    return hsic / (norm1 * norm2)

def nll_similarity(rdm1, rdm2):
    ""
    return