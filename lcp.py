# the simplest lcp solver
# projected gauss-sediel
# http://image.diku.dk/kenny/download/erleben.13.siggraph.course.notes.pdf
import numpy as np
import torch


def eyes_like(mat, n=None):
    if n is None:
        n = mat.shape[-1]
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    while len(eye.shape) < len(mat.shape):
        eye = eye[None,:]
    return eye.expand(*mat.shape[:-2], -1, -1)


class CvxpySolver:
    def __init__(self, n):
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        x = cp.Variable(n)
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        objective = cp.Minimize(0.5 * cp.sum_squares(A@x) + x.T @ b)
        constraints = [x >= 0]
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        self.cvxpylayer = cvxpylayer

    def __call__(self, M, q):
        return self.cvxpylayer(torch.cholesky(M).transpose(-1, -2), q)[0]


class QpthSolver:
    def __init__(self, max_iter=20):
        from qpth.qp import QPFunction
        self.f = QPFunction(eps=1e-12, verbose=True, maxIter=max_iter, notImprovedLim=10)

    def __call__(self, M, q):
        """
        min 1/2 u^TMu + q^Tu s.t. -Mu<=q and -u<=0
        """
        A = torch.tensor([], dtype=M.dtype, device=M.device)
        b = torch.tensor([], dtype=M.dtype, device=M.device)
        G = M.new_zeros(M.shape[0], M.shape[1] * 2, M.shape[2])
        G[:,:M.shape[1]] = -M
        G[:,M.shape[1]:] = -eyes_like(M)
        h = M.new_zeros(M.shape[0], M.shape[1] * 2)
        h[:,:M.shape[1]] = q
        return self.f(M, q, G, h, A, b)


class LCPPhysics:
    def __init__(self):
        from lcp_physics.lcp.lcp import LCPFunction
        self.f = LCPFunction(max_iter=30, verbose=True, not_improved_lim=5, eps=1e-15)

    def __call__(self, M, q):
        n = M.shape[-1]
        # Q, p, G, h, A, b, F
        # M  q  G  m        -F
        A = torch.tensor([])
        b = torch.tensor([])
        h = q
        Q = eyes_like(M)
        p = q * 0
        G = -eyes_like(M)
        F = M + G
        out = self.f(Q, p, G, h, A, b, F)
        return out

class LCPPhysics2(LCPPhysics):
    def __init__(self):
        from lcp_physics.lcp.lcp import LCPFunction
        class LCPFunction2(LCPFunction):
            def backward(*args, **kwargs):
                grad = LCPFunction.backward(*args, **kwargs)
                return grad[:-1]+(-grad[-1],)

        self.f = LCPFunction2(max_iter=30, verbose=True, not_improved_lim=5, eps=1e-15)
