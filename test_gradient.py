import torch
from lcp import CvxpySolver, QpthSolver, LCPPhysics, LCPPhysics2

def print_table(data, width='auto', i0=True):
    if not isinstance(width, int):
        if width == 'auto':
            width = []
            for i in data:
                for idx, j in enumerate(i):
                    if idx >= len(width):
                        width += [0]
                    width[idx] = max(len(str(j))+2, width[idx])
        else:
            raise NotImplementedError

    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(width if isinstance(
            width, int) else width[idx]) for idx, x in enumerate(d))
        print(line)
        if i == 0 and i0:
            print('-' * len(line))

def test():
    torch.manual_seed(0)

    n = 50
    batch_size = 256

    L = torch.randn((batch_size, n, n))
    M = L @ L.transpose(-1, -2) + torch.eye(n)[None,:] * 0.001

    q = torch.randn((batch_size, n))

    M = M.cuda()
    q = q.cuda()

    M.requires_grad = True
    q.requires_grad = True

    solver = {
        'cvxpy': CvxpySolver(n),
        'qpth': QpthSolver(),
        'lcp_phys': LCPPhysics(),
        'lcp_phys_with_minus': LCPPhysics2()
    }

    sols = []
    M_grads = []
    q_grads = []

    algo = ['lcp_phys', 'lcp_phys_with_minus', 'cvxpy', 'qpth']
    for name in algo:
        M.grad, q.grad=None, None

        out = solver[name](M, q)
        (out ** 2).sum().backward()

        M_grads.append(M.grad)
        q_grads.append(q.grad)
        sols.append(out)
    sols = torch.stack(sols, dim=0)
    M_grads = torch.stack(M_grads, dim=0)
    q_grads = torch.stack(q_grads, dim=0)

    diff_sol = (sols[:, None] - sols[None, :]).reshape(len(algo), len(algo),
                                                         -1).abs().max(dim=-1)[0].detach().cpu().numpy()
    print("Parwise difference")

    print("")
    from robot.utils.print_table import print_table
    print_table([['solution']+algo]+[[name] + i.tolist() for name,i in zip(algo, diff_sol)])

    print("")

    diff = (M_grads[:, None] - M_grads[None, :]).reshape(len(algo), len(algo),
                                                         -1).abs().max(dim=-1)[0].detach().cpu().numpy()
    from robot.utils.print_table import print_table
    print_table([['gradient to M']+algo]+[[name] + i.tolist() for name,i in zip(algo, diff)])
    print(M_grads[1].max())



if __name__ == '__main__':
    test()
