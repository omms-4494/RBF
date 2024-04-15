import numpy as np
import scipy.io as sio
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import scipy.integrate as integrate
from plotly import graph_objects as go

# helpers ====================================

def v_soln(xs, ys, soln):
        m, n = np.shape(xs)
        res = np.zeros(shape=(m,n))
        for i in range(m):
            for j in range(n):
                res[i,j] = soln(xs[i,j], ys[i,j])
        return res

def filter_boundary(bc, mesh):
    boundary = []
    interior = []
    for ctr in mesh:
        if bc(ctr):
            boundary.append(ctr)
        else:
            interior.append(ctr)
    return np.array(interior), np.array(boundary)

def err_table(levels, mesh_types, l_infs, l_2s, lh_2s):
    h = ['Level', 'N', 'L_inf Error', 'L2 Error', '~L2 Error']
    h = [f'<b>{x}</b>' for x in h]
    m, n = len(levels), len(mesh_types)
    levs, mts = [], []
    for i in range(m):
        levs.append(levels[i])
        mts = np.concatenate((mts, mesh_types))
        for j in range(n-1):
            levs.append('')
            
    c = [levs, mts, l_infs, l_2s, lh_2s]

    fig = go.Figure(data=[go.Table(
        columnorder = [1,2,3,4,5],
        #columnwidth = [10,10,20,20,20],
        columnwidth = [10,10,50,50,50],
        header = dict(
            values = h,
            line_color='snow',
            fill=dict(color=['silver','silver','powderblue','powderblue','powderblue']),
            align=['center','center'],
            font=dict(color='white', size=17),
            height=60),
        cells=dict(
            values=c,
            line_color='snow',
            fill=dict(color=['aliceblue', 'aliceblue', 'aliceblue', 'aliceblue', 'aliceblue']),
            align=['center', 'center'],
            font_size=15,
            height=40)
        )
    ])
    #fig.update_layout(width=700, height=300*len(levels))
    fig.update_layout(width=1150, height=300)
    fig.show()
    

# rbf option 1 -------------------------------
def rho(r):
    if r >= 1:
        return 0
    else:
        t1 = ((1-r)**8) * (32*(r**3) + 25*(r**2) + 8*r + 1)
        return t1

def drho(r):
    if r >= 1:
        return 0
    else:
        t1 = -8 * ((1-r)**7) * (32*(r**3) + 25*(r**2) + 8*r + 1)
        t2 = ((1-r)**8) * (96*(r**2) + 50*r + 8)
        return t1 + t2

def d2rho(r):
    if r >= 1:
        return 0
    else:
        t1 = 56 * ((1-r)**6) * (32*(r**3) + 25*(r**2) + 8*r + 1)
        t2 = -16 * ((1-r)**7) * (96*(r**2) + 50*r + 8)
        t3 = ((1-r)**8) * (192*r + 50)
        return t1 + t2 + t3

# rbf option 2 ---------------------------------
def phi(r):
    if r >= 1:
        return 0
    else:
        p = (1-r)**5
        q = 8*(r**2) + 5*r + 1
        return p*q

def dphi(r):
    if r >= 1:
        return 0
    else:
        t1 = -5 * ((1-r)**4) * (8*(r**2) + 5*r + 1)
        t2 = ((1-r)**5) * (16*r + 5)
        return t1 + t2

def d2phi(r):
    if r >= 1:
        return 0
    else:
        t1 = 20 * ((1-r)**3) * (8*(r**2) + 5*r + 1)
        t2 = -10 * ((1-r)**4) * (16*r + 5)
        t3 = 16 * ((1-r)**5) 
        return t1 + t2 + t3

# rbf option 3 ---------------------------------
def psi(r):
    if r >= 1:
        return 0
    else:
        return ((1 - r) ** 3) * (3*r + 1)

def dpsi(r):
    if r >= 1:
        return 0
    else:
        ft = 3 * ((1 - r) ** 3)
        st = 3 * ((1 - r) ** 2) * (3*r + 1)
        return (ft - st)

def d2psi(r):
    if r >= 1:
        return 0
    else:
        ft = 6 * (1-r) * (3*r + 1)
        st = 18 * ((1 - r) ** 2)
        return (ft - st)


# rbf option 4 ---------------------------------
def eta(r):
    if r >= 1:
        return 0
    else:
        return ((1 - r) ** 4) * (4*r + 1)

def deta(r):
    if r >= 1:
        return 0
    else:
        ft = 4 * ((1 - r) ** 4)
        st = -4 * ((1 - r) ** 3) * (3*r + 1)
        return ft + st

def d2eta(r):
    if r >= 1:
        return 0
    else:
        ft = 12 * ((1 - r) ** 2) * (4*r + 1)
        st = -16 * ((1 - r) ** 3)
        return ft + st


# 1-dimensional generalized rbf

def rbf1(delta, x, xi):
    r = abs(x-xi)/delta  
    return rho(r)

def drbf1(delta, x, xi):
    r = abs(x-xi)/delta
    rx = np.sign(x-xi)/delta
    return drho(r) * rx

def d2rbf1(delta, x, xi):
    r = abs(x-xi)/delta
    return d2rho(r)/(delta**2)


def recreate_uk(x, centers, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi = centers[i]
        sum += alphas[i] * rbf1(deltas[i], x, xi)
    return sum

def recreate_ukx(x, centers, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi = centers[i]
        sum += alphas[i] * drbf1(deltas[i], x, xi)
    return sum

def recreate_ukxx(x, centers, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi = centers[i]
        sum += alphas[i] * d2rbf1(deltas[i], x, xi)
    return sum


# 2-dimensional generic rbf ======================================
def rbf2(delta, x, y, xi, yi):
    r = np.sqrt(((x-xi)**2) + ((y-yi)**2))/delta
    return rho(r)

def dxrbf2(delta, x, y, xi, yi):
    dist = np.sqrt(((x-xi)**2) + ((y-yi)**2))
    r = dist/delta
    if x == xi:
        rx = 0
    else:
        rx = (x-xi) / (delta * dist)
    return rx * drho(r)
    
def dyrbf2(delta, x, y, xi, yi):
    dist = np.sqrt(((x-xi)**2) + ((y-yi)**2))
    r = dist/delta
    if y == yi:
        ry = 0
    else:
        ry = (y-yi) / (delta * dist)
    return ry * drho(r)
    
def dxxrbf2(delta, x, y, xi, yi):
    dist = np.sqrt(((x-xi)**2) + ((y-yi)**2))
    r = dist/delta
    if x == xi:
        rx = 0
    else:
        rx = (x-xi) / (delta * dist)
    if y == yi:
        rxx = 0
    else:
        rxx = ((y-yi)**2) / (delta * (dist**3))
    
    return d2rho(r)*(rx**2) + drho(r)*rxx
    
def dyyrbf2(delta, x, y, xi, yi):
    dist = np.sqrt(((x-xi)**2) + ((y-yi)**2))
    r = dist/delta
    if y == yi:
        ry = 0
    else:
        ry = (y-yi) / (delta * dist)
    if x == xi:
        ryy = 0
    else:
        ryy = ((x-xi)**2) / (delta * (dist**3))
    
    return d2rho(r)*(ry**2) + drho(r)*ryy

def recreate_wk(x, y, cxs, cys, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi, yi = cxs[i], cys[i]
        sum += alphas[i] * rbf2(deltas[i], x, y, xi, yi)
    return sum

def recreate_wkx(x, y, cxs, cys, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi, yi = cxs[i], cys[i]
        sum += alphas[i] * dxrbf2(deltas[i], x, y, xi, yi)
    return sum

def recreate_wky(x, y, cxs, cys, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi, yi = cxs[i], cys[i]
        sum += alphas[i] * dyrbf2(deltas[i], x, y, xi, yi)
    return sum

def recreate_wkxx(x, y, cxs, cys, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi, yi = cxs[i], cys[i]
        sum += alphas[i] * dxxrbf2(deltas[i], x, y, xi, yi)
    return sum

def recreate_wkyy(x, y, cxs, cys, deltas, alphas):
    sum = 0 
    for i in range(len(alphas)):
        xi, yi = cxs[i], cys[i]
        sum += alphas[i] * dyyrbf2(deltas[i], x, y, xi, yi)
    return sum

def rp(eps, x, y, xi, yi, xj, yj):
    return rbf2(eps, x, y, xi, yi) * rbf2(eps, x, y, xj, yj)

def gp(eps, x, y, xi, yi, xj, yj):
    dx = dxrbf2(eps, x, y, xi, yi) * dxrbf2(eps, x, y, xj, yj)
    dy = dyrbf2(eps, x, y, xi, yi) * dyrbf2(eps, x, y, xj, yj)
    return dx + dy


# 1-dimensional galerkin or collocation approximator
class Approximator1D:
    def __init__(self, a, b, f, Ns, delta, outer, bc_type='neumann', app_type='galerkin', F=0, scale_delta=False):
        self.a = a
        self.b = b
        self.f = f
        self.Ns = Ns
        self.meshes = [np.linspace(a, b, N) for N in Ns]
        self.outer = outer
        self.bc_type = bc_type
        self.app_type = app_type
        sigma = 3 # floor(sigma) > 2 + d/2 (in 1-d we need sigma > 2.5) 
        if F == 0:
            self.F = lambda x: 0
        else:
            self.F = F
        if scale_delta:
            self.deltas = [(delta * (b-a)/N)**(1-(2/sigma)) for N in Ns]
        else:
            self.deltas = [delta for N in Ns]

    def l_inf(self, xs, ys):
        dist = np.sqrt((xs - ys)**2)
        return np.max(dist)

    def l_2(self, f1, f2):
        integrand = lambda x: (f1(x) - f2(x))**2
        integral, err = integrate.quad(integrand, self.a, self.b)
        return np.sqrt(integral)

    def lh_2(self, f1, f2):
        a = self.a + (self.b - self.a)/4
        b = self.b - (self.b - self.a)/4
        integrand = lambda x: (f1(x) - f2(x))**2
        integral, err = integrate.quad(integrand, a, b)
        return np.sqrt(integral)

    def A_(self, level):
        N = self.Ns[level]
        ctrs = self.meshes[level]
        delta = self.deltas[level]
        A = np.zeros(shape=(N,N))

        if self.app_type == 'collocation':
            for i in range(N):
                xi = ctrs[i]
                if self.bc_type == 'dirichlet':
                    A[0, i] = rbf1(delta, self.a, xi)
                    A[N-1, i] = rbf1(delta, self.b, xi)
                if self.bc_type == 'neumann':
                    A[0, i] = drbf1(delta, self.a, xi)
                    A[N-1, i] = drbf1(delta, self.b, xi)
                
            for i in range(1, N-1):
                for j in range(N):
                    A[i, j] = rbf1(delta, ctrs[i], ctrs[j]) - d2rbf1(delta, ctrs[i], ctrs[j])
        elif self.app_type == 'galerkin':
            for i in range(N):
                xi = ctrs[i]
                for j in range(i+1):
                    xj = ctrs[j]
                    integrand = lambda y: drbf1(delta, y, xi)*drbf1(delta, y, xj) + rbf1(delta, y, xi)*rbf1(delta, y, xj)
                    integral, err = integrate.quad(integrand, self.a, self.b)
                    A[i,j] = integral
                    A[j,i] = integral
        else:
            print('Check that you initiated the \'app_type\' variable correctly.')
            
        return A

    def fi(self, level):
        N = self.Ns[level]
        ctrs = self.meshes[level]
        delta = self.deltas[level]
        fs = np.zeros(shape=(N))

        if self.app_type == 'collocation':
            fs[0] = self.F(self.a)
            fs[N-1] = self.F(self.b)
            for i in range(1, N-1):
                fs[i] = self.f(ctrs[i])
        elif self.app_type == 'galerkin':
            for i in range(N):
                integrand = lambda y: self.f(y) * rbf1(delta, y, ctrs[i])
                integral, err = integrate.quad(integrand, self.a, self.b, limit=100)
                fs[i] = integral
        return fs

    def aip(self, u, v, ux, vx):
        integrand = lambda y: (ux(y) * vx(y)) + (u(y) * v(y))
        integral, err = integrate.quad(integrand, self.a, self.b, limit=100) 
        return integral

    def rhs(self, uk, ukx, ukxx, level):
        N = self.Ns[level]
        ctrs = self.meshes[level]
        delta = self.deltas[level]
        
        fs = self.fi(level)
        sk = np.zeros(shape=(N))
        if self.app_type == 'collocation':
            if self.bc_type == 'dirichlet':
                sk[0] = uk(self.a)
                sk[N-1] = uk(self.b)
            if self.bc_type == 'neumann':
                sk[0] = ukx(self.a)
                sk[N-1] = ukx(self.b)
            for i in range(1, N-1):
                sk[i] = uk(ctrs[i]) - ukxx(ctrs[i])
        if self.app_type == 'galerkin':
            for i in range(N):
                vx = lambda y : drbf1(delta, y, ctrs[i])
                v = lambda y : rbf1(delta, y, ctrs[i])
                sk[i] = self.aip(uk, v, ukx, vx)

        return np.subtract(fs, sk)

    def solve(self, uk, ukx, ukxx, level):
        A = self.A_(level)
        r = self.rhs(uk, ukx, ukxx, level)
        eigs = np.linalg.eigvals(A)
        cn = abs(np.max(eigs))/abs(np.min(eigs))
        A_inv = np.linalg.inv(A)
        alpha = A_inv @ r
        return alpha, cn

    def innerlevel(self, centers, deltas, alphas, u, neval):
        l_inf, l_2, lh_2, cns = [], [], [], []
        
        for level in range(len(self.Ns)):
            uk = lambda x: recreate_uk(x, centers, deltas, alphas)
            ukx = lambda x: recreate_ukx(x, centers, deltas, alphas)
            ukxx = lambda x: recreate_ukxx(x, centers, deltas, alphas)
            alpha, cn = self.solve(uk, ukx, ukxx, level)
            
            alphas = np.concatenate((alphas, alpha), axis=0)
            centers = np.concatenate((centers, self.meshes[level]), axis=0)
            deltas = np.concatenate((deltas, [self.deltas[level] for center in self.meshes[level]]), axis=0)
            
            uk = lambda x: recreate_uk(x, centers, deltas, alphas)

            xs = np.linspace(self.a, self.b, neval)
            app = [uk(x) for x in xs]
            solution = [u(x) for x in xs]
            l_inf_val = np.format_float_scientific(self.l_inf(np.array(app), np.array(solution)), precision=16)
            l_2_val = np.format_float_scientific(self.l_2(uk, u), precision=16)
            lh_2_val = np.format_float_scientific(self.lh_2(uk, u), precision=16)
            l_inf.append(l_inf_val)
            l_2.append(l_2_val)
            lh_2.append(lh_2_val)
            cns.append(cn)
            
        return centers, deltas, alphas, l_inf, l_2, lh_2, cns

    def multilevel(self, u, neval, plots='all', cn_plot=False, show_errs=False):
        centers, deltas, alphas = [], [], []
        l_infs, l_2s, lh_2s, cns = [], [], [], []

        xs = np.linspace(self.a, self.b, neval)
        solution = [u(x) for x in xs]
        
        for j in range(self.outer):
            centers, deltas, alphas, l_inf, l_2, lh_2, cn = self.innerlevel(centers, deltas, alphas, u, neval)
            l_infs = np.concatenate((l_infs, l_inf))
            l_2s = np.concatenate((l_2s, l_2))
            lh_2s = np.concatenate((lh_2s, lh_2))
            cns = np.concatenate((cns, cn))

        uk = lambda x: recreate_uk(x, centers, deltas, alphas)
        app = [uk(x) for x in xs]
        abs_error = [abs(uk(x) - u(x)) for x in xs]

        if plots == 'all':
            fig, axs = plt.subplots(1, 3, figsize=(18,6))
            #fig.suptitle(f'Multilevel PDE Approximation: Meshes = {self.Ns}, Levels = {self.outer}')
            fig.supxlabel('x', fontsize=17)
            fig.supylabel('y', fontsize=17)
            axs[0].plot(xs, solution, 'b')
            axs[1].plot(xs, app, 'r')
            axs[0].set_title(r'Exact Solution: $y = u(x)$', fontsize=20)
            axs[1].set_title(r'Approximation: $y = u_{K}(x)$', fontsize=20)
            axs[0].tick_params(axis='both', labelsize=16)
            axs[1].tick_params(axis='both', labelsize=16)

            max_err = max(abs_error)
            k=0
            while 10**(-1*k) > max_err:
                k+=1
            scaled_err = [y * (10**k) for y in abs_error]
            axs[2].plot(xs, scaled_err, 'k')
            axs[2].tick_params(axis='both', labelsize=16)
            axs[2].set_title(r'Absolute Error: $y = |u(x) - u_{K}(x)|$', fontsize=20)
            if k > 0:
                scale_text = f'Scale: 1e-{k}'
                axs[2].annotate(scale_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.show()

        if plots == 'true and approximation':
            fig, axs = plt.subplots(1, 1, figsize=(6,4.5))
            #fig.suptitle(f'Multilevel PDE Approximation: Meshes = {self.Ns}, Levels = {self.outer}')
            axs.set_xlabel('x', fontsize=16)
            axs.set_ylabel('y', fontsize=16)
            axs.plot(xs, solution, 'b', label=r'$u(x)$')
            axs.plot(xs, app, 'r', label=r'$u_N(x)$')
            axs.tick_params(axis='both', labelsize=16)
            axs.legend(fontsize=23)
            plt.tight_layout()
            plt.show()

        if plots == 'true only':
            fig, axs = plt.subplots(1, 1, figsize=(6,4.5))
            axs.set_xlabel('x', fontsize=16)
            axs.set_ylabel('y', fontsize=16)
            axs.set_title(r'Exact Solution: $y = u(x)$', fontsize=20)
            axs.plot(xs, solution, 'b')
            axs.tick_params(axis='both', labelsize=16)
            plt.tight_layout()
            plt.show()
        
        if plots == 'error only':
            fig, axs = plt.subplots(1, 1, figsize=(6,4.5))
            axs.set_xlabel('x', fontsize=16)
            axs.set_ylabel('y', fontsize=16)
            axs.tick_params(axis='both', labelsize=16)
            max_err = max(abs_error)
            k=0
            while 10**(-1*k) > max_err:
                k+=1
            scaled_err = [y * (10**k) for y in abs_error]
            axs.plot(xs, scaled_err, 'k')
            axs.tick_params(axis='both', labelsize=16)
            axs.set_title(r'Absolute Error: $y = |u(x) - u_{K}(x)|$', fontsize=20)
            if k > 0:
                scale_text = f'Scale: 1e-{k}'
                axs.annotate(scale_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.show()
        
        # condition number plot
        if cn_plot:
            print(cns)
            fig, ax = plt.subplots(1, 1, figsize=(6,3))
            plt.title(r'Condition Number Progression of Matrix $A$')
            plt.ylabel('K(A)')
            plt.xlabel('Level')
            xs = np.arange(1, self.outer*len(self.Ns)+1)
            plt.plot(xs, cns, marker='o')
            plt.xticks(np.arange(1, len(self.Ns) * (self.outer), step=len(self.Ns)))
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # error table
        if show_errs:
            err_table(np.arange(1, self.outer+1), self.Ns, l_infs, l_2s, lh_2s)

        return l_infs, l_2s, lh_2s


# 2-dimensional galerkin or collocation approximator

class Approximator2D:
    def __init__(self, a, b, c, d, f, Ns, delta, outer, boundary, bc_type='neumann', app_type='galerkin', F=0, Fx=0, Fy=0, scale_delta=False):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.f = f
        self.Ns = Ns
        self.outer = outer
        self.bc_type = bc_type
        self.app_type = app_type
        if F == 0:
            self.F = lambda x,y: 0
        else:
            self.F = F
        if Fx == 0:
            self.Fx = lambda x,y: 0
        else:
            self.Fx = Fx
        if Fy == 0:
            self.Fy = lambda x,y: 0
        else:
            self.Fy = Fy
        if scale_delta:
            sigma = 3
            self.deltas = [(delta * (b-a)/np.sqrt(N))**(1-(2/sigma)) for N in Ns]
        else:
            self.deltas = [delta for N in Ns]
        mats = [sio.loadmat(f'KGillowScripts/Data2D_{Ns[i]}u.mat') for i in range(len(Ns))]
        self.meshes = [(b-a) * mat['dsites'] + a for mat in mats]
        self.interiors, self.boundaries = [], []
        for ctrs in self.meshes:
            i, b = filter_boundary(boundary, ctrs)
            self.interiors.append(i)
            self.boundaries.append(b)
        if self.app_type == 'collocation':
            self.meshes = [np.concatenate((self.interiors[i], self.boundaries[i])) for i in range(len(self.interiors))]

    def l_inf(self, f1, f2, xs, ys):
        exact = f1(xs, ys)
        approx = f2(xs, ys)
        dist = np.sqrt((exact - approx)**2)
        return np.max(dist)

    def l_2(self, f1, f2):
        integrand = lambda x, y: (f1(x, y) - f2(x, y))**2
        integral, err = integrate.dblquad(integrand, self.c, self.d, self.a, self.b)
        return np.sqrt(integral)

    def lh_2(self, f1, f2):
        a = self.a + (self.b - self.a)/4
        b = self.b - (self.b - self.a)/4
        c = self.c + (self.d - self.c)/4
        d = self.d - (self.d - self.c)/4
        integrand = lambda x, y: (f1(x, y) - f2(x, y))**2
        integral, err = integrate.dblquad(integrand, c, d, a, b)
        return np.sqrt(integral)
        
    def A_(self, level):
        N = self.Ns[level]
        ctrs = self.meshes[level]
        interior = self.interiors[level]
        boundary = self.boundaries[level]
        delta = self.deltas[level]
        A = np.zeros(shape=(N,N))

        if self.app_type == 'collocation':
            n = len(interior)
            for i, ctr in enumerate(interior):
                xi, yi = ctr[0], ctr[1]
                for j, ectr in enumerate(interior):
                    xj, yj = ectr[0], ectr[1]
                    A[i, j] = rbf2(delta, xi, yi, xj, yj) - dxxrbf2(delta, xi, yi, xj, yj) - dyyrbf2(delta, xi, yi, xj, yj)
                for j, ectr in enumerate(boundary):
                    xj, yj = ectr[0], ectr[1]
                    A[i, n+j] = rbf2(delta, xi, yi, xj, yj) - dxxrbf2(delta, xi, yi, xj, yj) - dyyrbf2(delta, xi, yi, xj, yj)
            for i, ctr in enumerate(boundary):
                xi, yi = ctr[0], ctr[1]
                for j, ectr in enumerate(interior):
                    xj, yj = ectr[0], ectr[1]
                    if self.bc_type == 'dirichlet':
                        A[n+i, j] = rbf2(delta, xi, yi, xj, yj)
                    elif self.bc_type == 'neumann':
                        if xi == self.a:
                            A[n+i, j] = -1 * dxrbf2(delta, xi, yi, xj, yj)
                        elif xi == self.b:
                            A[n+i, j] = dxrbf2(delta, xi, yi, xj, yj)
                        elif yi == self.c:
                            A[n+i, j] = -1 * dyrbf2(delta, xi, yi, xj, yj)
                        elif yi == self.d:
                            A[n+i, j] = dyrbf2(delta, xi, yi, xj, yj)
                for j, ectr in enumerate(boundary):
                    xj, yj = ectr[0], ectr[1]
                    if self.bc_type == 'dirichlet':
                        A[n+i, n+j] = rbf2(delta, xi, yi, xj, yj)
                    elif self.bc_type == 'neumann':
                        if xi == self.a:
                            A[n+i, n+j] = -1 * dxrbf2(delta, xi, yi, xj, yj)
                        elif xi == self.b:
                            A[n+i, n+j] = dxrbf2(delta, xi, yi, xj, yj)
                        elif yi == self.c:
                            A[n+i, n+j] = -1 * dyrbf2(delta, xi, yi, xj, yj)
                        elif yi == self.d:
                            A[n+i, n+j] = dyrbf2(delta, xi, yi, xj, yj)
                        
        elif self.app_type == 'galerkin':
            for i in range(N):
                xi, yi = ctrs[i,0], ctrs[i, 1]
                for j in range(i+1):
                    xj, yj = ctrs[j,0], ctrs[j, 1]
                    integrand = lambda x, y: rp(delta, x, y, xi, yi, xj, yj) + gp(delta, x, y, xi, yi, xj, yj)
                    integral, err = integrate.dblquad(integrand, self.c, self.d, self.a, self.b)
                    A[i, j] = integral
                    A[j, i] = integral
        else:
            print('Check that you initiated the \'app_type\' variable correctly.')
            
        return A

    def fi(self, level):
        N = self.Ns[level]
        ctrs = self.meshes[level]
        delta = self.deltas[level]
        interior = self.interiors[level]
        boundary = self.boundaries[level]
        n = len(interior)
        fs = np.zeros(shape=(N))

        if self.app_type == 'collocation':
            for i, ctr in enumerate(interior):
                xi, yi = ctr[0], ctr[1]
                fs[i] = self.f(xi, yi)
            for i, ctr in enumerate(boundary):
                xi, yi = ctr[0], ctr[1]
                if self.bc_type == 'dirichlet':
                    fs[n+i] = self.F(xi,yi)
                elif self.bc_type == 'neumann':
                    if xi == self.a:
                        fs[n+i] = -1 * self.Fx(xi,yi)
                    elif xi == self.b:
                        fs[n+i] = self.Fx(xi,yi)
                    elif yi == self.c:
                        fs[n+i] = -1 * self.Fy(xi,yi)
                    else:
                        fs[n+i] = self.Fy(xi,yi)
        elif self.app_type == 'galerkin':
            for i in range(N):
                xi, yi = ctrs[i,0], ctrs[i, 1]
                integrand = lambda x, y: self.f(x, y) * rbf2(delta, x, y, xi, yi)
                integral, err = integrate.dblquad(integrand, self.c, self.d, self.a, self.b)
                fs[i] = integral
        return fs

    def aip(self, u, ux, uy, v, vx, vy):
        integrand = lambda x, y: u(x,y)*v(x,y) + ux(x,y)*vx(x,y) + uy(x,y)*vy(x,y)
        integral, err = integrate.dblquad(integrand, self.c, self.d, self.a, self.b)
        return integral

    def rhs(self, uk, ukx, uky, ukxx, ukyy, level):
        N = self.Ns[level]
        ctrs = self.meshes[level]
        delta = self.deltas[level]
        interior = self.interiors[level]
        boundary = self.boundaries[level]
        n = len(interior)
        
        fs = self.fi(level)
        sk = np.zeros(shape=(N))
        if self.app_type == 'collocation':
            for i, ctr in enumerate(interior):
                xi, yi = ctr[0], ctr[1]
                sk[i] = uk(xi,yi) - ukxx(xi,yi) - ukyy(xi,yi)
            for i, ctr in enumerate(boundary):
                xi, yi = ctr[0], ctr[1]
                if self.bc_type == 'dirichlet':
                    sk[n+i] = uk(xi,yi)
                if self.bc_type == 'neumann':
                    if xi == self.a:
                        sk[n+i] = -1 * ukx(xi,yi)
                    elif xi == self.b:
                        sk[n+i] = ukx(xi,yi)
                    elif yi == self.c:
                        sk[n+i] = -1 * uky(xi,yi)
                    else:
                        sk[n+i] = uky(xi,yi)
        if self.app_type == 'galerkin':
            for i in range(N):
                xi, yi = ctrs[i,0], ctrs[i, 1]
                v = lambda x,y: rbf2(delta, x, y, xi, yi)
                vx = lambda x,y: dxrbf2(delta, x, y, xi, yi)
                vy = lambda x,y: dyrbf2(delta, x, y, xi, yi)
                sk[i] = self.aip(uk, ukx, uky, v, vx, vy)

        return np.subtract(fs, sk)

    def solve(self, uk, ukx, uky, ukxx, ukyy, level):
        A = self.A_(level)
        r = self.rhs(uk, ukx, uky, ukxx, ukyy, level)
        eigs = np.linalg.eigvals(A)
        cn = abs(np.max(eigs))/abs(np.min(eigs))
        A_inv = np.linalg.inv(A)
        alpha = A_inv @ r
        return alpha, cn

    def innerlevel(self, cxs, cys, deltas, alphas, u, neval):
        l_inf, l_2, lh_2, cns = [], [], [], []
        
        for level in range(len(self.Ns)):
            
            uk = lambda x,y: recreate_wk(x, y, cxs, cys, deltas, alphas)
            ukx = lambda x,y: recreate_wkx(x, y, cxs, cys, deltas, alphas)
            uky = lambda x,y: recreate_wky(x, y, cxs, cys, deltas, alphas)
            ukxx = lambda x,y: recreate_wkxx(x, y, cxs, cys, deltas, alphas)
            ukyy = lambda x,y: recreate_wkyy(x, y, cxs, cys, deltas, alphas)
            alpha, cn = self.solve(uk, ukx, uky, ukxx, ukyy, level)

            alphas = np.concatenate((alphas, alpha), axis=0)
            cxs = np.concatenate((cxs, self.meshes[level][:,0]), axis=0)
            cys = np.concatenate((cys, self.meshes[level][:,1]), axis=0)
            deltas = np.concatenate((deltas, [self.deltas[level] for center in self.meshes[level]]), axis=0)
            cns.append(cn)

            # error analysis
            xs = np.linspace(self.a, self.b, neval)
            ys = np.linspace(self.c, self.d, neval)
            xs, ys = np.meshgrid(xs, ys)

            uk = lambda x,y: recreate_wk(x, y, cxs, cys, deltas, alphas)
            v_uk = lambda xs, ys: v_soln(xs, ys, uk)
            l_inf_val = np.format_float_scientific(self.l_inf(u, v_uk, xs, ys), precision=5)
            l_2_val = np.format_float_scientific(self.l_2(u, uk), precision=5)
            lh_2_val = np.format_float_scientific(self.lh_2(u, uk), precision=5)
            l_inf.append(l_inf_val)
            l_2.append(l_2_val)
            lh_2.append(lh_2_val)

        return cxs, cys, deltas, alphas, l_inf, l_2, lh_2, cns

    def multilevel(self, u, neval, cn_plot=False, show_errs=False):
        cxs, cys, deltas, alphas = np.array([]), np.array([]), np.array([]), np.array([])
        l_infs, l_2s, lh_2s, cns = np.array([]), np.array([]), np.array([]), np.array([])

        # plotting
        xs = np.linspace(self.a, self.b, neval)
        ys = np.linspace(self.c, self.d, neval)
        xs, ys = np.meshgrid(xs, ys)
        
        fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"}, figsize=(20,6))

        for j in range(self.outer):
            cxs, cys, deltas, alphas, l_inf, l_2, lh_2, cn = self.innerlevel(cxs, cys, deltas, alphas, u, neval)
            l_infs = np.concatenate((l_infs, l_inf))
            l_2s = np.concatenate((l_2s, l_2))
            lh_2s = np.concatenate((lh_2s, lh_2))
            cns = np.concatenate((cns, cn))
            
        uk = lambda x,y: recreate_wk(x, y, cxs, cys, deltas, alphas)
        zs = v_soln(xs, ys, uk)
        v_uk = lambda xs, ys: v_soln(xs, ys, uk)
        exact = u(xs, ys)
        errs = np.sqrt((exact - zs)**2)

        max_err = max([max(row) for row in errs])
        k=0
        while 10**(-1*k) > max_err:
            k+=1
        scaled_errs = errs * (10**k)
        color_dimension = scaled_errs 
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = colors.Normalize(minn, maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

        surf = axs[0].plot_surface(xs, ys, exact, facecolors=fcolors, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        axs[0].zaxis.set_major_locator(LinearLocator(5))
        axs[0].zaxis.set_major_formatter('{x:.02f}')
        axs[0].tick_params(labelsize=15)
        axs[0].set_xlabel('x', fontsize=17)
        axs[0].set_ylabel('y', fontsize=17)
        axs[0].set_zlabel('z', fontsize=17)
        
        surf = axs[1].plot_surface(xs, ys, zs, facecolors=fcolors, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        axs[1].zaxis.set_major_locator(LinearLocator(5))
        axs[1].zaxis.set_major_formatter('{x:.02f}')
        axs[1].tick_params(labelsize=15)
        axs[1].set_xlabel('x', fontsize=17)
        axs[1].set_ylabel('y', fontsize=17)
        axs[1].set_zlabel('z', fontsize=17)
            
        err = axs[2].plot_surface(xs, ys, scaled_errs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        axs[2].zaxis.set_major_locator(LinearLocator(5))
        axs[2].zaxis.set_major_formatter('{x:.02f}')
        axs[2].tick_params(labelsize=15)
        axs[2].set_xlabel('x', fontsize=17)
        axs[2].set_ylabel('y', fontsize=17)
        axs[2].set_zlabel('z', fontsize=17)
        if k > 0:
            scale_text = f'Scale: 1e-{k}'
            axs[2].annotate(scale_text, xy=(0.85, 0.95), xycoords='axes fraction', fontsize=19,
                            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        axs[0].set_title(r'Exact Solution: $z = u(x,y)$', fontsize=20)
        axs[1].set_title(r'Approximation: $z = u_{K}(x,y)$', fontsize=20)
        axs[2].set_title(r'Absolute Error: $z = |u(x,y) - u_{K}(x,y)|$', fontsize=20)
            
        cbar = fig.colorbar(err, shrink=0.5, aspect=5)
        cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()    
        plt.show()

        if cn_plot:
            # condition number plot
            fig, ax = plt.subplots(1, 1, figsize=(6,3))
            plt.title(r'Condition Number Progression of Matrix $A$')
            plt.ylabel('K(A)')
            plt.xlabel('Level')
            xs = np.arange(1, self.outer*len(self.Ns)+1)
            plt.plot(xs, cns, marker='o')

            plt.xticks(np.arange(1, len(self.Ns) * (self.outer), step=len(self.Ns)))
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        if show_errs:
            # error table
            err_table(np.arange(1, self.outer+1), self.Ns, l_infs, l_2s, lh_2s)
    
    