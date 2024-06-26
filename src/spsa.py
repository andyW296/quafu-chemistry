import numpy as np
import warnings
import scipy

def bernoulli_perturbation(dim, perturbation_dims=None):
    """Get a Bernoulli random perturbation."""
    if perturbation_dims is None:
        return 1 - 2 * np.random.binomial(1, 0.5, size=dim)

    pert = 1 - 2 * np.random.binomial(1, 0.5, size=perturbation_dims)
    indices = np.random.choice(list(range(dim)), size=perturbation_dims, replace=False)
    result = np.zeros(dim)
    result[indices] = pert

    return result

def powerseries(eta=0.01, power=2, offset=0):
    """Yield a series decreasing by a powerlaw."""

    n = 1
    while True:
        yield eta / ((n + offset) ** power)
        n += 1

class SPSA:
    def __init__(self, measure, maxiter=1000, second_order=True, allow_increase=True):

        self.maxiter = maxiter
        self.second_order = second_order
        self.measure = measure
        self.allow_increase = allow_increase
        
    def calibrate(self, c=0.2, alpha=0.602, gamma=0.101, max_evals_grouped=1, modelspace=False, stability_constant=0):
        target_magnitude = 2*np.pi / 10
        dim = len(self.params)
        
        # compute the average magnitude of the first step
        steps = 25
        points = []
        for _ in range(steps):
            # compute the random directon
            pert = bernoulli_perturbation(dim)
            points += [self.params + c * pert, self.params - c * pert]
        
        losses = []
        for i in range(len(points)):
            losses.append(self.measure(points[i]))
        
        avg_magnitudes = 0.0
        for i in range(steps):
            delta = losses[2 * i] - losses[2 * i + 1]
            avg_magnitudes += np.abs(delta / (2 * c))

        avg_magnitudes /= steps

        if modelspace:
            a = target_magnitude / (avg_magnitudes**2)
        else:
            a = target_magnitude / avg_magnitudes
        if a < 1e-10:
            warnings.warn(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude
        
        def learning_rate():
            return powerseries(a, alpha, stability_constant)

        def perturbation():
            return powerseries(c, gamma)

        return learning_rate, perturbation
        
        
    def _make_spd(self, smoothed, bias=0.01):
        identity = np.identity(smoothed.shape[0])
        psd = scipy.linalg.sqrtm(smoothed.dot(smoothed))
        return psd + bias*identity
        
        
    def _updata(self, k, eps):
        values = []
        grad = np.zeros(self.s)
        hessian = np.zeros((self.s, self.s))
        
        detals = list(bernoulli_perturbation(self.s))
        if self.second_order:
            detals += list(bernoulli_perturbation(self.s))
        detals = np.array(detals)
        points = [self.params + eps* detals[:self.s], self.params-eps*detals[:self.s]]
        if self.second_order:
            points += [self.params + eps*(detals[:self.s]+detals[self.s:]), self.params+eps*(-detals[:self.s]+detals[self.s:])]
        points = np.array(points)
        
        for i in range(len(points)):
            values.append(self.measure(points[i]))
            
        grad = (values[0]-values[1])/(2*eps*detals[:self.s])
        
        if self.second_order:
            diff = (values[2]-values[0]) - (values[3]-values[1])
            diff /= 2*eps**2
            rank_one = np.outer(detals[:self.s], detals[self.s:])
            hessian = diff * (rank_one + rank_one.T)/2
            smoothed = k/(k+1) * self.smooth_hession + 1/(k+1) * hessian
            self.smooth_hession = smoothed
            spd_hession = self._make_spd(smoothed)
            grad = np.real(np.linalg.solve(spd_hession, grad))
        return grad
        
    def run(self, params=[]):
        
        self.params = params
        get_eta, get_eps = self.calibrate()
        eta, eps = get_eta(), get_eps()
        self.smooth_hession = np.identity(self.params.size)
        self.s = self.params.size
        energys = []
        best_value = self.measure(self.params)
        energys.append(best_value)
        #print(f"0 iterations, energy is: {best_value}")
        
        k = 0
        while k < self.maxiter:
            k += 1
            grad = self._updata(k, next(eps))
            next_params = self.params - next(eta)*grad
            v = self.measure(next_params)
            if abs(v-best_value) < 1e-12:
                print('Success')
                energys.append(v)
                self.params = next_params
                break
            if self.allow_increase:
                self.params = next_params
                energys.append(v)
            else:
                if v < best_value:
                    print(f"{k} iteration, energy is:{v}")
                    self.params = next_params
                    best_value = v
                    energys.append(v)
        #print(len(energys), energys)
        return self.params, energys