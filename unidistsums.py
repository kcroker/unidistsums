import scipy.stats as st
from scipy.special import factorial
import numpy as np
import itertools
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

class luskwright(st.rv_continuous):

    def __init__(self, *args, **kwds):

        # Set some resolution for the CDF and PPF
        N = 10000
        
        # Sanity check
        for i in kwds['intervals']:
            if not isinstance(i, tuple) and not len(i) == 2 and not i[0] < i[1] and not i[0] == 0.0:
                raise Exception

        # Remove from kwds the thing we care about
        self.intervals = np.asarray(kwds['intervals'])

        print(self.intervals)
            
        # Call underlying machinery, where we need to remove
        # this keyword in order to avoid breaking upstream
        del(kwds['intervals'])
        super().__init__(args, kwds)
        
        # Define Lusk & Wright c_i
        self._ci = self.intervals[:,1]
        
        # Compute the Lusk & Wright cstar
        self._cstar = np.prod(self._ci)

        # Compute the (fixed) denominator of Lusk & Wright (1)
        self._pdfdenominator = 1./(self._cstar * factorial(len(self._ci) - 1))
        
        # Compute the Lusk & Wright q
        self._q = 2**len(self._ci)

        # Compute the maximum support for the pdf
        self._maxsupport = np.sum(self._ci)
        
        # Compute the Lust & Wright l_k's
        # Sticking a zero combination in the front gets the length and values of the l's correct.
        combos = [(0,)]
        for count in range(len(self._ci)):
            combos += list(sorted(itertools.combinations(self._ci, count+1)))

        self._l = np.asarray([np.sum(combo) for combo in combos])

        print(combos)

        # Sanity check
        if not len(combos) == self._q:
            print("TROUBLE: Disagreement between number of combinations and q")
            exit(1)
            
        # Compute the Lust & Wright o(l_k)
        # By definition, o(l_0) := 0
        self._o = [0] + [len(combo) for combo in combos[1:]]
        
        # Debug
        print("Zero offset upper bounds: ", self._ci)
        print("o(l_k): ", self._o)
        print("l_k: ", self._l)
        
        # Use quad repeatedly to approximate the CDF on a dense domain
        dom = np.linspace(0, self._maxsupport, N)
        cumt = cumtrapz([self.h(w) for w in dom], x=dom, initial=0.0) # cumt = np.array([quad(self.h, 0, w)[0] for w in dom])

        # Use interpolation to create the point percent function for rapid variates
        self.internal_cdf = interp1d(dom, cumt, bounds_error=False, fill_value=(0.0,1.0))
        self.internal_ppf = interp1d(cumt, dom)
        
    # Define the Heaviside
    def theta(self, w, w_c):
        return np.where(w > w_c, 1,
                        np.where(w < w_c, 0, 0.5))
            
    # This is the Lusk & Wright g(w) function
    # We reindex his sum to be zero indexed.
    def h(self, w):
        return np.where(w > self._maxsupport,
                        0,
                        self._pdfdenominator * np.sum(np.asarray([(-1)**self._o[k]*(w - self._l[k])**(len(self._ci) - 1)*self.theta(w, self._l[k]) for k in range(self._q - 1)]), axis=0))

    def _pdf(self, w):
        return self.h(w)

    def _cdf(self, w):
        return self.internal_cdf(w)

    def _ppf(self, P):
        return self.internal_ppf(P)

class parsibrosh_unweighted(st.rv_continuous):

    def __init__(self, *args, **kwds):

        # Sanity check
        for i in kwds['intervals']:
            if not isinstance(i, tuple) and not len(i) == 2 and not i[0] < i[1]:
                raise Exception

        # Remove from kwds the thing we care about
        self.intervals = np.asarray(kwds['intervals'])

        print(self.intervals)
            
        # Call underlying machinery, where we need to remove
        # this keyword in order to avoid breaking upstream
        del(kwds['intervals'])
        super().__init__(args, kwds)

        # Define some convenience stuff
        self.beta_minus_alpha = self.intervals[:,1] - self.intervals[:,0]
        self.beta_plus_alpha = np.sum(self.intervals, axis=1)

        # Define quantities that appear in Parsi-Brosh (9)
        self.N = len(self.intervals)
        self.mu = 0.5*np.sum(self.beta_plus_alpha)
        self.tilde_a = np.prod(self.beta_minus_alpha)
        
        # Compute all possible permutations in sign
        self.sign_sets = itertools.product((-1,1), repeat=self.N)
        self.S = lambda signs : 0.5*np.sum(signs*self.beta_minus_alpha)
        self.tilde_s = lambda signs : np.prod(signs)

        # Compute the f(y) prefactor once
        self.prefactor = (-1)**self.N/( factorial(self.N - 1) * self.tilde_a)

        # Use quad repeatedly to approximate the CDF on a dense domain
        dom = np.linspace(np.min(self.intervals[:,0]), np.max(self.intervals[:,1]), 1000)
        cumt = cumtrapz([self.f(y) for y in dom], x=dom, initial=0.0) # cumt = np.array([quad(self.h, 0, w)[0] for w in dom])

        # Use interpolation to create the point percent function for rapid variates
        self.internal_cdf = interp1d(dom, cumt, bounds_error=False, fill_value=(0.0,1.0))
        self.internal_ppf = interp1d(cumt, dom)
        
    # Define the Heaviside
    def theta(self, w, w_c):
        return np.where(w > w_c, 1,
                        np.where(w < w_c, 0, 0.5))
            
    # Define the Parsi-Brosh f(y) PDF
    def f(self, y):
        return self.prefactor * np.sum([ self.tilde_s(signs) * (y - self.mu - S(signs))**(self.N - 1) * self.theta(y - self.mu - S(signs)) for signs in self.sign_sets])
        
    def _pdf(self, y):
        return self.f(y)

    def _cdf(self, y):
        return self.internal_cdf(y)

    def _ppf(self, P):
        return self.internal_ppf(P)
 
import matplotlib.pyplot as plt
# # Driver stub
# test = luskwright(intervals=[(0,1), (0,2)])
# dom = np.linspace(0, 4, 100)
# print(test._pdf(dom))
# plt.plot(dom, test._pdf(dom))
# plt.show()

# # Regenerates Figure 1 of Lusk and Wright

# Now lets try Duncan's thing for the BH mass biases
test = parsibrosh_unweighted(intervals=[(-0.4, -0.3),
                                        (-0.3, -0.1),
                                        (0.0, 0.3),
                                        (0.0, 0.3)])

dom = np.linspace(-0.8, 0.4, 100)
print(test._pdf(dom))
print(test._cdf(dom))

# Get some variates
draws = test.rvs(size=10000)
mu = np.mean(draws)
variance = np.var(draws)
plt.hist(draws, bins=40, density=True)

plt.plot(dom, test._pdf(dom))

plt.plot(dom, np.exp(-(dom - mu)**2/(2*variance))/(np.sqrt(2*np.pi*variance)))
plt.yscale('log')
plt.show()






