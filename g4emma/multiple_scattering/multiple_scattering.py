import numpy as np
from scipy.optimize import fsolve
import scipy.integrate as integrate
import scipy.special as special

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.stats as stats


class model_parameters: 

    def __init__(self, E, Mc2, z, Z, A, t):
        self.E = E #incident energy (MeV)
        self.Mc2 = Mc2 #incident particle rest mass (MeV/c^2)
        self.z = z # incident particle z
        self.Z = Z # target z
        self.A = A # target A
        self.t = t # target areal density (g/cm^2)
        
        self.beta2_ = self.beta2()
        self.p2v2_ = self.p2v2()
        self.b_ = self.b()
        self.xc_ = self.xc()
        self.xc_x0_ = self.xc_x0()
        self.B_ = self.B()
        self.gamma_ = self.gamma()

    #------------#------------#------------#------------#------------#------------#------------#
    # Parameter Functions
    #------------#------------#------------#------------#------------#------------#------------#
    
    def beta2(self):
    
        a = 1 - ( 1 + (self.E/self.Mc2))**(-2)
    
        return a

    def p2v2(self):
    
        a = (self.E**2 +2*self.E*self.Mc2)*self.beta2()
    
        return a 


    def b(self): 
    
        a = np.log( 2730*(self.Z+1)*(self.Z**(1/3))*(self.z**2)*self.t / (self.A*self.beta2()) )- 0.1544
    
        return a
                   

    def xc(self):

        D = 0.1569*( (self.Z*(self.Z+1)*self.z**2 * self.t)/(self.A*self.p2v2() ) ) 
    
        return np.sqrt(D)     
    
    def xc_x0(self):
    
        a = 2730 * ((self.Z+1)*self.Z**(1/3)*self.z**2*self.t / (self.A*self.beta2()))
    
        return np.sqrt(a)

    def B(self):

        func = lambda B : B - np.log(B) - self.b() 

        initial_guess = 7

        a = fsolve(func, initial_guess)

        return float(a)
    
    # upper bound for the integrals
    def gamma(self):

        #exact form
        #gamma = np.sqrt(B)*((xc_x0) + 0.5*xc)

        #approximate form - makes almost no difference
        gamma = np.sqrt(self.B())*np.exp((self.B()-1.5)/2.2)

        return gamma 
    
    
#------------#------------#------------#------------#------------#------------#------------#
# Integration Functions
#------------#------------#------------#------------#------------#------------#------------#
    

def F0(x): 
    return 2*np.exp(-x**2)

def f1(u, x):
    return u**3 * special.jv(0,u*x) * np.log(0.25*u**2) * np.exp(-0.25*u**2) 

def F1(x, gamma):
    I = integrate.quad(f1, 0, gamma, args=(x))
    I = float(I[0]) * (1/4)
    return I

def f2(u, x):
    return u**5 * special.jv(0,u*x) * (np.log(0.25*u**4))**2 * np.exp(-0.25*u**2) 

def F2(x, gamma):
    I =  integrate.quad(f2, 0, gamma, args=(x))
    I = float(I[0]) * (1/16)
    return I

def F(x, xc, B, gamma):
    c = (1/(xc**2 * B))
    d = F0(x) + (1/B)*F1(x,gamma) + (1/(2*B**2))*F2(x,gamma)
    return float(c*d)



#------------#------------#------------#------------#------------#------------#------------#
# Evaluation
#------------#------------#------------#------------#------------#------------#------------#
    

# this will be called in the view function. The code in there will handle the rest
def set_parameters(E, Mc2, z, Z, A, t):
    
    params = model_parameters(E, Mc2, z, Z, A, t)
    
    return params

def gaussian_sigma(x, v2, params, rad=False):
    inter = interp1d(v2(x), x)
    a = 1/np.e
    pt = inter(a)
    sigma = pt/np.sqrt(2) #np.sqrt(-0.5 * pt**2 * np.log(a)) #
    sigma_rad = sigma*params.xc_*np.sqrt(params.B_)
    if rad == True:
        return sigma_rad
    else: 
        return sigma_rad*(180/np.pi)

def gaussian_FWHM(sigma):
    return 2*np.sqrt(2*np.log(2))*sigma



