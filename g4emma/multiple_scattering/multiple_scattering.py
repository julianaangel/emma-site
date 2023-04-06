import numpy as np
from scipy.optimize import fsolve
import scipy.integrate as integrate
import scipy.special as special

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.stats as stats


class model_parameters:
# variables and helper functions for the quantum derivation from Marion and Zimmerman 

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
    
    
class model_parameters_classical:
# parameters and helper functions for classical derivation, from Sigmund and Winterbon

    def __init__(self, E, z, Z, Mc2t, t):
        self.z = z # incident particle z
        self.Z = Z # target z
        self.Mc2t = Mc2t  # target particle rest mass (in g) - this is technically used to calculate
                                # the number of scattering centers (nuclei) per unit volume
        self.t = t  # target areal density  (in g/cm^2)
        self.E = E  #incident particle energy in MeV

        self.a_ = self.a()
        self.Nt_ = self.Nt()
        self.tau_ = self.tau()
        self.alpha_1_tf_ = self.alpha(self.alpha_tilde_1_tf())
        self.alpha_1_lj_ = self.alpha(self.alpha_tilde_1_lj())
        self.alpha_2_ = self.alpha(self.alpha_tilde_2())
        self.alpha_3_ = self.alpha(self.alpha_tilde_3())
        self.alpha_4_ = self.alpha(self.alpha_tilde_4())
        self.weighted_alpha_ = self.weighted_alpha()

    def a(self):
        a = (0.885*0.529e-8)/(self.z**(2/3)+self.Z**(2/3))**0.5
        return a

    def Nt(self):
        Nt = self.t/self.Mc2t
        return Nt

    def tau(self): 
        tau = np.pi*self.Nt()*self.a()**2
        return tau

    # different alpha_tilde functions for different tau domains ------------------#
    def alpha_tilde_1_tf(self): #thomas-fermi screening for tau < 0.1
        Cm = 1.05
        m = 0.311
        alpha_tilde = Cm*self.tau_**(1/(2*m))
        return alpha_tilde

    def alpha_tilde_1_lj(self): #lenz-jensen screening for tau < 0.1
        Cm = 3.45
        m = 0.191
        alpha_tilde = Cm*self.tau_**(1/(2*m))
        return alpha_tilde
    
    def alpha_tilde_2(self): #for 1 < tau < 5
        Cm = 0.25
        m = 0.5
        alpha_tilde = Cm*self.tau_**(1/(2*m))
        return alpha_tilde
    
    def alpha_tilde_3(self): # for 40 < tau < 500
        Cm = 0.92
        m = 0.56
        alpha_tilde = Cm*self.tau_**(m)
        return alpha_tilde
    
    def alpha_tilde_4(self): # for tau > 1000
        Cm = 1.00
        m = 0.55
        alpha_tilde = Cm*self.tau_**(m)
        return alpha_tilde
    #-----------------------------------------------------------------------------#

    def alpha(self, alpha_tilde):
        a = self.a()*1e13  #convert a (units of cm) to units of fm
        alpha = alpha_tilde*(2*self.z*self.Z*1.4399764)/(self.E*a) # elementary charge e^2 is equal to 1.4399764 MeV*fm
        return alpha
    
    def weighted_alpha(self):
        tau = self.tau_
        x = np.log10(self.tau_)
        if tau <= 0.1: 
            w_alpha_tf = self.alpha_1_tf_
            w_alpha_lj = self.alpha_1_lj_
            return w_alpha_tf, w_alpha_lj
        elif tau > 0.1 and tau <= 1:
            w_alpha_tf = (np.log10(1)-x)*self.alpha_1_tf_+(x-np.log10(0.1))*self.alpha_2_
            w_alpha_lj = (np.log10(1)-x)*self.alpha_1_lj_+(x-np.log10(0.1))*self.alpha_2_
            return w_alpha_tf, w_alpha_lj
        elif tau > 1 and tau <= 5: 
            w_alpha = self.alpha_2_
            return w_alpha, 0.0
        elif tau > 5 and tau <= 40: 
            d = np.log10(40) - np.log10(5)
            w_alpha = (np.log10(40)-x)*self.alpha_2_+(x-np.log10(5))*self.alpha_3_
            w_alpha = w_alpha/d
            return w_alpha, 0.0 
        elif tau > 40 and tau <= 500:
            w_alpha = self.alpha_3_
            return w_alpha, 0.0
        elif tau > 500 and tau <=1000: 
            d = np.log10(1000) - np.log10(500)
            w_alpha = (np.log10(1000)-x)*self.alpha_3_+(x-np.log10(500))*self.alpha_4_
            w_alpha = w_alpha/d
            return w_alpha, 0.0 
        elif tau > 1000: 
            w_alpha = self.alpha_4_ 
            return w_alpha, 0.0
        
    
    def ratio(self):
        a = self.a()*1e13
        r = (self.E*a)/(2*self.z*self.Z*1.4399764)
        return r
    
#------------#------------#------------#------------#------------#------------#------------#
# Integration Functions for quantum method
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
def set_parameters_marion(E, Mc2, z, Z, A, t):
    
    params = model_parameters(E, Mc2, z, Z, A, t)
    
    return params

def set_parameters_sigmund(E, z, Z, Mc2t, t):
  
    params = model_parameters_classical(E,z,Z,Mc2t,t)
    
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



