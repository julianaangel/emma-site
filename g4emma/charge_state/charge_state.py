
#!/usr/bin/env python
#
#  A simple script to test the accuracy of a couple different charge state
#  calculations 

import matplotlib.pyplot as plt
import numpy as np
#import g4emma.charge_state.nuclear_tools as nr
from g4emma.charge_state import nuclear_tools as nr

import time

#Units based library for conversion and convenience
from g4emma.charge_state.nuclear_tools import U

#Define a mole in terms of particle count
#U.define('particle_num = ( 1. / 6.022140857E23 ) * mol = num')
#a_mole = ((1*U.mol).to('num') / U.mol) 


#Nuclear Constants
e2 = 1.44*U.MeV * U.fm  #e^2 / 4*pi*epsilon_0
r_0 = 1.2*U.fm


#src: Nikolaev and Dmitriev, Physics Letters A, 28(A), p277-278 (1968)
def ND_q(A_beam, Z_beam, Z_target, E_beam, units=True, include_uncertainty=False):
    A_beam, Z_beam, Z_target = 1.*A_beam, 1.*Z_beam, 1.*Z_target
    
    #v_prime = 0.013.6E8 * U.cm / U.s  #const
    v_prime = 0.012 * U.c  #Useful to recast v/vprime as beta/0.012

    #Check to make sure it doesn't already have units
    if (type(E_beam) != type(1.*U.MeV) and units):
        E_beam = E_beam*U.MeV

    m=nr.mass_nuclear(A_beam, Z_beam)

    #Get mean velocity
    v = np.sqrt(2. * E_beam / m)
    
    #alpha = 0.45, k =0.6, v_prime is above
    q_centroid =  Z_beam*((1. + (v / ((Z_beam**0.45) *v_prime))**(-1./0.6))**(-0.6))

    q_sigma = 0.5*(np.sqrt(q_centroid*(1-((q_centroid / Z_beam)**(1./0.6)) ) ))

    if include_uncertainty:
        #taken from paper, ~5% for Z>20
        q_centroid_uncertainty = 0.05*q_centroid
        q_sigma_uncertainty = 0.2*q_centroid
    
    
    return q_centroid.m, q_sigma.m


    
#src: Shima, Ishihara, Mikumo NIM 200 (1982) 605-608
#     Shima, Ishihara, Mikumo NIM B 2 (1984) 222-226
#By it's own admission, valid for Z_beam >7, Z_targ >3, E<6 MeV/u
# and when v_ND > 1.45 (for sigma_q)
def Shima_q(A_beam, Z_beam, Z_target, E_beam, units=True):
    A_beam, Z_beam, Z_target = 1.*A_beam, 1.*Z_beam, 1.*Z_target
    
    #v_prime = 3.6E8 * U.cm / U.s  #const
    v_prime = 0.012 * U.c

    #Check to make sure it doesn't already have units
    if (type(E_beam) != type(1.*U.MeV) and units):
        E_beam = E_beam*U.MeV

    m=nr.mass_nuclear(A_beam, Z_beam)
    v = (np.sqrt(2. * E_beam / m))

    #Get reduced ion velocity as defined by ND (see above)
    v_ND = (v / v_prime) / (Z_beam**0.45)

    #Semi-empirical expansion of ND formula out to v**3 for for Z_targ=6
    q_centroid = Z_beam*(1-np.exp(-1.25*v_ND + 0.32*v_ND**2. - 0.11*v_ND**3.))

    #Now the average chargestate is shifted based upon Z_targ !=6
    q_centroid = q_centroid * (1 - 0.0019*(Z_target - 6)*np.sqrt(v_ND) + 1E-5*v_ND*(Z_target - 6)**2)

    #Found as an aside in NIM B (1984) 222-226
    q_sigma = (Z_beam**0.27)*(0.76 - 0.16*v_ND)

    return q_centroid.m, q_sigma.m
    
#src: Schiwietz, Grande NIM B 175-177 (2001) 125-131
def Schiwietz_q(A_beam, Z_beam, Z_target, E_beam, units=True):

    A_beam, Z_beam, Z_target = 1.*A_beam, 1.*Z_beam, 1.*Z_target
    
    #Bohr velocity, v_0 = e**2 / hbar == c / alpha , where alpha = fine struct const
    v_0 = 1.*U.c / 137.035999  
    
    #Check to make sure it doesn't already have units
    if ( not (nr.has_units(E_beam)) and units):
        E_beam = E_beam*U.MeV

    m=nr.mass_nuclear(A_beam, Z_beam)

    #Get mean velocity
    #Note, needed to convert here cause the next step had trouble with units of c
    v = np.sqrt(2. * E_beam / m)

    #least squares reduced velocity param
    #x = ( (v / v_0) * (Z_beam**-0.52) * (Z_target**(-0.017*(Z_beam**-0.52)*(v/v_0)))  )**(1+(0.4/Z_beam))  #gas target
    x = ( (v / (1.68 *v_0)) * (Z_beam**-0.52) * (Z_target**(-0.019*(Z_beam**-0.52)*(v/v_0)))  )**(1+(1.8/Z_beam))

    q_centroid = Z_beam * ((12*x + x**4.)/ ( (0.07/x) + 6 + 0.3*(x**0.5) + 10.37*x + x**4.)) 


    #I chose 'n' even though they use 'x' to prevent conflict with the reduced velocity param 'x'
    def f(n):
        #Including abs for near fully stripped (where we end up with sqrt( -val )
        #Seems appropriate given initial definition of d in ND paper and how it'b being used in Schiwietz
        # Added April 18 2019
        return np.sqrt(np.abs(1 + (0.37*Z_beam**0.6) / n))

    #getting d == q_sigma, from their defined 'w', a conserved width
    q_sigma =  0.7 /( (Z_beam ** -0.27) * (Z_target ** (0.035 - 0.0009*Z_beam)) * f(q_centroid) * f(Z_beam - q_centroid) )

    return q_centroid.m, q_sigma.m


#Pretty useless for low energies. no knowledge of Z_target
#Crazy overfit....
def Winger_q(A_beam, Z_beam, Z_target, E_beam, units=True):
    #Check to make sure it doesn't already have units
    if (type(E_beam) != type(1.*U.MeV) and units):
        E_beam = E_beam*U.MeV
    
    m=nr.mass_nuclear(A_beam, Z_beam)

    #Get mean velocity
    #Note, needed to convert here cause the next step had trouble with units of c
    v = np.sqrt(2. * E_beam / m)

    
    
    v_ND = (v / U.c).to_base_units() / (0.012 * Z_beam**0.45)
    Z = np.log(Z_beam)  #reduced Z
    X = np.log(v_ND)
    
    alpha_0 = 0.4662
    alpha_1 = -0.5491 * np.exp( 0.7028*Z - 0.1089*(Z**2.) + 1.644E-3*(Z**3.) - 0.5155*Z + 0.05633*(Z**2.))
    alpha_2 = 5.447E-3 * np.exp( 0.8795 *Z - 1.091 *Z)
    alpha_3 = -8.26E-4 * np.exp( 2.848 * (Z**7.) - 0.2442*(Z**9.))
    alpha_4 = -9.239E-5

    q_avg = Z_beam * ( 1 - np.exp( alpha_0 + alpha_1 * v_ND + alpha_2 * v_ND**2. + alpha_3 * v_ND**3. + alpha_4 * v_ND**4.))


    beta_0  = 0.7254 + 0.3961*Z + 0.1215*(Z**2.)
    beta_1  = 3.114 - 1.827*Z
    beta_2  = -4.142 + 2.010*Z -0.1678*(Z**2.)
    gamma_0 = 0.2137 - 0.3749*Y + 0.1283*(Y**2.)
    gamma_1 = 1.114 * np.exp( -0.04798 * Y + 0.09366 *Z)
    gamma_2 = 0.1751
    
    q_sigma = np.exp( beta_0 + beta_1 * Y + beta_2 * (Y**2.)) * ( 1 - np.exp (gamma_0 + gamma_1*Z + gamma_2*(Z**2.)))

    return 'lol'
    #return q_avg, q_sigma

    
#Returns precentage of a beam that'll charge change for a given residual pressure
# for some length of ion path 
def charge_change_in_vac(E_beam, A_beam, Z_beam, q_avg, P, dist, units=True):

    #Check to make sure it doesn't already have units
    if (type(dist) != type(1.*U.cm) and units):
        dist = dist*U.cm  #default distance is cm

    if (type(P) != type(1.*U.torr) and units):
        P = P*U.torr  #default distance is cm

    #Finds number of gas molecules in the given volume, ala PV=nRT
    def iaction(dx, cs, P = 5E-8*U.torr, T=(15.0+273.15)*U.degK ):
        #if we've not labelled 
        if (type(cs) != type(1.*U.cm**2) and units):
            dx = dx*U.torr #default P is torr

        if (type(P) != type(1.*U.torr) and units):
            dx = dx*U.torr #default P is torr

        if (type(T) != type(1.*U.degK) and units):
            dx = dx*U.degK #default T is C
    
        #Ideal gas const, from wiki
        R = 62.363577 *U.torr * U.L / (U.mol *  U.degK)

        N = P*cs*dx / (R*T)
        N = N.to_base_units()

        return (N.to('num'))

    #from Hseuh, IEEE vol. NS-32, No. 5, October 1985
    #Double checked with experimental results from Franzke IEEE Vol. NS-28, No. 3, June 1981
    #Note, the 'fitted' params in this formulation are for N2
    def cs(q, v):
        if (type(v) != type(1.*U.m / U.s) and units):
            v = v*U.m/U.s #default v is m / s

        #get the reduced velocity
        beta = (v / U.c).to_base_units()
        
        sigma_loss = 9E-19 * q**(-2./5.)* beta**(-2.)*U.cm**2
        sigma_capture = 0#3E-28* q**(2.5)* beta**(-7.)*U.cm**2

        return (sigma_loss + sigma_capture) 

    #Bohr-Linhard formulation, from Betz RevModPhys 1972
    #Found to be not that great, overestimates by 2 orders of magnitude energy charge-changing collisions
    #def cs_2(v,Z,Z_t, q):
    #
    #    #Bohr radius, h_bar**2 / (m_e e**2), val from wiki
    #    a_0 = 5.2917721067E-9 * U.cm
    #
    #    #Bohr velocity, e**2 / h_bar = c/ alpha (fine struct const)
    #    v_0 = 1.*U.c / 137.035999
    #    
    #    sigma_loss = np.pi* (a_0**2.) * (Z_t**(2./3.)) * (Z ** (4./3.)) * q**(-3.) *((v / v_0)**2.)
    #    sigma_capture = np.pi* (a_0**2.) * (Z_t**(1./3.)) * q**(2.) * ((v_0 / v)** 3.)
    #
    #    return (sigma_loss + sigma_capture)

    m=nr.mass_nuclear(A_beam, Z_beam)

    #Get mean velocity
    v = np.sqrt(2. * E_beam / m)

    a_cs = cs(q_avg, v)
    
    return (1. - np.exp(-1.*iaction(dist, a_cs, P=P).m))

############################################################################################
############################################################################################
############################################################################################
#  Updated charge state predictions with associated errors
#
#
#  Should integrate into the above functions when I have time
#
#





#src: Nikolaev and Dmitriev, Physics Letters A, 28(A), p277-278 (1968)
def ND_q2(A_beam, Z_beam, Z_target, E_beam, units=True, include_uncert=False):
    A_beam, Z_beam, Z_target = 1.*A_beam, 1.*Z_beam, 1.*Z_target
    
    #v_prime = 0.013.6E8 * U.cm / U.s  #const
    v_prime = 0.012 * U.c  #Useful to recast v/vprime as beta/0.012

    #Check to make sure it doesn't already have units
    if (type(E_beam) != type(1.*U.MeV) and units):
        E_beam = E_beam*U.MeV

    m=nr.mass_nuclear(A_beam, Z_beam)

    #Get mean velocity
    v = np.sqrt(2. * E_beam / m)
    
    #alpha = 0.45, k =0.6, v_prime is above
    q_centroid =  Z_beam*((1. + (v / ((Z_beam**0.45) *v_prime))**(-1./0.6))**(-0.6))

    def q_sigma(a_q):
        return 0.5*(np.sqrt(a_q*(1-((a_q / Z_beam)**(1./0.6)) ) ))

    if include_uncert:
        #assert (Z_beam >= 20) #Should include this in the final code
        #taken from paper, ~5% for Z>20 for q, ~20% for sigma (not used cause we get 
        #NOTE:  Schiwietz et al (NIM B 2001) report it as 3.3% / Z_beam, though that's not present in the work
        q_uncert = 0.05*q_centroid
        
        q_sigma_lo =  q_sigma(q_centroid.m - q_uncert.m)
        q_sigma_hi =  q_sigma(q_centroid.m + q_uncert.m)
                
        return [q_centroid.m, q_centroid.m - q_uncert.m, q_centroid.m + q_uncert.m],\
               [q_sigma(q_centroid.m), q_sigma_lo, q_sigma_hi ], q_uncert
    
    
    return q_centroid.m, q_sigma(q_centroid.m)

#src: Shima, Ishihara, Mikumo NIM 200 (1982) 605-608
#     Shima, Ishihara, Mikumo NIM B 2 (1984) 222-226
#By it's own admission, valid for Z_beam >7, Z_targ >3, E<6 MeV/u
# and when v_ND > 1.45 (for sigma_q)
def Shima_q2(A_beam, Z_beam, Z_target, E_beam, units=True, include_uncert=False):
    A_beam, Z_beam, Z_target = 1.*A_beam, 1.*Z_beam, 1.*Z_target
    
    #v_prime = 3.6E8 * U.cm / U.s  #const
    v_prime = 0.012 * U.c

    #Check to make sure it doesn't already have units
    if (type(E_beam) != type(1.*U.MeV) and units):
        E_beam = E_beam*U.MeV

    m=nr.mass_nuclear(A_beam, Z_beam)
    v = (np.sqrt(2. * E_beam / m))

    #Get reduced ion velocity as defined by ND (see above)
    v_ND = (v / v_prime) / (Z_beam**0.45)

    #Semi-empirical expansion of ND formula out to v**3 for for Z_targ=6
    q_centroid = Z_beam*(1-np.exp(-1.25*v_ND + 0.32*v_ND**2. - 0.11*v_ND**3.))

    #Now the average chargestate is shifted based upon Z_targ !=6
    q_centroid = q_centroid * (1 - 0.0019*(Z_target - 6)*np.sqrt(v_ND) + 1E-5*v_ND*(Z_target - 6)**2)

    #Found as an aside in NIM B (1984) 222-226
    #Only holds for Z_target - q_cent == n_e < 4 
    q_sigma = (Z_beam**0.27)*(0.76 - 0.16*v_ND)
    
    if include_uncert:
        #They're pretty specific about their applicable ranges:
        #  E_beam <= 6 MeV/u
        assert (Z_beam >= 8) and (4 <= Z_target <= 79) and (E_beam < A_beam * 6 *U.MeV)
        
        #Taken from Shima 1983 Phys Rev A V28, 4
        q_uncert = Z_beam * 0.03 # Though Shima 1982 says < 4%, 1983 says == 3%
        
        q_sigma_lo =  q_sigma.m
        q_sigma_hi =  q_sigma.m
        return [q_centroid.m, q_centroid.m - q_uncert, q_centroid.m+q_uncert], \
               [q_sigma.m, q_sigma_lo, q_sigma_hi], q_uncert
    
    return q_centroid.m, q_sigma.m


#src: Schiwietz, Grande NIM B 175-177 (2001) 125-131
def Schiwietz_q2(A_beam, Z_beam, Z_target, E_beam, include_uncert=False, units=True):

    A_beam, Z_beam, Z_target = 1.*A_beam, 1.*Z_beam, 1.*Z_target
    
    #Bohr velocity, v_0 = e**2 / hbar == c / alpha , where alpha = fine struct const
    v_0 = 1.*U.c / 137.035999  
    
    #Check to make sure it doesn't already have units
    if ( not (nr.has_units(E_beam)) and units):
        E_beam = E_beam*U.MeV

    m=nr.mass_nuclear(A_beam, Z_beam)

    #Get mean velocity
    #Note, needed to convert here cause the next step had trouble with units of c
    v = np.sqrt(2. * E_beam / m)

    #least squares reduced velocity param
    #x = ( (v / v_0) * (Z_beam**-0.52) * (Z_target**(-0.017*(Z_beam**-0.52)*(v/v_0)))  )**(1+(0.4/Z_beam))  #gas target
    x = ( (v / (1.68 *v_0)) * (Z_beam**-0.52) * (Z_target**(-0.019*(Z_beam**-0.52)*(v/v_0)))  )**(1+(1.8/Z_beam))

    q_centroid = Z_beam * ((12*x + x**4.)/ ( (0.07/x) + 6 + 0.3*(x**0.5) + 10.37*x + x**4.)) 

    #
    def q_sigma(a_q):
        #I chose 'n' even though they use 'x' to prevent conflict with the reduced velocity param 'x'
        def f(n):
            #Including abs for near fully stripped (where we end up with sqrt( -val )
            #Seems appropriate given initial definition of d in ND paper and how it'b being used in Schiwietz
            # Added April 18 2019
            return np.sqrt(np.abs(1 + (0.37*Z_beam**0.6) / n))
        
        #getting d == q_sigma, from their defined 'w', a conserved width
        q_sigma =  0.7 /( (Z_beam ** -0.27) * (Z_target ** (0.035 - 0.0009*Z_beam)) * f(a_q) * f(Z_beam - a_q) )
        return q_sigma
    
    if include_uncert:
        #Though absolute uncertainty is 0.54, using the relative uncertainties from the paper
        q_uncert = Z_beam * 0.023 # 2.3%
        
        if Z_beam <= 2: #for He, H only (unlike ND, which is up to Z=10)
            q_uncert = .1 * Z_beam #10%  
        
        q_sigma_lo =  q_sigma(q_centroid.m - q_uncert)
        q_sigma_hi =  q_sigma(q_centroid.m + q_uncert)
        return [q_centroid.m, q_centroid.m - q_uncert, q_centroid.m+q_uncert], \
               [q_sigma(q_centroid.m), q_sigma_lo, q_sigma_hi], q_uncert
    
    
    return q_centroid.m, q_sigma(q_centroid.m)




