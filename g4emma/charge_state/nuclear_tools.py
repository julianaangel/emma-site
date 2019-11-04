#!/usr/bin/env python
#
#
# A script to perform various tasks for some nuclear reactions.  Currently:
#      -Q-value calculations
#      -V_bass calculation
#      -V_coul calculation
#      -E_beam --> E_excitation


import numpy as np
import matplotlib.pyplot as plt

import time
import os

#from scipy.stats import rv_continuous
#class gaussian_angle(rv_continuous):
#    #
#    def _pdf(self, x, sigma):
#        return np.exp(-x**2. / (2.* sigma ) / (npsqrt( 2.0 *np.i) * np.sin(x) * sigma**2) )

    
#Units based library for conversion and convenience
from pint import UnitRegistry
U = UnitRegistry()

#Define a particle based upon the default count unit in pint
U.define('particle_num = ( 1 / 6.022140857E23 ) * mol = num')

#Use particle count to define beam current
unitless_nA = (1*U.nA).to(U.e / U.s).m
U.define('part_nanoamp  = ( '+ str(unitless_nA) +' ) * num / sec = pnA')
U.define('part_microamp = ( 1000. ) * pnA           = puA')
U.define('part_picoamp  = ( 1/ 1000. * pnA)                = ppA')  #as if we'll ever need this
U.define('part_milliamp = ( 1000. ) * puA           = pmA')  #as if we'll ever need this

a_mole = ((1*U.mol).to('num') / U.mol)


#Helper function to allow me to write clearer code
#TODO: actually implement it in all the older code
def has_units(obj):
    return type(obj) == type(1.0*U.g)


#Sub-functions
import g4emma.charge_state.charge_state as charge
#import cs_funcs as cs
#import emma_stats as EMMA

#Setup requires we get and initialize the AME2016 mass tables
import g4emma.charge_state.MassTableParser as MTP

script_dir = os.path.dirname(os.path.abspath(__file__)) #<-- absolute dir the script is in
masstable_file = '/mass16.txt'
AME_table = MTP.MassTableParser(script_dir + masstable_file)

#Nuclear Constants
e2 = 1.44*U.MeV * U.fm  #e^2 / 4*pi*epsilon_0
r_0 = 1.2*U.fm

#Weitzsacker equation params, from Loveland
a_V, a_S, a_Coul, a_sym, delta = 15.56*U.MeV, 17.23*U.MeV, 0.7*U.MeV, 23.285*U.MeV, 11*U.MeV


##NOTE, added neutron at beginning, now elements[Z] is correct mapping
elements=['n','H','He','Li','Be', 'B', 'C', 'N', 'O', 'F','Ne',\
             'Na','Mg','Al','Si', 'P', 'S','Cl','Ar', 'K','Ca',\
             'Sc','Ti', 'V','Cr','Mn','Fe','Co','Ni','Cu','Zn',\
             'Ga','Ge','As','Se','Br','Kr','Rb','Sr', 'Y','Zr',\
             'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',\
             'Sb','Te', 'I','Xe','Cs','Ba','La','Ce','Pr','Nd',\
             'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',\
             'Lu','Hf','Ta', 'W','Re','Os','Ir','Pt','Au','Hg',\
             'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',\
             'Pa', 'U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm',\
             'Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds',\
             'Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og','119','120',\
             '121','122','123','124','125','126','127','128']





##Returns the Q-value of a reaction.  Remember, convention hold Q = Rxt - Prod
## meaning positive value == energy released / negative val == energy required
##  NOTE -- this is only for fusion.  to get other reactions, need to specify exit channel
#def Q_value(A1, Z1, A2, Z2, A_p1=None, Z_p1=None, A_p2=None, Z_p2=None, units=True):
##    A1, Z1, A2, Z2 = int(A1), int(Z1), int(A2), int(Z2)
#
#    unit = 1.0
#    if units:
#        unit = unit *U.keV
#
#    if A_p1 == None:
#        return ((AME_table.mass_excess(A1, Z1) + AME_table.mass_excess(A2, Z2)\
#                - AME_table.mass_excess(A1+A2, Z1+Z2) ) * unit )
#
#    return ((AME_table.mass_excess(A1, Z1) + AME_table.mass_excess(A2, Z2) \
#             - AME_table.mass_excess(A_p1, Z_p1) - AME_table.mass_excess(A_p2, Z_p2) ) *unit)
#


#def prod_rate(cross_section, target_thickness, I_beam, target_density, target_molar_mass,):
#    #Target thickness
#    target = 500 * (U.ug / U.cm**2)
#    density = 11.7 * (U.g / U.cm**3) #Pb   #9.78 * (U.g / U.cm**3) #Bi, from wikipedia
#    molar_mass = 232 * (U.g / U.mol)  
#
#    #Cross section
#    sigma  = 280*U.nb
#
#    ##Particle beam current
#    I_beam = (1*U.uA).to(U.e / U.s)
#    I_beam = I_beam * (1*U.num / (1*U.e))  #Particle charge conversion already done
#
#    I_beam = 5E9 * (U.num / U.s)
#
#
#    ##Electrical beam current
#    ##Direct beam intensity
#    #I_beam = 1.20E5 * (U.num / U.s)  # particles / second
#
#    #Rate is R = sigma*target_thickness * I_beam
#    #Remove 1 U.num, as target &beam both have U.num, but combines to single particle
#    rate_prod = cross_section * ( target_thickness / molar_mass * a_mole ) * I_beam / U.num  
#
#    rate_prod.ito_base_units()
#
#    #Convert to particles
#    rate_prod = rate_prod * a_mole
#    return 








#Returns the Q-value of a reaction.  Remember, convention hold Q = Rxt - Prod
# meaning positive value == energy released / negative val == energy required
def V_coul(A1, Z1, A2, Z2, units=True):
    V = (e2 * Z1 * Z2 / (r_0 * ( ((A1)**(1./3.)) + ((A2)**(1./3.)) ) ) )

    if units:
        return V
    return V.magnitude
    

#Returns the Bass Potential of two touching nuclei (ie, fusion)
# ref: Nuclear Physics A231 (1974) 45 - 63
def V_Bass(A1, Z1, A2, Z2, do_fusion=True, units=True):
    A1, Z1, A2, Z2 = float(A1), float(Z1), float(A2), float(Z2)
    
    
    #Bass eq fitted d_int, d to large dataset using this param, so to avoid doing a lot of work,
    #we just use this value here....

    r_0, d = 1.07, 1.35
    d_int = 2*d

    #Make local copy to edit within function
    a_e2, a_aS = e2, a_S
    
    
    #Put all params into units, or none into units
    if units:
        r_0, d, d_int = r_0*U.fm, d*U.fm, d_int*U.fm
    else:
        a_e2, a_aS = a_e2.magnitude, a_aS.magnitude
        


    #Dimensionless param, ratio of Coloumb to nuclear force
    x = a_e2 / (r_0 *a_aS) * (Z1 * Z2 / ( ((A1*A2)**(1./3.)) * ((A1**(1./3.)) + (A2**(1./3.))) ) )

    r_12 = r_0 * ( ((A1)**(1./3.)) + ((A2)**(1./3.)) )

    #Set up like this b/c it's the form from the paper, they pairwise it with d_int = d_fus
    if do_fusion:
        #d_int becomes d_fus
        d_int = d * (-1. * np.log(x) / (1 - 2*(d / r_12)))
    
    # From paper:
    #V_fus = Z1*Z2 *e^2/ r_12 * (  r_12 / (r_12+d_int) - 1/x * d/r_12 *exp(-d_int / d))

    return  (a_e2 * Z1 * Z2 / r_12)  *( (r_12/ (r_12+d_int)) - ( (d / (x * r_12)) *np.exp( -1.* d_int / d)))

#
#TODO:  incorporate exit channels, energy broadeningy due to exit channels
def E_CN(A_target, Z_target, A_beam, Z_beam, E_beam):
    if not has_units(E_beam):
        E_beam = E_beam*U.MeV
    return (1.*A_beam ) / (A_target +A_beam) * E_beam
    

def E_excitation(A_target, Z_target, A_beam, Z_beam, E_beam, units=True):
    A1, Z1, A2, Z2 = float(A_beam), float(Z_beam), float(A_target), float(Z_target)

    if not has_units(E_beam):
        E_beam = E_beam*U.MeV

    #Get into center of mass frame
    E_cm = (A2) / (A1 + A2) * E_beam
    
    Q = Q_value(A1, Z1, A2, Z2)

    #If we don't do units, make it MeV and remove any unit signifier
    if not units:
        Q = (Q.magnitude / 1000.)

    return (E_cm + Q)


def inverse_E_excitation(A_target, Z_target, A_beam, Z_beam, E_excitation, to_lab=True, units=True):
    A1, Z1, A2, Z2 = float(A_beam), float(Z_beam), float(A_target), float(Z_target)

    if not has_units(E_excitation):
        E_excitation = E_excitation*U.MeV

    Q = Q_value(A1, Z1, A2, Z2)

    E = E_excitation - Q #E_cm

    if to_lab: #Move it into the lab frame
        E = (A1 + A2) / A2 * E
    
    if units:
        return E 
    #Else return it sans units
    return E.m


#Gets physical binding energy of a given nuclei (from mass tables)
def BE(A,Z, over_A=True, units=True):
    BE = (float(A-Z) * AME_table.mass_excess(1,0) + float(Z) * AME_table.mass_excess(1,1) \
         - AME_table.mass_excess(A,Z)) *U.keV
    
    #We usually want it in MeV, and BE is negative cause it's binding
    BE = -1.0*BE.to(U.MeV)
    
    if not units:
        BE = BE.magnitude 
        
    if over_A:
        return BE / float(A)
    return (BE)


def mass_nuclear(A, Z, units=True):
    #Units from Wiki, unused ones kept here for completeness
    M_n = 939.5654133*U.MeV/((U.c)**2.)
    M_Z = 938.2720813*U.MeV/((U.c)**2.)
    M_u = 931.4940954*U.MeV/((U.c)**2.)
    M_e = .5109989461*U.MeV/((U.c)**2.)

    #get BE for A,Z
    BE_A = BE(A,Z, over_A=False) / ((U.c)**2.)

    #BE = [(A-Z)*M_n + Z*M_p - M_(A,Z) ]*c**2
    M = (A-Z)*M_n + Z*(M_Z) + BE_A  #Note, BE_A added b/c the -1 was included in BE() func

    if not units:
        return M.magnitude    
    return M

def mass_atomic(A,Z, units=True):
    M_u = 931.4940954*U.MeV/((U.c)**2.)

    return AME_table.atomic_mass(A,Z)*M_u
    

#Using Myers-Swiatecki formulation from 1967, as directly adapted from the one used by HIVAP
def BE_LDM67_HIVAP(A, Z, over_A=True, units=True):
    a_V, a_S, a_Coul, a_pair = 15.4941*U.MeV, 17.9439*U.MeV, 0.7053*U.MeV, 11.0*U.MeV
    a_CoulShape = 1.15303*U.MeV
    a_over_r0 = 0.444           #In MS 66, this value is fitted to 0.27
    kappa = 1.7826 #in HIVAP, it's refered to as GAMMA
    c, C = 0.325, 5.8*U.MeV     #in MS66, this value is c =0.27, C = 5.8 MeV
    M_N, M_Z = 8.07144*U.MeV, 7.28899*U.MeV
    
    N = A - Z
    delta = -1.0 -(2.0*(N/2) - N) - (2.0*(Z/2) - Z) #even-even -> +1, odd-odd -> -1, odd A -> 0
    
    A, Z, N = float(A), float(Z), float(N)

    asymmetry = (1. - kappa*((N-Z)/A)**2.)
    
    #Macro binding energy
                #Vol, Surface, Coulomb, Coulomb that corrects arbitrary shaped droplet diffuseness, pairing
    macro_BE = (-1.0*a_V*A + a_S*(A**(2./3.)) ) * asymmetry \
               + (a_Coul * (Z**2.) / (A**(1./3.))) \
               - (a_CoulShape * (Z**2.) / A) \
               +(delta* a_pair / (A ** 0.5))


    ### SHELL CORRECTION
    # NOTE:  Shell corrections have the form:
    # shell_corrections = E*theta**2 - F*theta**3 * cos(3*gamma) + S*exp(-theta**2)
    # where S is the gross shell corrections from a magic number step function
    # and E, F terms describe deformation as a function of the deformation magnitude theta
    # where theta = alpha/ alpha_0,  where alpha_0 is a/r0, a fitting parameter

    
    #Calculating 'S', 
    magic = np.array([0, 2, 8, 14, 28, 50, 82, 126, 184, 258], dtype='float')

    def Fval(X):
        #finds the magic number  by finding the zero-cross
        i =  np.where(np.diff(np.sign(magic - X )))[0][0]

        #NOTE- MS switches notation from q(n) to q_n at eq 5 vs the one following
        q = (3. / 5.) * ( (magic[i+1] ** (5./3.) ) - (magic[i]**(5./3.) ) ) / ( magic[i+1] - magic[i])

        return ( q * (X - magic[i]) - ((3./5.) * ( (X ** (5./3.)) - (magic[i] ** (5./3.)) ) ) )

    #Shell correction, from MyersSwiatecki 1965, Eq 5
    #NOTE: S(N,Z) = C * s(N,Z)    
    S = C * ((2.0 / A) **( 2. / 3.) * (  Fval(N) + Fval(Z)) - c*(A**(1./3.)))

    ### Calculating E, F

    #x == fissility == coul/2*surf
    fissility = (a_Coul * (Z**2.) / (A**(1./3.))) /  (2. * ( a_S*(A**(2./3.)) ) * asymmetry)
    
    S_crit = 2. * (a_S * asymmetry) * ((a_over_r0)**2.) * (1-fissility)   ##NOTE, HIVAP has this as EE


    ##I'm missing the connection between FF in HIVAP and F (eq 10) in MS66
    ##What follows is the HIVAP stuff
    F =  0.42591771  *(a_S * asymmetry) * ((a_over_r0)**3.) * (1. + 2.*fissility) / (A **(1./3.))

    
    tot_or_single = 1.  #Just make it a const if it's not overA
    if over_A:
        tot_or_single = 1. / A
        
 
    ##Calculating theta
    if ( (S_crit * (1. - 3.*(S / S_crit)) ) >= 0.0*U.MeV ):
        return tot_or_single*( macro_BE + S)

    #Else solve for theta using HIVAP's algorithm
    else:

        #if, while, we're trying to find suitable theta/theta_0 values,
        # we need to reset the search params, do it here
        def restart_theta_0():
            for i in [0.1*x for x in range(1,20)]:
                temp = (S_crit * (1. - ((1.5*F/S_crit)* i ) - (S / S_crit) * (3. - 2.*(i**2.)) * np.exp(-1. * i**2.)))
                if (temp.magnitude > 0.0):
                    return i
            return None

        #Find a converged value for theta given theta_0
        def get_theta(theta_0):
            a_theta = 0.0
            #While theta and theta_0 haven't converged to an accuracy of 0.0001 (ala HIVAP)
            while True:
                a_theta = theta_0 -(1.0 - (1.5* F / S_crit)*theta_0 - (S / S_crit)*(3 - 2.*(theta_0**2.))*np.exp(-1. * theta_0**2.))  / \
                        ( (-1.5*F / S_crit) + (S / S_crit)*(10. * theta_0 - 4.0 * (theta_0**3.))*(np.exp(-1. * theta_0 **2.)) ) 

#                print('a_theta: ' + str(round(a_theta, 4)) + '  -vs- theta_0: ' + str(round(theta_0,4)) \
#                          + '--- diff: ' + str(round( abs(a_theta - theta_0), 4) ))
                        
                #if they've reached the convergence condition
                if ( round( abs(a_theta - theta_0), 4) <= 0. ):
                    break
                elif (a_theta < 0.0):                 #if we've gone off the rails....
                    theta_0 = restart_theta_0()
                else:
                    theta_0 = a_theta
                
                
            return a_theta

        #Another hivap test....
        def is_good_theta(a_theta):

            exp_term = 0.0
            if (a_theta**2. < 25.0):
                exp_term = np.exp(-1.0 * a_theta**2.)
            
            temp = (2.0* S_crit * (1.0 - (2.0*(1.5 * F / S_crit) * a_theta )\
                                    -( ( S / S_crit ) * (3.0 - 12.0*(a_theta**2.)\
                                    + 4.0*(a_theta**4.)) * exp_term) ) )


            return (temp.magnitude > 0.0)


        #Okay, now that we've defined these funky test functions, lets actually run them
        theta = get_theta(1.0)         #guess start == 1.0

        #If we failed to find a good theta, try once to correct
        if not is_good_theta(theta):
            a_tetha_0 = restart_theta_0()
            theta = get_theta(a_tetha_0)

        #IF it's a good theta, then we can give the fully calculated BE
        if is_good_theta(theta):
            return  tot_or_single * ( macro_BE + S_crit*(theta**2.) - F*(theta**3.) + S*(1. - 2.*(theta**2.) )*(np.exp(-1.0*theta**2.))) 

        



#  Calculates a production rate for a given cs, beam, target_thickness
#  CS assumes millibarn if no units provided
#  Target thickness assumes ug/cm**2
#  Beam current assumes pnA 
def production_rate(a_cs, target_thickness, A_target, beam_current):
    #Cross section
    if not has_units(a_cs):
        a_cs = a_cs * (U.mb)   

    #Target
    if not has_units(target_thickness):
        target_thickness = target_thickness * (U.ug / U.cm**2)
    molar_mass = A_target * (U.g / U.mol)
    molar_mass = molar_mass.to(U.g/U.num)  #(Num nuclei in a gram)**-1

    #Beam
    if not has_units(beam_current):
        beam_current = beam_current*(U.nA)  #assumes pnA

    #Check if we have a current, converts to pps
    if (beam_current.to_base_units().u == U.A):
        beam_current = beam_current.to(U.e / U.s) * (1.*(U.num / U.e))

    #remove one of the U.num
    rate = (a_cs * (target_thickness / molar_mass) * beam_current) / U.num

    return (rate.to(U.num / U.s))


#  Does some easy, pretty printing of units
#
def print_unit(a_unit, decimals = 1, abbrev=True):
    #Extract and truncate the float
    if (a_unit.m > 0 ) and (np.abs(np.log10(a_unit.m)) >= decimals):
        form = 'e'
    else:
        form = 'f'
    
    num_str = ('{0:.'+ str(decimals) + form + '}').format(float(a_unit.m))
    
    if abbrev:
        unit = '{:~}'.format(a_unit.u).replace(' ','')
        #tack on the abbreviated unit
        return num_str + ' ' + unit

    #Tack on the full unit
    return num_str + ' ' +str(a_unit.u)





#def v_EVR(A_target, Z_target, A_beam, Z_beam, E_beam, n_evap=0):

#    E_CN = E_beam * (A_beam / (A_target + A_beam))
    
