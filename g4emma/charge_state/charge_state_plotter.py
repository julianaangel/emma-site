#! /usr/bin/env python
#

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import g4emma.charge_state.nuclear_tools as nr
from g4emma.charge_state.nuclear_tools import U

# normalized gauss function made for this purpose alone
def gauss(x, mean, sigma):
    # return A / (sigma * np.sqrt(2. * np.pi)) * np.exp(-1. * (x - mean)**2. / (2.* sigma**2) ) 
    return (1./(sigma * np.sqrt(2*np.pi))) * np.exp(-1. * (x - mean)**2. / (2.* sigma**2) )


#-------------------------------------------------------------
def plot_with_errors(qs, sigmas, q_uncert, comment, zb):#, i_color=0):
    q, q_lo, q_hi = qs
    sigma, sigma_lo, sigma_hi = sigmas
    
    #Place to build our "error range"
    lower, upper = [], []

#    x = np.arange(0, zb, 0.01)
### 220831 K.Pak modified ###
    x = np.arange(0, zb+0.01, 0.01)
### END ###
    for i in x:
        minus, plus = gauss(i, q_lo, sigma_lo), gauss(i, q_hi, sigma_hi)
        if minus < plus:
            lower.append(minus)
            upper.append(plus)
        else:
            lower.append(plus)
            upper.append(minus)
        if (i > (q - q_uncert)) and (i < (q + q_uncert)):
            #build a line between q_lo, and q_hi, as the height might be a bit different since sigma is a bit diff
            m, b =  np.polyfit( [q_lo, q, q_hi], [ gauss(q_lo, q_lo, sigma_lo),  gauss(q, q, sigma),  gauss(q_hi, q_hi, sigma_hi) ], 1)
            #modify the newly appended value
            upper[-1] = m*i + b + 0.005 #+ a little bit to make it look okay with the thicker linewidths in the graphs, not real!!!
            
            
    base_line = plt.plot(x, gauss(x, q, sigma), lw=0)
    color = base_line[0].get_color()

    
    #Plot errors (underneath the actual plot
    plt.plot(x, gauss(x, q_lo, sigma_lo), '--', color=color)
    plt.plot(x, gauss(x, q_hi, sigma_hi), '--', color=color)
    plt.fill_between(x, lower, upper, color=color, alpha=0.25)
    
    #plot the actual plot
    plt.plot(x, gauss(x, q, sigma), label = comment + ', q=' + str(round(q, 3)) + '$\pm$' + str(round(q_uncert, 3)) + ', $\sigma={' + str(round(sigma, 3)) + '}^{+' + str(round(sigma_hi - sigma, 4)) + '}_{-'+ str(round(sigma - sigma_lo, 4)) + '}$', lw=3, color=color )
    return x, gauss(x, q, sigma)
  
#--------------------------------------------
def generate_charge_state_plots(ab, zb, zt, eb):
    
    #A_beam, Z_beam = 16, 8
    #Z_target = 6 #CH2 target
    #E_beam = 1.1*A_beam *U.MeV # 2MeV/u
    
    A_beam, Z_beam = ab, zb
    Z_target = zt
    E_beam = eb*A_beam *U.MeV
    
    FIG = plt.figure(figsize=(28,25))
    plt.subplot(2,2,1)        
    qs_ND, sigmas_ND, q_uncert= nr.charge.ND_q2(A_beam, Z_beam, Z_target, E_beam, include_uncert=True)
    x_data, y_data_ND = plot_with_errors(qs_ND, sigmas_ND, q_uncert, 'ND', Z_beam)
    qs_Shima, sigmas_Shima, q_uncert= nr.charge.Shima_q2(A_beam, Z_beam, Z_target, E_beam, include_uncert=True)
    x_data, y_data_Shima = plot_with_errors(qs_Shima, sigmas_Shima, q_uncert, 'Shima', Z_beam)
    qs_Schiw, sigmas_Schiw, q_uncert= nr.charge.Schiwietz_q2(A_beam, Z_beam, Z_target, E_beam, include_uncert=True)
    x_data, y_data_Schiw = plot_with_errors(qs_Schiw, sigmas_Schiw, q_uncert, 'Schiwietz', Z_beam)
    plt.legend(fontsize=18,loc=2)
    plt.title(nr.elements[Z_beam] + '-' + str(A_beam) +' at ' + str(round(E_beam.m, 3)) + 'MeV' + ' through ' + nr.elements[Z_target] + ', all',fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('Charge State', fontsize=20)
    plt.ylabel('Charge State Fraction',fontsize=20)
    plt.ylim(-0.05, 0.6)

    plt.subplot(2,2,2)
    qs, sigmas, q_uncert= nr.charge.ND_q2(A_beam, Z_beam, Z_target, E_beam, include_uncert=True)
    plot_with_errors(qs, sigmas, q_uncert, 'ND', Z_beam)
    plt.legend(fontsize=18,loc=2)
    plt.title(nr.elements[Z_beam] + '-' + str(A_beam) +' at ' + str(round(E_beam.m, 3)) + 'MeV' + ' through ' + nr.elements[Z_target] + ', ND',fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('Charge State', fontsize=20)
    plt.ylabel('Charge State Fraction',fontsize=20)
    plt.ylim(-0.05, 0.6)

    plt.subplot(2,2,3)
    plt.plot(0,0) #quick, dirty way to maintain color of above
    qs, sigmas, q_uncert= nr.charge.Shima_q2(A_beam, Z_beam, Z_target, E_beam, include_uncert=True)
    plot_with_errors(qs, sigmas, q_uncert, 'Shima', Z_beam)
    plt.title(nr.elements[Z_beam] + '-' + str(A_beam) +' at ' + str(round(E_beam.m, 3)) + 'MeV' + ' through ' + nr.elements[Z_target] + ', Shima',fontsize=22)
    plt.legend(fontsize=18,loc=2)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('Charge State', fontsize=20)
    plt.ylabel('Charge State Fraction',fontsize=20)
    plt.ylim(-0.05, 0.6)

    plt.subplot(2,2,4)
    plt.plot(0,0) #quick, dirty way to maintain color of above
    plt.plot(0,0)
    qs, sigmas, q_uncert= nr.charge.Schiwietz_q2(A_beam, Z_beam, Z_target, E_beam, include_uncert=True)
    plot_with_errors(qs, sigmas, q_uncert, 'Schiwietz', Z_beam)
    plt.title(nr.elements[Z_beam] + '-' + str(A_beam) +' at ' + str(round(E_beam.m, 3)) + 'MeV' + ' through ' + nr.elements[Z_target] + ', Schiwietz',fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=18,loc=2)
    plt.xlabel('Charge State', fontsize=20)
    plt.ylabel('Charge State Fraction',fontsize=20)
    plt.ylim(-0.05, 0.6)

    # plt.suptitle(str(round(eb,3))+' MeV/u, ' + nr.elements[Z_beam] + '-' + str(A_beam), fontsize=24)
    plt.suptitle(str(round(eb,3))+' MeV/nucleon, ' + nr.elements[Z_beam] + '-' + str(A_beam), fontsize=24)    
    plt.close(FIG)
    return FIG, qs_ND, sigmas_ND, qs_Shima, sigmas_Shima, qs_Schiw, sigmas_Schiw, x_data, y_data_ND, y_data_Shima, y_data_Schiw 


