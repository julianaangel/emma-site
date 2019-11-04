#! /usr/bin/env python
#
# A class to read and understand and parse AME2016 tables
#   "The Ame2016 atomic mass evaluation (I)"
#    M.Wang, G.Audi, F.G.Kondev, W.J.Huang, S.Naimi and X.Xu
#


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re #for string manipulation
import time
from django import forms

class MassTableParser:

    def __init__(self, mass_filename):
        
        #FORTRAN format:  a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5\
        #    ,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.5
        widths= [1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11]
        names = ['cc','N-Z','N','Z','A','_1','El','origin','_2','MassExcess', 'MassExcess_Uncert',\
                 'BE_over_A', 'BE_over_A_Uncert', '_3', 'Beta_mode', 'E_beta', 'E_beta_uncert',\
                 '_4', 'Mass_u', '_5', 'Mass_u_decimal', 'Mass_u_Uncert' ]
        header = 38        
        
        col_width = [[0,widths[0]]]  #FIRST INTERVAL
        for i in range(1, len(widths)): #FOR EACH SUBSEQUENT INTERVAL
            lower = col_width[-1][1]  #GRAB THE LAST UPPER INTERVAL, 
            upper = lower + widths[i] #GO FOR THE NEXT WIDTH
            col_width.append([lower,upper])
        
        self.__mass_table = pd.read_fwf(mass_filename, colspecs=col_width,header=header, names=names)


    #Quick and neat way to access specific values based upon A,Z and the label up in names[]
    def __get(self, A, Z, label):
        N = A - Z
        lst = list(self.__mass_table[(self.__mass_table.N == N) & (self.__mass_table.Z == Z)].to_dict()[label].values())
        if len(lst) == 0:
            raise forms.ValidationError("Can not find isotope data: are you sure those are the correct A and Z values ?")
        return lst[0]


    #special helper function to concatenate mass which is separated in the table
    def __retrieve_mass_u(self, A, Z):
        mass_u_1 =  str(self.__get(A,Z, 'Mass_u'))  #Bulk of the mass in u, usually ~A
        mass_u_2 =  str(self.__get(A,Z, 'Mass_u_decimal'))  #remainder of the mass in micro-u
        #A '#' is used to indicate caluclated vs
        if(mass_u_2[-1] == '#'):
            mass_u_2 = mass_u_2[:-1]



        
        return float(mass_u_1 + mass_u_2) * 1E-6
        
    #Returns the mass excess value from the Dataform
    def mass_excess(self, A, Z):
        mass_excess = self.__get(A, Z, 'MassExcess')

        #A '#' is used to indicate caluclated vs
        if(isinstance(mass_excess, str) and mass_excess[-1] == '#'):
            mass_excess = mass_excess[:-1]
        
        return float(mass_excess)

    #The atomic mass, M_Nuclide + Z*M_e, in u  
    def atomic_mass(self, A, Z):
        return self.__retrieve_mass_u(A,Z)

    #The beta energy, if there is one
    def beta_energy(self, A, Z):
        E_beta =  self.__get(A,Z,'E_beta')

        if (isinstance(E_beta, str)):
            #If we have a '#', remove it
            if ( E_beta[-1] == '#'):
                E_beta = E_beta[:-1]

            if ( '*' in E_beta) :
                E_beta = 0.

        return float(E_beta)




                    
