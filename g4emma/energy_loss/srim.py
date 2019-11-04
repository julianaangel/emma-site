#! /usr/bin/env python
#
# Provides an interface for running the SRIM module and manipulating the output
# 
#  Defaults:  srim_run = curr_dir/srim/
#             srim_run = curr_dir/srim_files/
#
#  Run with:
#   >  import srim
#   >  material = srim.Material()
#   >  material.make_target('sy')  -OR-  material.<specialized generator>
#   >  material.set_thickness(XXX*U.unit)
#   >  srim.energy_loss(A, Z, KE, material)
#   >  srim.exit_energy( ... )


#
# TODO fix all the broken shit in here, have it throw errors when having trouble parsing
#


import os, subprocess #interact with system, run processes
import re             #for string manipulation
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from django import forms
import g4emma.charge_state.nuclear_tools as nr
U = nr.U   #unit registry 


curr_dir = os.path.dirname(os.path.abspath(__file__)) #<-- absolute dir the script is inos.path.dirname(__file__)
SRIM_run_dir = curr_dir + os.sep + 'srim' + os.sep 
SRIM_out_dir = curr_dir + os.sep + 'srim_files' + os.sep


#Get the SRIM atomic table data
names = ['Z', 'atomic_symbol', 'name', 'mass_int', 'weight_most_abundant_isotope',\
         'weight', 'density', 'number_density', 'fermi_velocity', 'heat_sublimation',\
         'density_gas', 'number_density_gas']
widths= [2,3,15,4,8,8,8,8,6,6,9,9]


atomic_table = pd.read_fwf(SRIM_run_dir + 'atomic_data-SRIM.dat', widths=widths, names=names, skiprows=2, skipfooter=1)

#Organize by Z
atomic_table = atomic_table.set_index('Z')

#Default units to make sure we convert properly
default_E_unit = U.MeV
default_dist_unit = U.um


#Helper function to go from P to number density via PV=nRT
# Defaults to 15C for gas temp...
def gas_density(molar_mass, P, T=(273.15+15.)*U.degK, units=True):

    if (type(molar_mass) != type(1.*U.g / U.mole)):
        molar_mass = molar_mass*U.g / U.mL

    if (type(P) != type(1.*U.torr)):
        P = P * U.torr
    
    if (type(T) != type(1.*U.degK) and units):
        T = T*U.degK

    #Ideal gas const, from wiki
    R = 62.363577 *U.torr * U.L / (U.mol *  U.degK)

    # n/V = P / RT, converted from molar mass
    return(P / (R*T) *molar_mass).to(U.g / U.cm**3)




# A containter-type class to create and organize target compounds
#for SRIM input.  
class Material:
    def __init__(self):
        #Bulk qualities of the target
        self.name = ''
        self.density = -1.0* (U.g / U.cm**3) #nonsensical, needs to be modified for compounds
        self.correction = 1.0
        self.is_gas = False

         #Individual components of compound
        self.Z = [0]
#        self.mass = [0]
        self.stoich = [0]
        
        #NON SRIM quantity, to be set manually after building target material
        self.thickness = 0.0 * (U.ug / U.cm**2)

    #Builds a mono-element material, usually a target
    # Z_target : int of the Z of  the target element
    def set_element(self, Z_target, mass=None ):
        if (type(Z_target) == str):  #if we have a string, assume it's the element symbol
            Z_target = nr.elements.index(Z_target)  #should throw error if it's not contained in nr.elements
        
        self.Z = [Z_target]
#        if mass:
#            self.mass=[mass]
#        else:
#            self.mass = [ atomic_table.loc[target.Z[i]] ['weight'] ]
                
        
        self.stoich = [1] #we only have one element, lol

        self.name = nr.elements[Z_target]
        self.correction = 1.0  #reminder: mono-elemental have correction = 1.0 for SRIM
        self.density = atomic_table.loc[ Z_target ]['density'] * (U.g / U.cm**3)
        

    #Builds complex chemical compounds based upon their stoicheometry,
    # compound = str of the form CaseLetter_Digit, ie H2O or C2H7Cl
    def set_compound(self, compound, mass=None):
        #Use regex to split by CaseLetter_Digit 
        target = re.findall(r'([A-Z][a-z]*)(\d*)', compound)
        
        #Get's rid of empty strings for non_digit elements (ie H2O), compresses to 1D array
        target = [ a if a else '1'  for element in target for a in element]

        #slice by even / odd indice
        element_symbols, stoich = target[::2], target[1::2]        

        
        
        self.Z = [nr.elements.index(a_symbol) for a_symbol in element_symbols]
#        if mass:
#            self.mass=[mass]
#        else:
#            self.mass = [ atomic_table.loc[target.Z[i]] ['weight'] ]
#
        
        self.stoich = [int(a) for a in stoich]

        self.name = compound
        #Have specialized functions (or knowledgeable programs) set the density, correction factor


    #Kept for historical purposes, will hopefully eliminate soon.  Maps to set_compound()
    def make_target(self, var):
        self.set_compound(var)

    def set_thickness(self, thickness):
        if not nr.has_units(thickness):
            thickness = thickness * (U.ug / U.cm**2)
        elif (thickness.to_base_units().u == U.m):  #if we're dealing with linear distance
            thickness = (thickness * self.density).to(U.ug / U.cm**2)

        self.thickness = thickness
        
    def set_density(self, d):
        self.density = d
        
    #Set up isobutane 
    def isobutane(self, P):
        self.set_compound('C4H10')
        
        if (type(P) != type(1.*U.torr)):
            P = P*U.torr
        #isobutane:
        molar_mass = 0.*(U.g / U.mol)
        for i in range(len(self.Z)):
            molar_mass += self.stoich[i] * atomic_table.loc[ self.Z[i] ]['weight'] * (U.g / U.mol)
            
        self.name = 'isobutane' + '_' + str(P.m)+ str(P.u)
        self.density = gas_density(molar_mass, P)
        self.is_gas = True
        self.correction = 1.037196  #from SRIM, for Z_ion >= 3

    #Set up isobutane 
    def mylar(self):
        self.set_compound('C10H8O4')
        self.name = 'mylar'
        
        #from SRIM, for Z_ion >= 3        
        self.density = 1.397 * (U.g / U.cm**3)
        self.correction = 1.037196
        
    #Set up isobutane 
    def polyethylene(self):
        self.set_compound('C2H4')
        self.name = 'CH2'
        
        #from SRIM, for Z_ion >= 3        
        self.density = 0.93 * (U.g / U.cm**3)
        self.correction = 0.9843957

    #Set up  
    def CD2(self):
        self.set_compound('C2H4')
        self.name = 'CD2'
        #Density from J Nucl Mat. 405 (2010) 181-185
        #Correction from SRIM's polyethylene foil
        #mass == 2.01410 u
        self.density = 1.06 * (U.g / U.cm**3)
        self.correction = 0.9843957

    def LiF(self):
        self.set_compound('LiF')
        self.name = 'LiF'

        #Density from srim, confirmed on Wikipedia
        self.density = 2.635 * (U.g / U.cm**3)
        #self.correction = No correction supplied from srim
        

    def propanediol(self):
        self.set_compound('C3H8O2')
        self.name = 'propanediol'

        self.density = 1.0597 * (U.g / U.cm**3)
        self.correction = 0.9457121


#inputs:
#Target has the form:  [Z_target]
#E_max defaults to MeV
#output file is 'ASy-tar.dat', where A is the mass number, Sy is the elemental symbol, tar is the target
def get_table(A, Z, target, E_max, outfile_name='', E_min=10.*U.keV, verbose=False):
    
    #Start dealing with the target & ion        
    if not nr.has_units(E_max):
        E_max = E_max*default_E_unit

    E_max = np.ceil(E_max)

    #Set it in kev, SRIM's native E unit
    E_min, E_max = E_min.to(U.keV), E_max.to(U.keV)

    if not A:
        A = atomic_table.loc[Z]['weight']

    if not outfile_name:
        outfile_name = SRIM_out_dir+ str(A) + nr.elements[Z] + '-' + target.name + '.dat'
        
    #Ensure that we don't overwhelm SRIM with our Zs
    if Z > 92:
        Z = 92
    
    #If we have already run SRIM for this dataset...
    if os.path.isfile(outfile_name) and os.path.getsize(outfile_name) > 0:
        if verbose:
            print(outfile_name)
        
        curr_srim_E, _ = read_srim_table(outfile_name, verbose=verbose)

        #And our SRIM output covers the E range we're intersted in
        #Have some floating point conversion issues, can't do direct comparison
        #if (curr_srim_E[0] <= E_min) and (curr_srim_E[-1] >= E_max):
        close_to_zero = 1E-9*U.keV #within an eV
        if ((curr_srim_E[0] - E_min) <= close_to_zero) and ((E_max - curr_srim_E[-1]) <= close_to_zero):
            return outfile_name
        else:
            raise forms.ValidationError("Data Table for the given parameters has not yet been simulated in SRIM. Live SRIM interfacing is coming soon.")
    else:
        raise forms.ValidationError("Data Table for the given parameters has not yet been simulated in SRIM. Live SRIM interfacing is coming soon.")
    #ELSE, setup SR.IN and run SRIM:
    
    #SRIM input file is easy, but needs the comments to be parsed correctly
    #NOTE:  Ion energy can be int or float
    #       Target name is unnecesary, so just left blank ""
    #       Any whitespace is fine to differentiate values
    #       Carriage return needed for windows prog 
    srim_input = '---Stopping/Range Input Data(Number-format: Period = Decimal Point)\r\n' +\
                 '---Output File Name\r\n' +\
                 '"' + outfile_name + '"\r\n'+\
                 '---Ion(Z), Ion Mass(u)\r\n'+\
                 str(Z) + ' ' + str(A) + '\r\n'+\
                 '---Target Data: (Solid=0,Gas=1), Density(g/cm3), Compound Corr.\r\n'+\
                 str(int(target.is_gas))+' '+str(target.density.m)+' '+str(target.correction)+'\r\n'+\
                 '---Number of Target Elements\r\n'+\
                 ' ' + str(len(target.Z)) +' \r\n'+\
                 '---Target Elements: (Z), Target name, Stoich, Target Mass(u)\r\n'
    
    for i in range(len(target.Z)):
                #'Z "" stoich atomic_weight\r\n' <-- line syntax
        srim_input += str(target.Z[i]) + ' "" ' + str(target.stoich[i]) + ' ' +\
                      str(atomic_table.loc[target.Z[i]] ['weight']) + '\r\n'
        
    srim_input +='---Output Stopping Units (1-8)\r\n'+\
                 ' 5\r\n'+\
                 '---Ion Energy : E-Min(keV), E-Max(keV)\r\n'+\
                 ' ' + str(E_min.m) + ' ' + str(E_max.m)
    
    out = open(SRIM_run_dir + 'SR.IN', 'w')
    out.write(srim_input)
    out.flush()

    
    ##RUNNING SRModule.exe via WINE via bash
    pwd = os.getcwd()
    os.chdir(SRIM_run_dir)
    cmd = 'bash srim.sh'

    pout = subprocess.check_output(cmd, shell=True)    

    os.chdir(pwd)
    

    #srim_err = p.stderr.readline()
    known_srim_err = '0000:fixme:ole:OaBuildVersion Version value not known yet. Please investigate it !\n'
    #
    #if srim_out:
    #    print('SRIM OUTPUT\n  >'+ srim_out)
    #
    #if (srim_err and srim_err != known_srim_err):
    if pout[4:] != known_srim_err[4:]: #first four char above are an incrementor
        print('SRIM ERR\n  >'+ pout)
        

    return outfile_name


#taken from old code, should re-write to make it take full SRIM input instead of the redacted shit
#Updated old code to allow for full SRIM files instead of chopped ones
def read_srim_table(filename, verbose=False):
    f = open(filename, 'r')

    if verbose:
        print('Reading SRIM table: ' + filename)
    
    #Stopping and Range table (SRIM std. output expected)
    eArr=[]  #Array of energies
    rArr=[]  #Array of ranges in material
    line=f.readline()
    
    #Test for old style input files (where we manually cut the header and footer)
    test_array = line.split()

    #Either/Or:  
    #  Either we have a valid first value (ie, manually cut up SRIM file)
    #  Or we have the full SRIM file
    try:
        float(test_array[0])
        in_header, in_footer = False, False   #We start in the body, no such thing as a header or footer
    except ValueError:
        in_header, in_footer = True, False    #We start in the header, not the body or the footer

    #Expects this to be in std SRIM output
    while(line !=''):
        #if we reached the first line of the footer
        # flagged by '-----' in the first five chars,
        # latches to a True
        if (not in_header) and (not in_footer) and (line[0:5] == '-----'):
            in_footer = True

        #If we're in the body of the SRIM file
        if ( not in_footer) and (not in_header):
            currLine=line.split()  #w/o argument, splits on whitespace (score!)

            #Throw an error if we've not parsed the unit correctly
            e_unit = currLine[1]
            if e_unit == 'eV':
                e_unit = 1.*U.eV
            elif e_unit == 'keV':     
                e_unit = 1.*U.keV
            elif e_unit == 'MeV':
                e_unit = 1.*U.MeV
            elif e_unit == 'GeV':
                e_unit = 1.*U.GeV
            eArr.append(float(currLine[0]) * e_unit)
        
            r_unit = U.um
            if currLine[5] == 'A':
                r_unit = 0.1*U.nm
            elif currLine[5] == 'mm':
                r_unit = 1.*U.mm
            elif currLine[5] == 'm':
                r_unit = 1.*U.m
            rArr.append(float(currLine[4]) * r_unit)

                
        #if we reached the last line of the header
        # flagged by '  ---' in the first five chars,
        # latches to a False
        #NOTE SRModule.exe doesn't ahve the leading whitespace
        if (in_header) and (line[0:5] == '-----'):  
            in_header = False
            
        #read next line
        line=f.readline()

    #Converting to numpy arrays means extracting magnitude, then re-attaching unit
    eArr = np.array([measure.to(default_E_unit).magnitude for measure in eArr]) * default_E_unit
    rArr = np.array([measure.to(default_dist_unit).magnitude for measure in rArr]) * default_dist_unit
    
    return (eArr, rArr)


#Calculates the exit energy for an energetic ion moving through a material
#
#  E_ent     : entrance energy
#  distance  : distance travelled
#  deg       : DEFAULT 4,  polynomial degree for fitting the SRIM data
#  upper_win : window about E_ent and E_exit in SRIM table, this is the upper value for E_ent
#  lower_win : window about E_ent and E_exit in SRIM table, this is the lower value for E_exit
def exit_energy(E_ent, distance, SRIM_file, deg = 4, upper_win = 5, lower_win = 5, show=False, units=True):

    #Verify we are coming in with units
    if not nr.has_units(E_ent):
        E_ent = E_ent*default_E_unit
    if not nr.has_units(distance):
        distance = distance*default_dist_unit

    distance = distance.to(default_dist_unit)

    #TODO  UPdate and include areal density sensitivity

        

    orig_srim_eloss, orig_srim_distance = read_srim_table(SRIM_file)

    #Ensure we're working in the same units
    E_ent = E_ent.to(default_E_unit)
    distance = distance.to(default_dist_unit)



    #####Truncate the srim values about our starting energy....

    #Search for the nearest indice for our starting energy val
    i_near_E = 0
    for i in range(len(orig_srim_eloss)):
        #If we've passed over initial E value, w
        if orig_srim_eloss[i] > E_ent:
            #Assume we found the right val
            i_near_E = i

            #should throw errors if we're at the boundaries,
            #that's users fault for supplying shitty SRIM vals
            #if E_ent is less than the average of these two values, it's closer to the lower one
            if (orig_srim_eloss[i] + orig_srim_eloss[i-1])/2. > E_ent:
                i_near_E = i -1

            if show:
                print('E_starting: ' + str(E_ent) + '  -vs-  srim_val: ' + str(orig_srim_eloss[i_near_E]) + ' through ' +str(distance))
                
            #We've found the value needed, lets break out of this loop
            break

    upper_i = i_near_E + upper_win
    lower_i = i_near_E - lower_win

    #make sure we don't overrun bounds
    if upper_i >= len(orig_srim_eloss):
        upper_i = len(orig_srim_eloss) - 1
    if lower_i <0:
        lower_i = 0

    srim_eloss = orig_srim_eloss[lower_i:upper_i]
    srim_distance = orig_srim_distance[lower_i : upper_i]
    ############################
    
    
    #energy as a function of distance
    E_dist = np.poly1d(np.polyfit(srim_distance, srim_eloss, deg))

    if show:
        plt.scatter(srim_distance, srim_eloss, label ="SRIM")
        plt.plot(srim_distance, E_dist(srim_distance), label ="E_dist")

        #plt.plot(dist_E(srim_eloss), srim_eloss, label ="dist_E")
        

        plt.xlabel('Dist (' + str(srim_distance.u) + ')')
        plt.ylabel('Energy (' + str(srim_eloss.u) + ')')
        
        #init_dist = dist_E(E_ent)
        
        #plt.plot(init_dist, E_ent, 'ro', ms=5, label='Init E')

        #final_dist = init_dist - distance.m

        #plt.plot(final_dist, E_dist(final_dist), 'go', ms=5, label='Final E')
        plt.legend(loc='best')
        plt.show()


    #distnace as a function of energy
    #A lot of funky shit in this block.  Should rethink it but fuck it
    def dist_E(var_E):
        
        #Cludge to only work with unit'd guys, will work with any unit in type()
        if isinstance(var_E, type(1.*default_E_unit)):
            #Once the proper order of magnitude is set, extract the magnitude to work with 
            var_E = var_E.to(default_E_unit).m
            
        #If this isn't a numpy array, make it one
        #Also, kinda weird, but because of recastin, we sometimes wind up with a nonarray np.array,
        # so make sure it's iterable
        if not isinstance(var_E, np.ndarray) or np.size(var_E) == 1:
            var_E = np.array([var_E])
        
        return_distances = []
        
        #Go beyond the known present min, max by drawing a line from second min, max through min, max; to a real min, max
        min_dist, max_dist = (2.*srim_distance[0] - srim_distance[1]).m, (2.*srim_distance[-1] - srim_distance[-2]).m

        #for each input in the list, calculate distance
        for an_E in var_E:
            #find possible values for d by solving for the roots, len(possible_val) == degree
            possible_dists = (E_dist - an_E).roots

            #mask away complex roots (which naturally arise) and any out-of bounds roots
            possible_dists = [ x.real for x in possible_dists if np.isreal(x) and (min_dist < x < max_dist) ]

            if show and len(possible_dists) == 0:
                print('Couldnt find a good value in inverse function: ' + str((E_dist - an_E).roots) +
                      ' -vs-  min:' + str(min_dist) +'  max: ' +  str(max_dist))
            
            return_distances.append(possible_dists)
            
        return np.concatenate(return_distances)
        
    if show:
        plt.scatter(srim_distance, srim_eloss, label ="SRIM")
        plt.plot(srim_distance, E_dist(srim_distance), label ="E_dist")

        plt.plot(dist_E(srim_eloss), srim_eloss, label ="dist_E")
        

        plt.xlabel('Dist (' + str(srim_distance.u) + ')')
        plt.ylabel('Energy (' + str(srim_eloss.u) + ')')
        
        init_dist = dist_E(E_ent)
        
        plt.plot(init_dist, E_ent, 'ro', ms=5, label='Init E')

        final_dist = init_dist - distance.m

        plt.plot(final_dist, E_dist(final_dist), 'go', ms=5, label='Final E')
        plt.legend(loc='best')
        plt.show()


    
    final_position = dist_E(E_ent) - distance.m

    
    #If we've reached the edge of our SRIM data and it's still coming up 0
    if final_position < 0.0 and lower_i == 0:
        return 0.0*default_E_unit

    #If we've truncated too much for the final E, move bottom window
    elif final_position < srim_distance[0].m:
        if show:
            print('widening sample window: '+ str( lower_win + 2))
            time.sleep(2)
        return exit_energy(E_ent, distance, SRIM_file, \
                           deg = deg, lower_win = lower_win + 2, show=show)

    return E_dist(final_position)[0]*default_E_unit


#An updated version of the exit_energy function, to make use of srim.Material knowledge
#Calculates the exit energy for an energetic ion moving through a material
#
#  E_ent     : entrance energy
#  distance  : distance travelled
#  deg       : DEFAULT 4,  polynomial degree for fitting the SRIM data
#  upper_win : window about E_ent and E_exit in SRIM table, this is the upper value for E_ent
#  lower_win : window about E_ent and E_exit in SRIM table, this is the lower value for E_exit
def energy_loss_old(A_beam, Z_beam, E_beam, target, target_thickness=None, deg = 4, upper_win = 5, lower_win = 5, show=False, units=True):

    
    #Verify we are coming in with units
    if not nr.has_units(E_beam):
        E_beam = E_beam*default_E_unit


    #round up to next largest power of 10 for srim_table
    max_Ebeam = (10**int(np.ceil(np.log10(E_beam.m))))*E_beam.u


    if target_thickness is None:
        target_thickness = target.thickness
        
    SRIM_table = get_table(A_beam, Z_beam, target, max_Ebeam)
    
    orig_srim_eloss, orig_srim_distance = read_srim_table(SRIM_table)

    #Ensure we're working in the same units
    E_beam = E_beam.to(default_E_unit)
    distance = (target_thickness / target.density).to(default_dist_unit)
      
    
    #####Truncate the srim values about our starting energy....

    #Search for the nearest indice for our starting energy val
    i_near_E = 0
    for i in range(len(orig_srim_eloss)):
        #If we've passed over initial E value, w
        if orig_srim_eloss[i] > E_beam:
            #Assume we found the right val
            i_near_E = i

            #should throw errors if we're at the boundaries,
            #that's users fault for supplying shitty SRIM vals
            #if E_ent is less than the average of these two values, it's closer to the lower one
            if (orig_srim_eloss[i] + orig_srim_eloss[i-1])/2. > E_beam:
                i_near_E = i -1

            if show:
                print('E_starting: ' + str(E_beam) + '  -vs-  srim_val: ' + str(orig_srim_eloss[i_near_E]) + ' through ' +str(distance))
                
            #We've found the value needed, lets break out of this loop
            break

    upper_i = i_near_E + upper_win
    lower_i = i_near_E - lower_win

    #make sure we don't overrun bounds
    if upper_i >= len(orig_srim_eloss):
        upper_i = len(orig_srim_eloss) #- 1
    if lower_i <0:
        lower_i = 0

    srim_eloss = orig_srim_eloss[lower_i:upper_i]
    srim_distance = orig_srim_distance[lower_i : upper_i]

    if show:
        print('Srim energy window: ' + str(srim_eloss[0]) + ' up to ' + str(srim_eloss[-1]))
    ############################
    
    
    #energy as a function of distance
    E_dist = np.poly1d(np.polyfit(srim_distance, srim_eloss, deg))

    if show:
        plt.scatter(srim_distance, srim_eloss, label ="SRIM")
        plt.plot(srim_distance, E_dist(srim_distance), label ="E_dist")


        plt.xlabel('Dist (' + str(srim_distance.u) + ')')
        plt.ylabel('Energy (' + str(srim_eloss.u) + ')')
        plt.title( nr.elements[Z_beam]+ ' through ' + target.name)        
        #init_dist = dist_E(E_ent)
        
        #plt.plot(init_dist, E_ent, 'ro', ms=5, label='Init E')

        #final_dist = init_dist - distance.m

        #plt.plot(final_dist, E_dist(final_dist), 'go', ms=5, label='Final E')
        plt.legend(loc='best')
        plt.show()


    #distnace as a function of energy
    #A lot of funky shit in this block.  Should rethink it but fuck it
    def dist_E(var_E):
        
        #Cludge to only work with unit'd guys, will work with any unit in type()
        if isinstance(var_E, type(1.*default_E_unit)):
            #Once the proper order of magnitude is set, extract the magnitude to work with 
            var_E = var_E.to(default_E_unit).m
            
        #If this isn't a numpy array, make it one
        #Also, kinda weird, but because of recastin, we sometimes wind up with a nonarray np.array,
        # so make sure it's iterable
        if not isinstance(var_E, np.ndarray) or np.size(var_E) == 1:
            var_E = np.array([var_E])
        
        return_distances = []
        
        #Go beyond the known present min, max by drawing a line from second min, max through min, max; to a real min, max
        min_dist, max_dist = (2.*srim_distance[0] - srim_distance[1]).m, (2.*srim_distance[-1] - srim_distance[-2]).m
        #min_dist, max_dist = (srim_distance[0] - 2.*(srim_distance[1] - srim_distance[0])).m, (srim_distance[-1] + 2.*(srim_distance[-1] - srim_distance[-2])).m

        print('dis_E: ' + str(min_dist) + '_' + str(max_dist))
        

        #for each input in the list, calculate distance
        for an_E in var_E:
            #find possible values for d by solving for the roots, len(possible_val) == degree
            possible_dists = (E_dist - an_E).roots

#            print possible_dists

            #mask away complex roots (which naturally arise) and any out-of bounds roots
            possible_dists = [ x.real for x in possible_dists if np.isreal(x) and (min_dist < x < max_dist) ]

            if show and len(possible_dists) == 0:
                
                print('Couldnt find a good value in inverse function: ' + str((E_dist - an_E).roots) +
                      ' -vs-  min:' + str(min_dist) +'  max: ' +  str(max_dist))
            
            return_distances.append(possible_dists)

        print('dist_E' + str(return_distances))
#        if possible_dists.len != 1:
#            print 'fuck'
            #START YOUR WORK HERE, MOTHERFUCKER!!!!

#        print return_distances
            
        return np.concatenate(return_distances)
        
    if show:
        plt.scatter(srim_distance, srim_eloss, label ="SRIM")
        plt.plot(srim_distance, E_dist(srim_distance), label ="E_dist")

        plt.plot(dist_E(srim_eloss), srim_eloss, label ="dist_E")
        

        plt.xlabel('Dist (' + str(srim_distance.u) + ')')
        plt.ylabel('Energy (' + str(srim_eloss.u) + ')')
        
        init_dist = dist_E(E_beam)
        
        plt.plot(init_dist, E_beam, 'ro', ms=5, label='Init E')

        final_dist = init_dist - distance.m

        plt.plot(final_dist, E_dist(final_dist), 'go', ms=5, label='Final E')
        plt.legend(loc='best')
        plt.show()

    print('-----------------------------------------------------------------------------')
    final_position = dist_E(E_beam) - distance.m
    
    #If we've reached the edge of our SRIM data and it's still coming up 0
    if final_position < 0.0 and lower_i == 0:
        return 0.0*default_E_unit

    #If we've truncated too much for the final E, move bottom window
    elif final_position < srim_distance[0].m:
        if show:
            print('widening sample window: '+ str( lower_win + 2))
            time.sleep(2)
        return energy_loss_old(A_beam, Z_beam, E_beam, target, target_thickness=target_thickness, \
                           deg = deg, lower_win = lower_win + 2, show=show)

    print(str(E_beam) + '--' + str(lower_win) + ' ' +str(final_position) + ' -- ' + str(dist_E(E_beam)) + ', ' + str(distance.m))

    return E_dist(final_position)[0]*default_E_unit


#An re-written, quicker, and cleaner version of energy_loss function, which makes use of srim.Material knowledge
#Calculates the exit energy for an energetic ion moving through a material
#
#  Rewritten Jun 3, 2019 cause I was fed up with this cludgy, old code
#
#  A_beam             : A of incident ions 
#  Z_beam             : Z of incident ions
#  E_beam             : Energy of incident ions (if no units, defaults to MeV)
#  stopping_materials : A srim material, as constructed by the Material() class
#  material_thickness : thickness of stopping medium, (DEFAULT None, takes it from Material() obj)
#  deg                : DEFAULT 4, degree of polynomial to fit the E_loss vs dist function
#  upper_win          : upper window (in terms of entries) above the initial E_beam 
#  lower_win          : lower window ""
def energy_loss(A_beam, Z_beam, E_beam, stopping_material, material_thickness=None, deg=4, upper_win=5, lower_win=5, show=False, units=True):

    if material_thickness is None:
        material_thickness = stopping_material.thickness 
    
    #Verify we are coming in with units
    if not nr.has_units(E_beam):
        E_beam = E_beam*default_E_unit

    #Ensure we're working in the correct units
    E_beam = E_beam.to(default_E_unit)
    dist_material = (material_thickness / stopping_material.density).to(default_dist_unit)
    
    #Get the SRIM sim outputs:
    #  1) Get an upper Energy bound ( by rounding up to next largest power of 10)
    #  2) generate the stopping and range table
    #  3) read in the table
    max_Ebeam = (10**int(np.ceil(np.log10(E_beam.m))))*E_beam.u        
    SRIM_table = get_table(A_beam, Z_beam, stopping_material, max_Ebeam)    
    srim_Estop, srim_dist = read_srim_table(SRIM_table)

    ##Get the index closest to our starting E_beam val
    idx_E = np.argmin(np.abs(srim_Estop - E_beam))
    
    #If we fully stop in this material, will recoil out of material with 0.0 energy
    if srim_dist[idx_E] < dist_material:
        return 0.*default_E_unit
    
    #Create out little windows for the poly_fit
    i_upper = idx_E + upper_win if ((idx_E + upper_win) < len(srim_Estop)) else len(srim_Estop)
    i_lower = idx_E - lower_win if ((idx_E - lower_win) > 0) else 0
    
    srim_Estop = srim_Estop[ i_lower : i_upper ]
    srim_dist  = srim_dist[  i_lower : i_upper ]       
    
    if show:
        print('Srim energy window (about ' + str(srim_Estop[idx_E - (i_lower + lower_win)]) + ', i=' +str(idx_E) +'): ' +\
              str(srim_Estop[0]) + '(i=' + str(i_lower) +') up to ' + str(srim_Estop[-1]) + '(i=' + str(i_upper) +')')

    # I like being explicity that E_dist, dist_E is a function:
    poly_funct_E_dist = np.poly1d(np.polyfit(srim_dist, srim_Estop, deg))
    def E_dist(a_dist):
        return poly_funct_E_dist(a_dist)

    poly_funct_dist_E = np.poly1d(np.polyfit(srim_Estop, srim_dist, deg))   
    def dist_E(an_E):
        return poly_funct_dist_E(an_E)

    #I can't do two different polyfits b/c the slight differences in the fits causes dist uncertainties ~10 um,
    # which is unacceptable for super thin targets / near stopped beams. I need an actual inverse function:
    # ** From experience, found that dist_E fits are the best at reproducing the curve behavior,
    #    as theyr'e kinda parabolic when x: E and y: dist (so inverted parabola for our normal view)
    #    so gonna invert dist_E.
    # ** This also relies on only being run within the domain of srim_dist and range of srim_Estop
    def E_dist2(var_dist):
        #Strip the units if it's unit'd
        if nr.has_units(var_dist):
            var_dist = var_dist.to(default_dist_unit).m
        #ensure it's an iterable numpy 
        if not isinstance(var_dist, np.ndarray) or np.size(var_dist) == 1:
            var_dist = np.array([var_dist])          
        
        return_E =  []

        #can do this cause, even for arbitrary polynomial orders,
        # the Eloss vs dist travelled funciton should be monotonically increasing
        # with the srim E and dist windows
        for a_d in var_dist:
            #Bit of loose constraint:  the correct value willbe the maximum (usually >0), since this has
            # parabolic behavior where the vertex E~0.
            var_E = [an_E.real for an_E in (poly_funct_dist_E - a_d).roots if np.isreal(an_E) ]

            #get the closest value to what E_dist predicted
            # (though E_dist isn't that accurate of a regression, it'll get us right ballpark
            var_E = var_E[ np.argmin(np.abs(var_E - E_dist(a_d))) ] 
            
            #if not len(var_E) > 0 :
            #    raise Exception(str(a_d) + ': ' + str((poly_funct_dist_E - a_d).roots) +'----- should be near:' + str(E_dist(a_d)))
            
            return_E.append(var_E)

        return np.array(return_E)
    

    #Get residual distance  (equivalent to residual energy)
    residual_dist = dist_E(E_beam) - dist_material.m
    srim_data_covers_distance = residual_dist > srim_dist[0].m

    if show:
        #capture a range just beyond the actual range, for plotting purposes
        E_range, d_range = srim_Estop.m, srim_dist.m        
        E_range = np.arange(E_range[0] - (E_range[1] - E_range[0]), E_range[-1] + (E_range[-1] - E_range[-2]), 0.1)
        d_range = np.arange(d_range[0] - (d_range[1] - d_range[0]), d_range[-1] + (d_range[-1] - d_range[-2]), 0.1)

        plt.subplot(2,1,1)
        plt.plot(d_range, E_beam.m*np.ones(d_range.size), color='black', lw=2, label='Init E_beam')
        plt.plot(dist_E(E_beam)*np.ones(E_range.size), E_range, color='black', lw=2) # initial distance
        plt.plot(dist_E(E_beam) - dist_material.m*np.ones(E_range.size), E_range, color='black', lw=2  ) # initial distnace - material thickness        

        plt.scatter(srim_dist, srim_Estop, label ="SRIM")
        plt.plot(d_range, E_dist(d_range), '--', label ='E_dist (not used for regression)')
        plt.plot(dist_E(E_range), E_range, '--', label = 'dist_E (used for regression)')

        #Only perform 
        if srim_data_covers_distance:

            plt.plot(dist_E(E_beam), E_beam, 'ro', ms=5, label='Init E')
            plt.plot(residual_dist, E_dist2(residual_dist), 'go', ms=5, label='Final E, res_dist')
        
            plt.xlabel('Dist (' + str(srim_dist.u) + ')')
            plt.ylabel('Energy (' + str(srim_Estop.u) + ')')
            plt.title( nr.elements[Z_beam]+ ' through ' + stopping_material.name)        
            plt.legend(loc='best')
            xlim = plt.gca().get_xlim()

            plt.subplot(2,1,2)
            plt.scatter(srim_dist, srim_Estop.m - E_dist2(srim_dist))
            plt.title('Residual diff between fit and SRIM points')
            plt.xlabel('Dist (' + str(srim_dist.u) + ')')
            plt.ylabel('Residual diff, E_SRIM - E_fit, ' + str(srim_Estop.u))
            plt.xlim(xlim) #match the above xlim

#        plt.tight_layout()
        plt.show()

    #If we've gone over the whole range, and still not found a positive residual energy, it's 0.
    if i_lower == 0 and residual_dist < 0.:
        return 0.*default_E_unit
   
    #If we've truncated too much for the final E, widen bottom window for fit
    # this should be avoided, as it'll introduce errors in the local fits
    if not srim_data_covers_distance:
        if show:
            print('residual_dist: ' + str(residual_dist) + ', widening sample window: '+ str( lower_win + 2))
        return energy_loss(A_beam, Z_beam, E_beam, stopping_material, material_thickness=material_thickness, \
                           deg = deg, lower_win = lower_win + 2, show=show)
    
    energy_remaining = E_dist2(residual_dist)[0]*default_E_unit
    if energy_remaining.m > 0.:
        return energy_remaining
    else:
        return 0.*default_E_unit

#An reversed version of energy_loss
def energy_deposited(A_beam, Z_beam, E_beam, stopping_material, material_thickness=None, deg = 4, upper_win = 5, lower_win = 5, show=False, units=True):
    return E_beam - energy_loss(A_beam, Z_beam, E_beam, stopping_material,
                                material_thickness=material_thickness, deg = deg, upper_win = upper_win, lower_win = lower_win, show=show, units=units)








