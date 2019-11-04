#! /usr/bin/env python
#
#  NICK, 2017-10-17;  What started as a script to run SRIM has become something of a hodge-podge E, m/q checker for reactions
#   obvs still relying on running srim.  This is a good spot to stop, and continue dev in file check_rxn.py
#
#   Note, as of now, the program calculates E spread and m/q spread, with dumb spreads in these values, for a list of manually inputed
#  cross sections for different channels.

import os
from sarge import run, Capture  #for OS prog manipulation
import re  #for string manipulation
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import matplotlib._color_data as mcd
css = mcd.CSS4_COLORS


import nuclear_reactions as nr
from nuclear_reactions import U #Unit Regsitry


#Get the SRIM atomic table data
names = ['Z', 'atomic_symbol', 'name', 'mass_int', 'weight_most_abundant_isotope',\
         'weight', 'density', 'number_density', 'fermi_velocity', 'heat_sublimation',\
         'density_gas', 'number_density_gas']
widths= [2,3,15,4,8,8,8,8,6,6,9,9]
atomic_table = pd.read_fwf('atomic_data-SRIM.dat', widths=widths, names=names, skiprows=2, skipfooter=1)

#Organize by Z
atomic_table = atomic_table.set_index('Z')


#Default units to make sure we convert properly
default_E_unit = U.MeV
default_dist_unit = U.um





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
        self.stoich = [0]

        #Individual components of compound
        self.Z = [0]
        self.stoich = [0]

        

    ##################################################
    #
    # methods to make our target

    def make_target(self, target):     #figure out better name....
        self.name = target
        
        #Use regex to split by CaseLetter_Digit 
        target = re.findall(r'([A-Z][a-z]*)(\d*)', target)
        
        #Get's rid of empty strings for non_digit elements (ie H2O), compresses to 1D array
        target = [ a if a else '1'  for element in target for a in element]

        #slice by even / odd indice
        element_symbols, stoich = target[::2], target[1::2]        
        self.Z = [nr.elements.index(a_symbol) for a_symbol in element_symbols]
        self.stoich = [int(a) for a in stoich]

        #Defaults for monoisotopics
        if len(self.Z) == 1:
            #Default target_name to elemental symbol
            self.name = nr.elements[ self.Z[0] ]
            self.density = atomic_table.loc[ self.Z[0] ]['density'] * (U.g / U.cm**3)
            #REMINDER: only mono-elemental have correction=1
            self.correction = 1.0

    #Set up isobutane 
    def isobutane(self, P):
        self.make_target('C4H10')
        
        if (type(P) != type(1.*U.torr)):
            P = P*U.torr
        #isobutane:
        molar_mass = 0.*(U.g / U.mol)
        for i in range(len(self.Z)):
            print self.stoich[i]
            molar_mass += self.stoich[i] * atomic_table.loc[ self.Z[i] ]['weight'] * (U.g / U.mol)
            
        self.name = 'isobutane' + '_' + str(P.m)+ str(P.u)
        self.density = gas_density(molar_mass, P)
        self.is_gas = True
        self.correction = 1.037196  #from SRIM, for Z_ion >= 3

    #Set up isobutane 
    def mylar(self):
        self.make_target('C10H8O4')
        self.name = 'mylar'
        
        #from SRIM, for Z_ion >= 3        
        self.density = 1.397 * (U.g / U.cm**3)
        self.correction = 1.037196

    def propanediol(self):
        self.make_target('C3H8O2')
        self.name = 'propanediol'

        self.density = 1.0597 * (U.g / U.cm**3)
        self.correction = 0.9457121


#inputs:
#Target has the form:  [Z_target]
#E_max defaults to MeV
#output file is 'ASy-tar.dat', where A is the mass number, Sy is the elemental symbol, tar is the target
def run_srim(A, Z, target, E_max, outfile_name='', E_min=10.*U.keV):

    #Start dealing with the target & ion        
    if (type(E_max) != type(1.*U.MeV)):
        E_max = E_max*U.MeV

    #Set it in kev, SRIM's native E unit
    E_min, E_max = E_min.to(U.keV), E_max.to(U.keV)

    if not A:
        A = atomic_table.loc[Z]['weight']
    
    if not outfile_name:
        outfile_name = str(A) + nr.elements[Z] + '-' + target.name + '.dat'
        
    #If we have already run SRIM for this dataset...
    if os.path.isfile(outfile_name):
        E, _ = read_srim_table(outfile_name)
        #And our SRIM output covers the E range we're intersted in
        if (E[0] <= E_min) and (E[-1] >= E_max):
            return
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
    
    out = open('SR.IN', 'w')
    out.write(srim_input)
    out.flush()

    ##RUNNING SRModule.exe via WINE, should be adapted for other OSs (but not by me, lol)
    cmd = 'wine SRModule.exe'
    p = run(cmd, stdout=Capture(), stderr=Capture(), async=True) # returns immediately
    p.close() # wait for completion

    #Display any janky or unknown output info...
    srim_out = p.stdout.readline()
    srim_err = p.stderr.readline()
    known_srim_err = 'fixme:ole:OaBuildVersion Version value not known yet. Please investigate it !\n'

    if srim_out:
        print('SRIM OUTPUT\n  >'+ srim_out)

    if (srim_err and srim_err != known_srim_err):
        print('SRIM ERR\n  >'+ srim_err)

    return


#taken from old code, should re-write to make it take full SRIM input instead of the redacted shit
#Updated old code to allow for full SRIM files instead of chopped ones
def read_srim_table(filename):
    f = open(filename, 'r')

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
    
            e_unit = 1.*U.MeV
            if currLine[1] == 'keV':     
                e_unit = 1.*U.keV
            elif currLine[1] == 'eV':
                e_unit = 1.*U.eV
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
    if type(E_ent) != type(1.0*default_E_unit) :
        E_ent = E_ent*default_E_unit
    if type(distance) != type(1.0*default_dist_unit) :
        distance = distance*default_dist_unit

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
                print('E_starting: ' + str(E_ent) + '  -vs-  srim_val: ' + str(orig_srim_eloss[i_near_E]))
                
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


def run_rxn(A_target, Z_target, target_thickness, A_beam, Z_beam, E_beam, I_beam,  A_prod, Z_prod, cross_section, color, cent_E, cent_mq,  srim_dir='srim_files/'):    
    #If we don't have units, assumes ug/cm**2
    if not nr.has_units(target_thickness):
        target_thickness = 1.0*(U.ug / U.cm**2)
    if not nr.has_units(E_beam):
        E_beam = E_beam*U.MeV
    if not nr.has_units(I_beam):
        I_beam = I_beam*(U.num / U.s)
    if not nr.has_units(cross_section):
        E_beam = E_beam*U.mb
    
    #Make a target material
    target = Material()
    target.make_target(nr.elements[Z_target]) #uses elemental symbol for target gen

    #Be considerate of other units...
    if (target_thickness.to_base_units().u == U.m):
        target_thickness = (target.density * target_thickness).to(U.ug / U.cm**2)

    if (I_beam.to_base_units().u == U.A):
        I_beam = I_beam.to(U.e / U.s) * (1.*(U.num / U.e))
        
    #Insensitive to spitting out composit nuclei, only does p,n right now
    ejecta, Z_diff = '', Z_target + Z_beam - Z_prod
    if (Z_diff > 0):
        ejecta += str(Z_diff) + 'p'
        ejecta += str(A_target +A_beam -A_prod - Z_diff) + 'n'
    else:
        ejecta += str(A_target +A_beam -A_prod) + 'n'

    TAB = '  '
    print('Running ' + nr.elements[Z_target] + '-' + str(A_target) + '(' +\
          nr.elements[Z_beam] + '-' + str(A_beam) + ',' + ejecta + ')' +\
          nr.elements[Z_prod] + '-' + str(A_prod) + ' :\n' + TAB +\
          nr.print_unit(I_beam) +' of ' + nr.print_unit(E_beam) + ' beam on ' +\
          nr.print_unit(target_thickness) + ', cs=' + nr.print_unit(cross_section))




    def gaussian(x, mu, sig, A):
        return A/(np.sqrt(2.*np.pi)*sig)*np.exp(-1.*((x - mu)**2. / ( 2 * sig**2.) ) )

    
    #####################################################
    #  reaction rate info
    rate = nr.production_rate(cross_section, target_thickness, A_target, I_beam)

    #####################################################
    #  E
    #Go to the next biggest digit to ensure the upper E bounds are fully covered
    max_E = (10**int(np.ceil(np.log10(E_beam.m))))*E_beam.u
    min_E = (10**int(np.floor(np.log10(E_beam.m))))*E_beam.u

    #Do the srim shit
    SRIM_table = srim_dir + str(A_prod) + nr.elements[Z_prod] + '-' + target.name + '.dat'
    run_srim(A_prod, Z_prod, target, max_E, SRIM_table)


    #exit_E needs linear distance, not areal density
    new_E = exit_energy((1.*A_beam / A_prod) * E_beam, (0.5*target_thickness / target.density), SRIM_table)
    A_E = rate.m

    #Placeholder
    sigma_E = new_E.m * 0.03



    
    #####################################################
    #  m/q
    q_bar, q_sigma = nr.charge.Schiwietz_q(A_prod, Z_prod, Z_target,new_E)
    mq = A_prod / round(q_bar)  #m/q

    #placeholder
    sigma_mq = mq *0.002
    
    A_mq = gaussian(round(q_bar), q_bar, q_sigma, rate.m) 
     


    main_E = cent_E
    upper_E, lower_E, E_bin = 0.3, 0.15, 0.01

    E_range = np.arange((main_E - (main_E*lower_E)).m, (main_E + (main_E*upper_E)).m, E_bin)*U.MeV

    q_bin = 1.
    q_range = np.arange(0,30, q_bin)*U.e

    q = np.around(q_bar)



    #Take the total rate, and find which % is made up of this 'q'
    q_percent = (gaussian(q, q_bar, q_sigma, rate.m)*q_bin) / rate.m
#    print(q_percent * rate.m)
    main_mq, mq_win, mq_bin  = cent_mq, 0.1, 0.01

    mq_range = np.arange(main_mq-(mq_win*main_mq), main_mq+(mq_win*main_mq), mq_bin)


    
    n_row, n_col = 2, 1
    
    
##    plt.subplot(n_row, n_col,1)
#    plt.plot(E_range, gaussian(E_range.m, new_E.m, sigma_E, rate.m)*0.01, label=(nr.elements[Z_prod] + str(A_prod) + ': ' + nr.print_unit(new_E, 2) +  '    {:.1f}'.format(np.sum(gaussian(E_range.m, new_E.m, sigma_E, rate.m)*E_bin))) +'pps', color = 'dark'+color)
#    plt.title('KE of EVR \n centered on ' + nr.print_unit(main_E) +'(win = +' + str(upper_E * 100) +'%, - ' + str(lower_E * 100) + '%)')
#    plt.xlabel('KE after target (MeV)')
#    plt.ylabel('Rate (pps)')
#    plt.legend()
    

#    plt.subplot(n_row, n_col,2)
#    plt.step(q_range, gaussian(q_range.m, q_bar, q_sigma, rate.m), color = color)
#    plt.subplot(n_row, n_col, 2)

    
    
    
    plt.step(mq_range, gaussian(mq_range, mq, sigma_mq, q_percent * rate.m)*mq_bin, color = css['dark'+color], label=('{:.0f}'.format(q) +'+ ' + nr.elements[Z_prod] + str(A_prod) +': {:.1f}'.format(np.sum(gaussian(mq_range, mq, sigma_mq, q_percent * rate.m)*mq_bin)) +'pps'))

    q_percent = (gaussian((q-1), q_bar, q_sigma, rate.m)*q_bin) / rate.m
    plt.step(mq_range, gaussian(mq_range, mq *(1.0 * q) / (q-1), sigma_mq, q_percent * rate.m)*mq_bin, color = css[color], label = ('{:.0f}'.format(q-1) + '+ ' +nr.elements[Z_prod] + str(A_prod) +': {:.1f}'.format(np.sum(gaussian(mq_range,  mq *(1.0 * q) / (q-1), sigma_mq, q_percent * rate.m)*mq_bin))) +'pps')

    plt.step(mq_range, gaussian(mq_range, mq *(1.0 * q) / (q+1), sigma_mq, q_percent * rate.m)*mq_bin, color = css['light'+color], label = ('{:.0f}'.format(q+1) + '+ ' +nr.elements[Z_prod] + str(A_prod) +': {:.1f}'.format(np.sum(gaussian(mq_range,  mq *(1.0 * q) / (q+1), sigma_mq, q_percent * rate.m)*mq_bin))) +'pps')

#    q_percent = (gaussian(q+1, q_bar, q_sigma, rate.m)*q_bin) / rate.m
#    plt.plot(mq_range, gaussian(mq_range, A_prod / (q+1), 0.025, q_percent * rate.m)*mq_bin, color = color)

#    print mq_range
#    print E_range
    plt.xlim([mq_range[0] - 0.6, mq_range[-1] + 0.252])
    plt.ylim([0.01, 150])

    plt.title('m/q of EVR\n centered at ' + str(main_mq) + ', window = $\pm$ '+ str(100*mq_win) + '%')
    plt.xlabel('m/q')
    plt.ylabel('Rate (pps)')

    plt.legend(loc='upper left')
    

    print(TAB + 'E: ' + nr.print_unit(new_E, 4) + ' q: {:.3f}'.format(q_bar) +', m/q: {:.3f}'.format(A_prod/q_bar))

    
    print(TAB + 'rate: ' +nr.print_unit(rate) + ' -vs- checkrate_E: ' + \
          '{:.1f}'.format(np.sum(gaussian(E_range.m, new_E.m, sigma_E, rate.m)*E_bin)) +\
          ' -vs - checkrate_m/q: ' +\
          '{:.1f}'.format(np.sum(gaussian(mq_range, A_prod / q_bar, (A_prod/q_sigma), rate.m)*mq_bin)))

    
    return

colors = ['blue', 'red', 'orange', 'cyan', 'violet', 'goldenrod',  'salmon', 'magenta', 'slateblue',  'seagreen', 'orchid', 'slategrey','khaki']

#run_rxn(63,29,1.25*U.um,   24,11,70.7*U.MeV,1E9*(U.num / U.s),   84,38, 159.08*U.mb, colors[0], 13.52*U.MeV, 7.9)
#run_rxn(63,29,1.25*U.um,   24,11,70.7*U.MeV,1E9*(U.num / U.s),   83,38, 106.46*U.mb, colors[1], 13.52*U.MeV, 7.9)
#run_rxn(63,29,1.25*U.um,   24,11,70.7*U.MeV,1E9*(U.num / U.s),   81,37,  28.64*U.mb, colors[2], 13.52*U.MeV, 7.9)
#run_rxn(63,29,1.25*U.um,   24,11,70.7*U.MeV,1E9*(U.num / U.s),   82,37,  52.67*U.mb, colors[3], 13.52*U.MeV, 7.9)
#run_rxn(63,29,1.25*U.um,   24,11,70.7*U.MeV,1E9*(U.num / U.s),   81,36,    4.3*U.mb, colors[4], 13.52*U.MeV, 7.9)
#
#
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   84,38,  77.41*U.mb, colors[5], 13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   87,40,  17.10*U.mb, colors[6], 13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   86,40,  27.80*U.mb, colors[7], 13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   86,39, 177.37*U.mb, colors[8], 13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   81,36,   6.79*U.mb, colors[9], 13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   85,38,  93.74*U.mb, colors[10],13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   86,38,  41.99*U.mb, colors[11],13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   85,39,  77.43*U.mb, colors[0], 13.52*U.MeV, 7.9)
#run_rxn(65,29,1.25*U.um,   24,11,70.5*U.MeV,1E9*(U.num / U.s),   83,37,  22.17*U.mb, colors[0], 13.52*U.MeV, 7.9)
#
#
#plt.suptitle('Cu-65,63 + Na-24 \nE_beam: 70.5 MeV, 1.25 um target, 1E9 pps beam', fontsize=18)
#plt.show()
#
#
#
#run_rxn(63,29,1.25*U.um,   24,11,139.77*U.MeV,1E9*(U.num / U.s),   79,36, 162.29*U.mb, colors[0], 31.20*U.MeV, 5.5)
#run_rxn(63,29,1.25*U.um,   24,11,139.77*U.MeV,1E9*(U.num / U.s),   83,38, 287.59*U.mb, colors[1], 31.20*U.MeV, 5.5)
#run_rxn(63,29,1.25*U.um,   24,11,139.77*U.MeV,1E9*(U.num / U.s),   78,36,  53.92*U.mb, colors[2], 31.20*U.MeV, 5.5)
#run_rxn(63,29,1.25*U.um,   24,11,139.77*U.MeV,1E9*(U.num / U.s),   82,37,  52.67*U.mb, colors[3], 31.20*U.MeV, 5.5)
#run_rxn(63,29,1.25*U.um,   24,11,139.77*U.MeV,1E9*(U.num / U.s),   81,37,  48.98*U.mb, colors[4], 31.20*U.MeV, 5.5)
#
#
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   80,36,  67.21*U.mb, colors[5], 31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   81,36, 157.52*U.mb, colors[6], 31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   85,38, 308.39*U.mb, colors[7], 31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   78,35,  15.74*U.mb, colors[8], 31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   82,37,  43.51*U.mb, colors[9], 31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   83,37,  29.66*U.mb, colors[10],31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   79,36,  24.76*U.mb, colors[11],31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   83,36,  41.90*U.mb, colors[0], 31.20*U.MeV, 5.5)
#run_rxn(65,29,1.25*U.um,   24,11,139.00*U.MeV,1E9*(U.num / U.s),   82,38,  23.08*U.mb, colors[0], 31.20*U.MeV, 5.5)
#
#
#plt.suptitle('Cu-65,63 + Na-24 \nE_beam: 139.25 MeV, 1.25 um target, 1E9 pps beam', fontsize=18)
#
#plt.show()
#
#
#run_rxn(51,23,1.25*U.um,   24,11,56.52*U.MeV,1E9*(U.num / U.s),   71,32,  75.53*U.mb, colors[0], 12.70*U.MeV, 7.0)
#run_rxn(51,23,1.25*U.um,   24,11,56.52*U.MeV,1E9*(U.num / U.s),   73,34,  99.15*U.mb, colors[1], 12.70*U.MeV, 7.0)
#run_rxn(51,23,1.25*U.um,   24,11,56.52*U.MeV,1E9*(U.num / U.s),   72,33, 164.76*U.mb, colors[2], 12.70*U.MeV, 7.0)
#run_rxn(51,23,1.25*U.um,   24,11,56.52*U.MeV,1E9*(U.num / U.s),   72,34,  31.33*U.mb, colors[3], 12.70*U.MeV, 7.0)
#run_rxn(51,23,1.25*U.um,   24,11,56.52*U.MeV,1E9*(U.num / U.s),   73,33,  27.47*U.mb, colors[4], 12.70*U.MeV, 7.0)
#plt.suptitle('V-51 + Na-24 \nE_beam: 56.5 MeV, 1.25 um target, 1E9 pps beam', fontsize=18)
#
#plt.show()
#
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   71,32, 275.53*U.mb, colors[0], 21.5*U.MeV, 5.5)
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   70,32,  43.56*U.mb, colors[1], 21.5*U.MeV, 5.5)
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   71,33,  63.69*U.mb, colors[2], 21.5*U.MeV, 5.5)
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   68,31,  43.48*U.mb, colors[3], 21.5*U.MeV, 5.5)
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   72,33,  23.41*U.mb, colors[4], 21.5*U.MeV, 5.5)
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   67,30,  70.85*U.mb, colors[4], 21.5*U.MeV, 5.5)
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   66,30,  21.60*U.mb, colors[4], 21.5*U.MeV, 5.5)
#run_rxn(51,23,1.25*U.um,   24,11,84.36*U.MeV,1E9*(U.num / U.s),   69,31,  20.97*U.mb, colors[4], 21.5*U.MeV, 5.5)
#plt.suptitle('V-51 + Na-24 \nE_beam: 84.6 MeV, 1.25 um target, 1E9 pps beam', fontsize=18)
#
#plt.show()





#PACE calc for 85.0MeV
A_target, Z_target, target_thickness     = 51, 23, 1.25*U.um
A_beam, Z_beam, E_beam, I_beam           = 24, 11, 85.0*U.MeV, 1E9*(U.num/U.s)

central_E, central_mq = 21.0*U.MeV, 5.25
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   72,34,   7.3 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   72,33,  11.5 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   72,32,   2.79*U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   71,34,  17.4 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   71,33, 162.0 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   71,32,  63.1 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   70,33,  27.4 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   70,32,  55.6 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   69,32,  17.3 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   69,31,  14.3 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   68,31, 208.0 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   68,30,  26.2 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   67,31,  69.6 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   66,30,  17.8 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   65,30, 115.0 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   65,29,  46.6 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   64,29,  13.1 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   62,28,  26.8 *U.mb, 'blue', central_E, central_mq)
plt.suptitle('V-51 + Na-24 w/ PACE \nE_beam: '+ nr.print_unit(E_beam)+', ' + nr.print_unit(target_thickness) +' target, ' + nr.print_unit(I_beam)+ 'pps beam', fontsize=18)

plt.show()



A_target, Z_target, target_thickness     = 63, 29, 1.25*U.um
A_beam, Z_beam, E_beam, I_beam           = 24, 11, 85.0*U.MeV, 1E9*(U.num/U.s)

central_E, central_mq = 16.93*U.MeV, 6.9
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   84,40,   8.87*U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   84,39,  31.7 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   84,38,  19.9 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   83,40,   9.75*U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   83,39, 113.0 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   83,38, 312.0 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   83,37,  30.9 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   82,38,  30.2 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   81,38,  22.8 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   81,37,  23.7 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   80,38,  14.9 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   80,37, 130.0 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   80,36,  49.6 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   78,36,  12.1 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   77,36,  15.4 *U.mb, 'blue', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   77,35,   8.92*U.mb, 'blue', central_E, central_mq)

A_target = 65
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   86,39,  15.4 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   85,40,  54.1 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   85,39, 249.0 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   85,38, 168.0 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   84,39,  60.2 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   84,38, 108.0 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   83,38,  12.6 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   82,38,  74.0 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   82,37,  76.5 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   82,36,  12.3 *U.mb, 'salmon', central_E, central_mq)
run_rxn(A_target,Z_target,target_thickness,   A_beam,Z_beam,E_beam,I_beam,   79,36,  16.1 *U.mb, 'salmon', central_E, central_mq)

plt.suptitle('Cu-63,65 + Na-24 w/ PACE \nE_beam: '+ nr.print_unit(E_beam)+', ' + nr.print_unit(target_thickness) +' target, ' + nr.print_unit(I_beam)+ 'pps beam', fontsize=18)

plt.show()







#
#
#
#target = Material()
#target.make_target('W')
#print target.density
#
#
#SRIM_table = 'test_output.snark'
#
#setup_srim(202, 85, target, 100.*U.MeV, SRIM_table)
#run_srim()
#
#new_E = exit_energy(13.8462*U.MeV, ((75.*U.ug/U.cm**2) / (19.25 * U.g / U.cm**3)), SRIM_table)
#
#
#
#
#print(new_E)
#

