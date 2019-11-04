from nuclear_reactions import U

from srim import *

import seaborn as sns
colors = sns.color_palette('colorblind')
sns.palplot(colors)
sns.set_style('whitegrid', {'grid.linestyle': '--'})
sns.set_palette(colors)

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

#mat = Material()
#mat.set_element('Al')
#mat.set_thickness(40*U.ug / U.cm**2)
#mat.set_thickness(249*U.mg / U.cm**2)
#print(energy_loss(4,2,50*U.MeV, mat))#, show=True))

A, Z  = 4,2

dE_det = Material()
dE_det.set_element('Si')
dE_det.set_thickness(87*U.um)

E_det = Material()
E_det.set_element('Si')
E_det.set_thickness(1*U.mm)

an_E = 36.5*U.MeV
deposited_in_dE_det = energy_deposited(A,Z,an_E, dE_det, show=True)
total_deposited = deposited_in_dE_det

if total_deposited != an_E:    
    total_deposited += energy_deposited(A,Z, (an_E - deposited_in_dE_det), E_det, show=True)
            
print ' dE: ' + str(deposited_in_dE_det)
print '  E: ' + str(energy_deposited(A,Z,an_E, E_det, show=True))
print 'tot: ' + str(total_deposited)


    
