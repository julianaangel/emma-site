#*************************************************************
# g4emma_input_setup.py
#
# Nadège Pulgar-Vidal
# 2017-Oct-03
#
# Some simple functions to set up the input, folders and files
# needed to run the EMMA Geant 4 Simulation (EMMAapp)
#*************************************************************

#===========================
# IMPORTS
#===========================
from shlex import quote
import subprocess as sp
import re
from pathlib import Path
import g4emma.infile_templates as IFileTemplates
from datetime import datetime, timedelta
from django.conf import settings
import shutil
import logging
from os import chmod
import stat #for permission modes

#===========================
# LOGGER SETUP
#===========================
stdlogger = logging.getLogger('django')


#===========================
# FUCNTIONS
#===========================

#---------------------------------------------------
# cleanup_old_userdirs
# PURPOSE: Get rid of all old user directories
#---------------------------------------------------
def cleanup_old_userdirs():
    p = Path(settings.DATA_DIRS) #where the userdirs are stored
    #awhile_ago = datetime.now() - timedelta(hours=36)
    awhile_ago = datetime.now() - timedelta(days=2)

    stdlogger.info("Removing all user dirs older than " + str(awhile_ago))

    # remove all user dirs older than 2 days
    for child in p.iterdir():
        if child.stat().st_ctime <  awhile_ago.timestamp():
            shutil.rmtree(str(child))


#---------------------------------------------------
# setup_unique_userdir
# PARAMETERs: The location of the user directory (with trailing'/')
# RETURNS: The name of the unique user dir that was
#          created
# PURPOSE: Setup unique directories so different users
#          running the simulation won't interfere with
#          each other
#
# It'll start with a "random" PID and increment until
# it finds a number not in use
#---------------------------------------------------
def setup_unique_userdir(user_dirs_path):
    # Get a unique name
    unique_part = sp.check_output("ps | grep ps", shell=True)

    #the slicing strips off extra quotes and a char: b'str_content'
    matched_part = re.match("b' *([0-9]+) .*'", str(unique_part))

    # Build full path
    # (left pad zeroes to width 5)
    curr_dir_num = int(matched_part.group(1))
    userdir = "UserDir_{:0>5}".format(curr_dir_num)
    userdir_path = "{}{}".format(user_dirs_path, userdir)

    # Cycle until you find a dir not in use
    while (Path(userdir_path).exists()):
        curr_dir_num += 1
        userdir = "UserDir_{:0>5}".format(curr_dir_num)
        userdir_path = "{}{}".format(user_dirs_path, userdir)

    # Setup the directory
    stdlogger.info("The chosen PID was available: "+ userdir)
    Path(userdir_path).mkdir()

    # Setup the subdirectories
    Path(userdir_path + "/UserInput").mkdir()
    Path(userdir_path + "/Results").mkdir()
    Path(userdir_path + "/BeamSampling").mkdir()
    Path(userdir_path + "/Plots").mkdir()

    # For some reason, on our server the above directories don't follow the umask
    # and don't get group write access, which I need to generate ROOT histograms
    # This is a quirk of Python on some systems apparently
    mode = stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH # 775
    chmod(userdir_path + "/Plots", mode)


    return userdir


#---------------------------------------------------
# merge_with_defaults
# PARAMETERS: The user input as a dictionary
# PURPOSE: Overlay the user'S input of a set of default
#          values to get a complete set of input values
#---------------------------------------------------
def merge_with_defaults(form_dict):
    MIN_THICKNESS = 0.000004 # must be bigger than genat4 step length (or something like that)

    # so we need a set of default values for most of the input fields
    # and then merge the two dictionaries with the user values overwriting the defaults

    # Remove the entries that have None as a value because we need those overwritten by the defaults
    # Since I can't remove items from a dictionary while iterating over it a quick fix is to wrap it in a list vvv
    for key,value in list(form_dict.items()):
        if value is None:
            del form_dict[key]

    if isinstance(form_dict, dict):
        #setup default values (they're all 0/"OUT"/"NO"/empty string)
        default_vals = dict(alpha_source_present = "NO",
                            alpha_source_kinetic_e = 0,
                            alpha_source_max_angle = 0,
                            #--------
                            # These 5 are the absolute required minimum for the simulation
                            # num_events = 0, #required
                            # beam_proton_num = 0, #required
                            # beam_nucleon_num = 0, #required
                            # beam_charge_state = 0, #required
                            # beam_kinetic_e = 0, #required
                            #--------
                            beam_e_spread = 0.1,
                            beam_diameter = 1,
                            beam_trans_emittance = 0, #TODO check this (does 0.3 mm pi rad over beta have correct units?)
                            #--------
                            # These 4 should be the same as the beam params above by default
                            center_traj_proton_num = form_dict['beam_proton_num'],
                            center_traj_nucleon_num = form_dict['beam_nucleon_num'],
                            center_traj_charge_state = form_dict['beam_charge_state'],
                            center_traj_kinetic_e = form_dict['beam_kinetic_e'],
                            #--------
                            ion_chamber_inserted = "OUT",
                            ion_chamber_pressure = 0,
                            ion_chamber_temp = 0,
                            mwpc_inserted = "IN", #this will always be in
                            rxn_z1_beam = form_dict['beam_proton_num'],
                            rxn_a1 = form_dict['beam_nucleon_num'],
                            rxn_z2_target = 0,
                            rxn_a2 = 0,
                            rxn_z3_recoil = 0,
                            rxn_a3 = 0,
                            rxn_z4_ejectile = 0,
                            rxn_a4 = 0,
                            rxn_min_angle = 0,
                            rxn_max_angle = 0,
                            rxn_recoil_charge = 0,
                            rxn_recoil_excitation_e = 0,
                            rxn_cross_sec = 0,
                            slit_1_inserted = "OUT",
                            slit_2_inserted = "OUT",
                            slit_3_inserted = "OUT",
                            slit_4_inserted = "OUT",
                            target_inserted = "OUT",
                            target_thickness = MIN_THICKNESS, #this is around the min thickness for the simulation not to crash
                            target_z_pos = 0, # this should not be changed/constant
                            target_density = 0,
                            target_num_elems = 0,
                            target_elems = "",
                            target_elem_1 = "",
                            target_elem_2 = "",
                            target_elem_3 = "",
                            target_elem_4 = "",
                            target_elem_5 = "",
                            degrader_1_inserted = "OUT",
                            degrader_1_thickness = MIN_THICKNESS,
                            degrader_1_density = 0,
                            degrader_1_num_elems = 0,
                            degrader_1_elems = "",
                            degrader_1_elem_1 = "",
                            degrader_1_elem_2 = "",
                            degrader_1_elem_3 = "",
                            degrader_1_elem_4 = "",
                            degrader_1_elem_5 = "",
                            degrader_2_inserted = "OUT",
                            degrader_2_thickness = MIN_THICKNESS,
                            degrader_2_density = 0,
                            degrader_2_num_elems = 0,
                            degrader_2_elems = "",
                            degrader_2_elem_1 = "",
                            degrader_2_elem_2 = "",
                            degrader_2_elem_3 = "",
                            degrader_2_elem_4 = "",
                            degrader_2_elem_5 = "",)

        default_vals.update(form_dict)


        # We need to apply a correction to change some 0/1 to OUT/IN or NO/YES
        fields_to_correct = ["ion_chamber_inserted", "target_inserted", "degrader_1_inserted", "degrader_2_inserted"]

        for field in fields_to_correct:
            if (int(default_vals[field])):
                default_vals[field] = "IN"
            else:
                default_vals[field] = "OUT"

        if (int(default_vals["alpha_source_present"])):
            default_vals["alpha_source_present"] = "YES"
        else:
            default_vals["alpha_source_present"] = "NO"

        # Change # of elements to 0 if object is not present
        num_elems_correction = ["target_", "degrader_1_", "degrader_2_"]

        for item in num_elems_correction:
            if default_vals[item+"inserted"] == "OUT":
                default_vals[item+"num_elems"] = 0

    else:
        #raise an error
        stdlogger.error("merge with defaults' parameter was not of type dict")
        print("error: merge_with_defaults, not dict") # placeholder

    return default_vals


#---------------------------------------------------
# write_input_files
# PARAMETERS: The user directory to place the files in
#             and the dictionary containing cleaned
#             user input
# PURPOSE: Set up the input files in the specified directory
#---------------------------------------------------
def write_input_files(userdir, form_dict):
    # Make the input files directory
    infile_dir_name = "/UserInput"
    infile_dir_path = userdir + infile_dir_name
    if Path(infile_dir_path).exists():

        # input file names
        alphaSource_file = infile_dir_path + "/alphaSource.dat"
        beam_file = infile_dir_path + "/beam.dat"
        central_traj_file = infile_dir_path + "/centralTrajectory.dat"
        ion_chamber_file = infile_dir_path + "/IonChamber.dat"
        mwpc_file = infile_dir_path + "/mwpc.dat"
        rxn_file = infile_dir_path + "/reaction.dat"
        slits_file = infile_dir_path + "/slits.dat"
        target_degraders_file = infile_dir_path + "/targetDegraders.dat"

        # write user input to the input files, be careful of the variable parts
        # write alphaSource
        f = open(alphaSource_file, 'w')
        f.write(IFileTemplates.alphaSource_datfile.format(**form_dict))
        f.close()
        stdlogger.info("Wrote "+alphaSource_file)

        # write beam
        f = open(beam_file, 'w')
        f.write(IFileTemplates.beam_datfile.format(**form_dict))
        f.close()
        stdlogger.info("Wrote "+beam_file)

        # write central trajectory
        f = open(central_traj_file, 'w')
        f.write(IFileTemplates.centralTrajectory_datfile.format(**form_dict))
        f.close()
        stdlogger.info("Wrote "+central_traj_file)

        # write ion chamber
        f = open(ion_chamber_file, 'w')
        f.write(IFileTemplates.ionChamber_datfile.format(**form_dict))
        f.close()
        stdlogger.info("Wrote "+ion_chamber_file)

        # write mwpc
        f = open(mwpc_file, 'w')
        f.write(IFileTemplates.mwpc_datfile.format(**form_dict))
        f.close()
        stdlogger.info("Wrote "+mwpc_file)

        # write reaction
        f = open(rxn_file, 'w')
        f.write(IFileTemplates.reaction_datfile.format(**form_dict))
        f.close()
        stdlogger.info("Wrote "+rxn_file)

        # write slits
        f = open(slits_file, 'w')
        f.write(IFileTemplates.slits_datfile.format(**form_dict))
        f.close()
        stdlogger.info("Wrote "+slits_file)

        write_target_degraders(target_degraders_file, form_dict)
        stdlogger.info("Wrote "+target_degraders_file)

    else:
        stdlogger.error("Userdir does not exist, could not write input files")
        # raise error
        print("error: write_input_files, user dir path doesn't exist")



#---------------------------------------------------
# write_target_degraders
# PARAMETERS: Filename/path, the input form dictionary
#             containing the values to write
# PURPOSE: The targetDegraders file is a bit more complicated
#          because of the variable element lines so this
#          function takes care of that task
#---------------------------------------------------
def write_target_degraders(outfile=None, form_dict=None):
    # check that we have the params we need
    if (outfile is None or form_dict is None):
        #raise error
        print("write_target_degraders didn't get a file/input dictionary")
    else:
        # first build the element fields into one multiline string then insert that into the dictionary and feed it into the output template

        # target elements
        target_elems_str = ""

        for i in range(int(form_dict['target_num_elems'])):
            elem_key = "target_elem_" + str(i+1)
            target_elems_str += IFileTemplates.element_format_str.format(elem_fields=form_dict[elem_key], elem_num=i+1)

        form_dict['target_elems'] = target_elems_str


        # degrader 1 elements
        degrader_1_elems_str = ""

        for i in range(int(form_dict['degrader_1_num_elems'])):
            elem_key = "degrader_1_elem_" + str(i+1)
            degrader_1_elems_str += IFileTemplates.element_format_str.format(elem_fields=form_dict[elem_key], elem_num=i+1)

        form_dict['degrader_1_elems'] = degrader_1_elems_str


        # degrader 2 elements
        degrader_2_elems_str = ""

        for i in range(int(form_dict['degrader_2_num_elems'])):
            elem_key = "degrader_2_elem_" + str(i+1)
            degrader_2_elems_str += IFileTemplates.element_format_str.format(elem_fields=form_dict[elem_key], elem_num=i+1)

        form_dict['degrader_2_elems'] = degrader_2_elems_str


        f = open(outfile, 'w')
        f.write(IFileTemplates.targetDegraders_datfile.format(**form_dict))
        f.close()







#
