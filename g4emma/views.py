from django.http import Http404, HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
import g4emma.forms as G4Forms
import subprocess as sp
from pathlib import Path
import g4emma.g4emma_input_setup as G4ISetup
from django.conf import settings
from os import environ
from channels import Channel
from django.core import serializers
from pathlib import Path
from os import environ
import logging

stdlogger = logging.getLogger('django')


def home(request):
    stdlogger.info("Call to home view")
    return render(request, 'g4emma/home.html')

def about(request):
    stdlogger.info("Call to about view")
    return render(request, 'g4emma/about.html')

def manual(request):
    stdlogger.info("Call to manual view")
    return render(request, 'g4emma/manual.html')

def simulation(request):
    stdlogger.info("Call to simulation view")
    results = "" #if there has been no post request, there are no results

    forms_list = [G4Forms.AlphaSourceChoiceForm,
    G4Forms.AlphaSourceForm,
    G4Forms.BeamForm,
    G4Forms.BeamEmittanceChoiceForm,
    G4Forms.BeamEmittanceForm,
    G4Forms.CentralTrajectoryChoiceForm,
    G4Forms.CentralTrajectoryForm,
    G4Forms.ReactionChoiceForm,
    G4Forms.ReactionForm,
    G4Forms.TargetChoiceForm,
    G4Forms.TargetForm,
    G4Forms.TargetElementsForm,
    G4Forms.Degrader1ChoiceForm,
    G4Forms.Degrader1Form,
    G4Forms.Degrader1ElementsForm,
    G4Forms.Degrader2ChoiceForm,
    G4Forms.Degrader2Form,
    G4Forms.Degrader2ElementsForm,
    G4Forms.Slit1ChoiceForm,
    G4Forms.Slit1Form,
    G4Forms.Slit2ChoiceForm,
    G4Forms.Slit2Form,
    G4Forms.Slit3ChoiceForm,
    G4Forms.Slit3Form,
    G4Forms.Slit4ChoiceForm,
    G4Forms.Slit4Form,
    G4Forms.MWPCForm,
    G4Forms.IonChamberChoiceForm,
    G4Forms.IonChamberForm
    ]

    # if a form was submitted (using POST)
    if request.method == 'POST':
        stdlogger.info("POST request in simulation view")
        forms_are_all_valid = True

        for index, input_form in enumerate(forms_list):
            #setup the forms
            forms_list[index] = input_form(request.POST)

            #test their validity (false if any form is not valid)
            forms_are_all_valid = forms_are_all_valid and forms_list[index].is_valid()


        if forms_are_all_valid:
            stdlogger.info("Forms are all valid")
            sim_params = {} #setup a blank start

            for input_form in forms_list:
                #agglomerate all the forms' input into one dictionary
                sim_params.update(input_form.cleaned_data)

            # Do some cleanup before adding another user dir
            stdlogger.info("Starting cleanup of old user directories")
            G4ISetup.cleanup_old_userdirs()

            # Setup a user directory, save its path
            user_dirs_path = settings.DATA_DIRS
            userdir = G4ISetup.setup_unique_userdir(user_dirs_path)
            userdir_path = "{}{}".format(user_dirs_path, userdir)

            stdlogger.info("Unique user dir created: " + userdir_path)


            # Overlay the user input on a set of default values so that we have a complete input set
            sim_params = G4ISetup.merge_with_defaults(sim_params)
            stdlogger.info("User params overlayed on defaults")


            # write to input files
            G4ISetup.write_input_files(userdir_path, sim_params)
            stdlogger.info("User input written to input files")

            # Build call to simulation wrapper
            wrapper_path = environ['G4EMMA_WRAPPER']
            command = " ".join((wrapper_path, environ['G4EMMA_SIM_PATH'], userdir_path + "/"))  #this last slash is important!!!

            # Store data in the session (everything the sim start consummer needs)
            request.session['cmd'] = command
            request.session['userdir'] = userdir
            request.session['userdir_path'] = userdir_path

            # The forms are not JSON serializable
            # TODO Fix this...
            # request.session['forms_list'] = data


            stdlogger.info("About to call simulation wrapper: "+command)

            # Send sim start msg on that consummer's channel
            Channel("sim_start_channel").send({
                'text': "start",
                'cmd': command,
                'userdir': userdir,
            })

            # There are multiple return statements in this function
            return redirect('progress')

        # forms are not all valid so send users back
        else:
            return render(request, 'g4emma/simulation.html', {'forms_list': forms_list})


    # If not POST
    else:
        # If rigidities.dat exists in this branch it means a simulation
        # error occured and we need to display it to the user
        if ('userdir_path' in request.session and
            Path(request.session['userdir_path']+"/Results/rigidities.dat").exists()):
            # let user know that something went wrong (give some ideas of what it could be)
            err_msg = ("An error occured when trying to run the simulation. Check that elements chosen are possible, "
            "and that the magnetic and electric rigidities determined by central "
            "trajectory parameters do not exceed maximum allowed values.\n\n")

            stdlogger.info("Checking the rigidities file for error msgs")

            rigidity_err_msgs = ""
            # read rigidities file and set form errors render form
            with open(request.session['userdir_path']+"/Results/rigidities.dat", 'r') as r_file:
                magnetic_rigidity = r_file.readline() #the first two lines are constant
                electric_rigidity = r_file.readline()
                # then will be 2-4 warning/error lines
                rigidity_err_msgs = r_file.read()

                rigidity_err_msgs = "{}\n{}\n{}".format(magnetic_rigidity,
                electric_rigidity,
                rigidity_err_msgs)

            # If we've gotten this far we should clear the http session so that
            # the next request for a clean form doesn't get misinterpreted as
            # an errored form

            # pull out what we need
            # TODO Fix non-JSON-serializable forms problem
            # forms_list = request.session.pop('forms_list', {})
            for index, input_form in enumerate(forms_list):
                #setup the forms
                forms_list[index] = input_form()

            # then clear away the rest
            request.session.clear()

            # There are multiple return statements in this function
            return render(request, 'g4emma/simulation.html',
                {'forms_list': forms_list, 'general_err_msg': err_msg, 'rigidity_err_msg': rigidity_err_msgs})

        # No rigidities.dat => no error => new empty form
        else:
            for index, input_form in enumerate(forms_list):
                #setup the forms
                forms_list[index] = input_form()


    # There are multiple return statements in this function
    return render(request, 'g4emma/simulation.html', {'forms_list': forms_list})


def tools(request):
    stdlogger.info("Call to tools view")
    return render(request, 'g4emma/tools.html')


def results(request):
    stdlogger.info("Call to results view")
    outfiles_list = {}
    outplots_list = {}
    outfiledir = ""
    outplotdir= ""

    if ('userdir' in request.session):
        # Prep the info we need
        outdir = "/media/userdirs/"+ request.session['userdir']
        outfiledir = outdir+"/Results/"
        outplotdir = outdir+"/Plots/"

        stdlogger.info("About to call ROOT macro wrapper")
        sp.call("$G4EMMA_ROOT_MACRO_WRAPPER_PATH "+request.session['userdir_path'], shell=True)

        #get a list of the generated output files
        outfiles = str(sp.check_output("ls -l "+ request.session['userdir_path'] +"/Results/ | awk '{print $9;}'", shell=True, universal_newlines=True))
        outplots = str(sp.check_output("ls -l "+ request.session['userdir_path'] +"/Plots/ | awk '{print $9;}'", shell=True, universal_newlines=True))

        # make a list from that command's output
        outfiles_list = outfiles.strip().splitlines()
        outplots_list = outplots.strip().splitlines()

        # Clear the session (leftover stuff is interpreted as indication of
        # a sim error if the user goes back to the simulation page)
        request.session.clear()

    return render(request, 'g4emma/results.html',
        {'outfiledir': outfiledir, 'outplotdir':outplotdir,
         'outfiles': outfiles_list, 'outplots': outplots_list})


def progress(request):
    # Make sure the user should be getting this page
    if ('userdir_path' in request.session):
        # Fetch number of events from the user input
        num_events = 0
        with open(request.session['userdir_path']+"/UserInput/beam.dat", 'r') as f:
            # should be the first token on the first line
            num_events = int(f.readline().split()[0])

        z2 = ""
        a2 = ""
        # Number of events also depends on whether there is a rxn specified
        with open(request.session['userdir_path']+"/UserInput/reaction.dat", 'r') as f:
            # should be the first token on the first line
            f.readline() # top comment
            f.readline() # z1
            f.readline() # a1
            z2 = f.readline().split()[0]
            a2 = f.readline().split()[0]

        # If there is a rxn, we have 3 sets of events: do beam, do prepare, and do rxn
        if (z2 != "0" or a2 != "0"):
            # I don't know if this is the logic they meant to express but
            # I'm mirroring what's in the simulation app so it's consistent
            num_events *= 3

        return render(request, 'g4emma/progress.html', { 'num_events': num_events})

    # The user shouldn't be accessing this page
    else:
        # So send them elsewhere
        return redirect('simulation')
