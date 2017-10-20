from django import forms

# Note: I use 0 and 1 instead of booleans to allow the JS to use them easily

class BeamForm(forms.Form):
    name = "beam_form"
    num_events = forms.IntegerField(required=True, label="n", help_text="(Number of events)")
    beam_proton_num = forms.IntegerField(required=True, label="Z", help_text="(Proton number)")
    beam_nucleon_num = forms.IntegerField(required=True, label="A", help_text="(Nucleon number)")
    beam_charge_state = forms.IntegerField(required=True, label="Q", help_text="(Charge state)")
    beam_kinetic_e = forms.DecimalField(required=True, label="E", help_text="MeV (Kinetic energy)")


class BeamEmittanceChoiceForm(forms.Form):
    name = "beam_emittance_choice_form"
    BEAM_EMITTANCE_CHOICES = (
        (0, "Zero"), #the first value is the actual value in the code
        (1, "Specify") # the second is the value the user sees
    )
    # toggle specify/zero
    specify_beam_emittance = forms.ChoiceField(required=True, label="Beam emittance", choices = BEAM_EMITTANCE_CHOICES, initial=0)

class BeamEmittanceForm(forms.Form):
    name = "beam_emittance_form"
    beam_e_spread = forms.DecimalField(label="\u03b4E/E", help_text="% (FWHM, beam energy spread)", required=False)
    beam_diameter = forms.DecimalField(label="d", help_text="mm (beam diameter)", required=False)
    beam_trans_emittance = forms.DecimalField(label="\u03b3\u03b2\u03b5", help_text="\u03c0 mm mrad (Beam transverse emittance)", required=0)


class CentralTrajectoryChoiceForm(forms.Form):
    name = "central_traj_choice_form"
    CENTRAL_TRAJECTORY_CHOICES = (
        (0, "Same as beam"),
        (1, "Specify")
    )
    specify_central_trajectory = forms.ChoiceField(required=True,
    label="Central trajectory", choices = CENTRAL_TRAJECTORY_CHOICES,
    initial=1)

class CentralTrajectoryForm(forms.Form):
    name = "central_traj_form"
    center_traj_proton_num = forms.IntegerField(label="ZC", help_text="(Proton number)", required=False)
    center_traj_nucleon_num = forms.IntegerField(label="AC", help_text="(Nucleon number)", required=False)
    center_traj_charge_state = forms.IntegerField(label="QC", help_text="(Charge state)", required=False)
    center_traj_kinetic_e = forms.DecimalField(label="EC", help_text="MeV (Kinetic Energy)", required=False)
