#===========================
# IMPORTS
#===========================
from django import forms
from django.forms.utils import flatatt
from django.utils.html import format_html
from django.core.validators import RegexValidator, MaxValueValidator, MinValueValidator


#===========================
# CUSTOM FIELDS AND WIDGETS
#===========================
# Hack from stackoverflow to enable me to add headers to forms
class HeaderWidget(forms.widgets.Widget):
    def render(self, name, value, attrs=None):
        attrs.update(self.attrs)
        return format_html('<div{0}>{1}</div>', flatatt(attrs), value)
    #see https://stackoverflow.com/questions/635583/inserting-a-heading-into-a-django-form


class ElementWidget(forms.MultiWidget):
    def __init__(self, widgets=None, *args, **kwargs):
        self.widgets = [
            forms.NumberInput(),
            forms.NumberInput(),
            forms.NumberInput()
        ]
        super(ElementWidget, self).__init__(self.widgets, *args, **kwargs)
    def decompress(self, value):
        if value:
            return value.split(' ')
        return [None, None, None]


class ElementField(forms.MultiValueField):
    widget = ElementWidget

    def __init__(self, *args, **kwargs):
        fields = (
            forms.IntegerField(
                error_messages={'incomplete': 'Enter a proton number.'}, label="Z", help_text=", ", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(1)]
            ),
            forms.DecimalField(
                error_messages={'incomplete': 'Enter a molar mass.'}, label="M", help_text="g/mol, ", validators=[MinValueValidator(1)]
            ),
            forms.DecimalField(
                error_messages={'incomplete': 'Enter a weight ratio.'}, label="\u03b7", help_text="%", validators=[MaxValueValidator(100), MinValueValidator(0)]
            ),
        )
        super(ElementField, self).__init__(
            fields=fields, require_all_fields=True, *args, **kwargs
        )

    def compress(self, data_list):
        # Documentation says we can assume that data is valid
        return ' '.join(str(x) for x in data_list)


#===========================
# FORM CLASSES
#===========================

MAX_PROTON_NUM = 300
MAX_NUCLEON_NUM = 500
MIN_THICKNESS = 0.000004
MAX_EVENTS = 5000
MIN_CS_BEAM_Z = 8 # charge state tool constant
MAX_TARGET_Z = 79 # charge state tool constant
MIN_TARGET_Z = 4  # charge state tool constant
# Note: I use 0 and 1 instead of booleans to allow the JS to use them easily

class AlphaSourceChoiceForm(forms.Form):
    name = "alpha_source_choice_form"
    ALPHA_SOURCE_CHOICES = (
        (0, "No"), #the first value is the actual value in the code
        (1, "Yes") # the second is the value the user sees
    )
    # toggle specify/zero
    alpha_source_present = forms.ChoiceField(required=True, label="Alpha source", help_text="(Beam/reaction will be simulated with alpha source parameters)", choices = ALPHA_SOURCE_CHOICES, initial=0)

class AlphaSourceForm(forms.Form):
    name = "alpha_source_form"
    alpha_source_kinetic_e = forms.DecimalField(required=False, label="\u03b1E", help_text="MeV (Alpha source kinetic energy)", validators=[MinValueValidator(0)])
    alpha_source_max_angle = forms.DecimalField(required=False, label="Max\u03b8", help_text="deg (Max angle alpha source)", validators=[MaxValueValidator(360), MinValueValidator(0)])



class BeamForm(forms.Form):
    name = "beam_form"
    num_events = forms.IntegerField(required=True, label="n", help_text="(Number of events) *required", validators=[MaxValueValidator(MAX_EVENTS, message=('Ensure this value is less than or equal to %(limit_value)s (to reduce chance of timeouts).')), MinValueValidator(1)])
    beam_proton_num = forms.IntegerField(required=True, label="Z", help_text="(Proton number) *required", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(0)])
    beam_nucleon_num = forms.IntegerField(required=True, label="A", help_text="(Nucleon number) *required", validators=[MaxValueValidator(MAX_NUCLEON_NUM), MinValueValidator(1)])
    beam_charge_state = forms.IntegerField(required=True, label="Q", help_text="(Charge state) *required")
    beam_kinetic_e = forms.DecimalField(required=True, label="E", help_text="MeV (Kinetic energy) *required", validators=[MinValueValidator(0)])


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
    beam_e_spread = forms.DecimalField(required=False, label="\u03b4E/E", help_text="% (FWHM, beam energy spread)", initial=0.1, validators=[MinValueValidator(0)])
    beam_diameter = forms.DecimalField(required=False, label="d", help_text="mm (beam diameter)", initial=1)
    beam_trans_emittance = forms.DecimalField(required=False, label="\u03b3\u03b2\u03b5", help_text="\u03c0 mm mrad (Beam transverse emittance)") # 0.3 mm pi mrad over beta default...


class CentralTrajectoryChoiceForm(forms.Form):
    name = "central_traj_choice_form"
    CENTRAL_TRAJECTORY_CHOICES = (
        (0, "Same as beam"),
        (1, "Specify")
    )
    specify_central_trajectory = forms.ChoiceField(required=True,
    label="Central trajectory", choices = CENTRAL_TRAJECTORY_CHOICES,
    initial=0)

class CentralTrajectoryForm(forms.Form):
    name = "central_traj_form"
    center_traj_proton_num = forms.IntegerField(required=False, label="ZC", help_text="(Proton number)", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(1)])
    center_traj_nucleon_num = forms.IntegerField(required=False, label="AC", help_text="(Nucleon number)", validators=[MaxValueValidator(MAX_NUCLEON_NUM), MinValueValidator(1)])
    center_traj_charge_state = forms.IntegerField(required=False, label="QC", help_text="(Charge state)")
    center_traj_kinetic_e = forms.DecimalField(required=False, label="EC", help_text="MeV (Kinetic Energy)", validators=[MinValueValidator(0)])


class ReactionChoiceForm(forms.Form):
    name = "reaction_choice_form"
    REACTION_CHOICES = (
        (0, "No"),
        (1, "Yes")
    )
    specify_reaction = forms.ChoiceField(required=True, label="Two-body reaction", choices=REACTION_CHOICES, initial=0)

class ReactionForm(forms.Form):
    name = "reaction_form"
    header_line1 = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Reaction: 1+2 --> 3+4')
    header_line2 = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Z1 = Z', label_suffix="")
    header_line3 = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='A1 = A', label_suffix="")

    rxn_z2_target = forms.IntegerField(required=False, label="Z2", help_text="(Z2 target) *required", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(0)])
    rxn_a2 = forms.IntegerField(required=False, label="A2", help_text="(A2 target) *required", validators=[MaxValueValidator(MAX_NUCLEON_NUM), MinValueValidator(1)])
    rxn_z3_recoil = forms.IntegerField(required=False, label="Z3", help_text="(Z3 recoil)", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(0)])
    rxn_a3 = forms.IntegerField(required=False, label="A3", help_text="(A3 recoil)", validators=[MaxValueValidator(MAX_NUCLEON_NUM), MinValueValidator(1)])
    rxn_z4_ejectile = forms.IntegerField(required=False, label="Z4", help_text="(Z4 ejectile)", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(0)])
    rxn_a4 = forms.IntegerField(required=False, label="A4", help_text="(A4 ejectile)", validators=[MaxValueValidator(MAX_NUCLEON_NUM), MinValueValidator(1)])

    header_line4 = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Properties of recoil (#3):')
    rxn_min_angle = forms.DecimalField(required=False, label="\u03b8cm,min", help_text="deg (min cm angle of ejectile)", validators=[MaxValueValidator(360), MinValueValidator(0)])
    rxn_max_angle = forms.DecimalField(required=False, label="\u03b8cm,max", help_text="deg (max cm angle of ejectile)", validators=[MaxValueValidator(360), MinValueValidator(0)])
    rxn_recoil_charge = forms.IntegerField(required=False, label="Q3", help_text="(charge state of recoil)")
    rxn_recoil_excitation_e = forms.DecimalField(required=False, label="Ex", help_text="MeV (excitation energy of recoil)")

    header_line5 = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Cross section:')
    rxn_cross_sec = forms.DecimalField(required=False, label="d\u03c3/d\u03a9cm", help_text="mb/sr (solid-angle averaged cross section)")


class TargetChoiceForm(forms.Form):
    name = "target_choice_form"
    TARGET_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    target_inserted = forms.ChoiceField(required=True, label="Target", choices=TARGET_CHOICES, initial=0)

class TargetForm(forms.Form):
    name = "target_form"
    target_thickness = forms.DecimalField(required=False, label="x0", help_text="\u03bcm (thickness)", validators=[MinValueValidator(MIN_THICKNESS)]) # this min value is determined by geant4 step size, this value is what I found to be close to the lower bound
    # target_z_pos = forms.DecimalField(required=False, label="z-pos", help_text="cm (target z position offset)") # should not be changeable
    target_density = forms.DecimalField(required=False, label="\u03c1", help_text="g/cm³ (target density)")

class TargetElementsForm(forms.Form):
    name = "target_elements_form"
    NUM_ELEM_CHOICES = (
        (1, "1"),
        (2, "2"),
        (3, "3"),
        (4, "4"),
        (5, "5")
    )
    target_num_elems = forms.ChoiceField(required=True, label="Number of elements", choices=NUM_ELEM_CHOICES, initial=1)
    elem_guideline = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Element #: Z = proton number, M = molar mass (g/mol), \u03b7 = weight ratio (%)', label_suffix="")
    target_elem_1 = ElementField(required=False, label="Element 1")
    target_elem_2 = ElementField(required=False, label="Element 2")
    target_elem_3 = ElementField(required=False, label="Element 3")
    target_elem_4 = ElementField(required=False, label="Element 4")
    target_elem_5 = ElementField(required=False, label="Element 5")


class Degrader1ChoiceForm(forms.Form):
    name = "degrader_1_choice_form"
    DEGRADER_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    degrader_1_inserted = forms.ChoiceField(required=True, label="Degrader 1", choices=DEGRADER_CHOICES, initial=0)

class Degrader1Form(forms.Form):
    name = "degrader_1_form"
    degrader_1_thickness = forms.DecimalField(required=False, label="x0", help_text="\u03bcm (thickness)", validators=[MinValueValidator(MIN_THICKNESS)]) #this min thickness is similar across geant4 i hope
    degrader_1_density = forms.DecimalField(required=False, label="\u03c1", help_text="g/cm³ (degrader 1 density)")

class Degrader1ElementsForm(forms.Form):
    name = "degrader_1_elements_form"
    NUM_ELEM_CHOICES = (
        (1, "1"),
        (2, "2"),
        (3, "3"),
        (4, "4"),
        (5, "5")
    )
    degrader_1_num_elems = forms.ChoiceField(required=True, label="Number of elements", choices=NUM_ELEM_CHOICES, initial=1)
    elem_guideline = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Element #: Z = proton number, M = molar mass (g/mol), \u03b7 = weight ratio (%)', label_suffix="")
    degrader_1_elem_1 = ElementField(required=False, label="Element 1")
    degrader_1_elem_2 = ElementField(required=False, label="Element 2")
    degrader_1_elem_3 = ElementField(required=False, label="Element 3")
    degrader_1_elem_4 = ElementField(required=False, label="Element 4")
    degrader_1_elem_5 = ElementField(required=False, label="Element 5")


class Degrader2ChoiceForm(forms.Form):
    name = "degrader_2_choice_form"
    DEGRADER_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    degrader_2_inserted = forms.ChoiceField(required=True, label="Degrader 2", choices=DEGRADER_CHOICES, initial=0)

class Degrader2Form(forms.Form):
    name = "degrader_2_form"
    degrader_2_thickness = forms.DecimalField(required=False, label="x0", help_text="\u03bcm (thickness)", validators=[MinValueValidator(MIN_THICKNESS)]) #this min thickness is similar across geant4 i hope
    degrader_2_density = forms.DecimalField(required=False, label="\u03c1", help_text="g/cm³ (degrader 2 density)")

class Degrader2ElementsForm(forms.Form):
    name = "degrader_2_elements_form"
    NUM_ELEM_CHOICES = (
        (1, "1"),
        (2, "2"),
        (3, "3"),
        (4, "4"),
        (5, "5")
    )
    degrader_2_num_elems = forms.ChoiceField(required=True, label="Number of elements", choices=NUM_ELEM_CHOICES, initial=1)
    elem_guideline = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Element #: Z = proton number, M = molar mass (g/mol), \u03b7 = weight ratio (%)', label_suffix="")
    degrader_2_elem_1 = ElementField(required=False, label="Element 1")
    degrader_2_elem_2 = ElementField(required=False, label="Element 2")
    degrader_2_elem_3 = ElementField(required=False, label="Element 3")
    degrader_2_elem_4 = ElementField(required=False, label="Element 4")
    degrader_2_elem_5 = ElementField(required=False, label="Element 5")


class Slit1ChoiceForm(forms.Form):
    name = "slit_1_choice_form"
    SLIT_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    slit1_inserted = forms.ChoiceField(required=True, label="Slit 1", choices=SLIT_CHOICES, initial=0)

class Slit1Form(forms.Form):
    name = "slit_1_form"
    slit_1_inserted = forms.DecimalField(required=False, label="Aperture (+/-)", help_text="mm")


class Slit2ChoiceForm(forms.Form):
    name = "slit_2_choice_form"
    SLIT_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    slit2_inserted = forms.ChoiceField(required=True, label="Slit 2", choices=SLIT_CHOICES, initial=0)

class Slit2Form(forms.Form):
    name = "slit_2_form"
    slit_2_inserted = forms.DecimalField(required=False, label="Aperture (+/-)", help_text="mm")


class Slit3ChoiceForm(forms.Form):
    name = "slit_3_choice_form"
    SLIT_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    slit3_inserted = forms.ChoiceField(required=True, label="Slit 3", choices=SLIT_CHOICES, initial=0)

class Slit3Form(forms.Form):
    name = "slit_3_form"
    slit_3_inserted = forms.DecimalField(required=False, label="Aperture (+/-)", help_text="mm")


class Slit4ChoiceForm(forms.Form):
    name = "slit_4_choice_form"
    SLIT_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    slit4_inserted = forms.ChoiceField(required=True, label="Slit 4", choices=SLIT_CHOICES, initial=0)

class Slit4Form(forms.Form):
    name = "slit_4_form"
    slit_4_inserted = forms.DecimalField(required=False, label="Aperture (+/-)", help_text="mm")


class MWPCForm(forms.Form):
    name = "mwpc_form"
    mwpc_guide = forms.CharField(widget=HeaderWidget(attrs={'class': 'form-header'}), initial='', required=False, label='Multiwire Proportional Counter (MWPC)', label_suffix="")
    mwpc_pressure = forms.DecimalField(required=True, label="p", help_text="Torr *required", validators=[MinValueValidator(0)])
    mwpc_temp = forms.DecimalField(required=True, label="T", help_text="\u00B0C *required", validators=[MinValueValidator(-273.15)])


class IonChamberChoiceForm(forms.Form):
    name = "ion_chamber_choice_form"
    ION_CHAMBER_CHOICES = (
        (0, "Out"),
        (1, "In")
    )
    ion_chamber_inserted = forms.ChoiceField(required=True, label="Ion chamber", choices=ION_CHAMBER_CHOICES, initial=0)

class IonChamberForm(forms.Form):
    name = "ion_chamber_form"
    ion_chamber_pressure = forms.DecimalField(required=False, label="p", help_text="Torr", validators=[MinValueValidator(0)])
    ion_chamber_temp = forms.DecimalField(required=False, label="T", help_text="\u00B0C", validators=[MinValueValidator(-273.15)])

#########################################
# Rigidity Tool Form
########################################

class RigidityForm(forms.Form):
    name = "rigidity_form"
    rig_kinetic_e = forms.DecimalField(required=True, label="E", help_text="MeV (Kinetic energy) *required",validators=[MinValueValidator(0)])
    rig_nucleon_num = forms.DecimalField(required=True, label="A", help_text="amu (Mass) *required",validators=[MaxValueValidator(MAX_NUCLEON_NUM), MinValueValidator(1)])
    rig_charge_state = forms.IntegerField(required=True, label="Q", help_text="(Charge state) *required")

#########################################
# Multiple Scattering Tool Form
########################################

class MultipleScatteringForm(forms.Form): 
    name = "multiple_scattering_form"
    ms_kinetic_e = forms.DecimalField(required=True, label="E", help_text="MeV (Incident Kinetic energy) *required", validators=[MinValueValidator(0)])
    ms_rest_mass = forms.DecimalField(required=True, label="M", help_text="amu (Incident rest mass) *required", validators=[MinValueValidator(0)])
    ms_incident_proton_num = forms.IntegerField(required=True, label="z", help_text=" (Incident atomic number) *required",validators=[MinValueValidator(1)])
    ms_target_proton_num = forms.IntegerField(required=True, label="Z", help_text=" (Target atomic number) *required",validators=[MinValueValidator(1)])
    ms_target_nucleon_num = forms.IntegerField(required=True, label="A", help_text=" (Target nucleon number) *required",validators=[MinValueValidator(1)])
    ms_target_density = forms.DecimalField(required=True, label="t", help_text = "μg/cm**2 (Target areal density) *required", validators=[MinValueValidator(0)])

##########################################
# Charge State Form
##########################################

class ChargeStateIonForm(forms.Form):
    name = "charge_state_ion_form"
    cs_kinetic_e = forms.DecimalField(required=True, label="E", help_text="MeV (Kinetic energy) *required", validators=[MinValueValidator(0)])
    cs_proton_num = forms.IntegerField(required=True, label="Z", help_text="(Proton number) *required", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(MIN_CS_BEAM_Z)])
    cs_nucleon_num = forms.DecimalField(required=True, label="A", help_text="(Nucleon number) *required", validators=[MaxValueValidator(MAX_NUCLEON_NUM), MinValueValidator(1)])

    def clean(self):
        cd = super(ChargeStateIonForm, self).clean()
        cs_kinetic_e = cd.get('cs_kinetic_e', None)
        cs_nucleon_num = cd.get('cs_nucleon_num', None)
        if not cs_nucleon_num == int(cs_nucleon_num):
            raise forms.ValidationError("Assertion Error, nucleon number must be integer")
        energy_per_nucleon = float(cs_kinetic_e)/float(cs_nucleon_num)
        if energy_per_nucleon >= 6.0:
            raise forms.ValidationError("Assertion Error, Energy per nucleon is too high.")
        return cd

class ChargeStateTargetForm(forms.Form):
    name = "charge_state_target_form"
    cs_T_proton_num = forms.IntegerField(required=True, label="Z", help_text="(Proton number) *required", validators=[MaxValueValidator(MAX_TARGET_Z), MinValueValidator(MIN_TARGET_Z)])

########################################################
# Energy Loss Forms
########################################################

class EnergyLossIonForm(forms.Form):
    name = "energy_loss_ion_form"
    el_nucleon_num = forms.IntegerField(required=True, label="A",help_text="(Nucleon number) *required", validators=[MaxValueValidator(MAX_NUCLEON_NUM),MinValueValidator(1)])
    el_proton_num = forms.IntegerField(required=True, label="Z", help_text="(Proton number) *required", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(1)])
    el_energy = forms.DecimalField(required=True, label="E", help_text="MeV (Kinetic energy) *required", validators=[MinValueValidator(0)])

class EnergyLossTargetChoiceForm(forms.Form):
    name = "energy_loss_target_choice_form"
    TARGET_CHOICES = ((1, ""),(2,"Pre-defined Compound"),(3,"Self-defined Element"), (4,"Self-defined Compound"))
    target_material_choice = forms.ChoiceField(required=True, label="Target Material",help_text=" *required",choices=TARGET_CHOICES, initial=1)

class EnergyLossPreDefinedChoiceForm(forms.Form):
    name = "energy_loss_pre_defined_target_choice_form"
    PRE_DEF_CHOICES = ((1, ""), (2, "isobutane"), (3, "mylar"), (4, "polyethylene"), (5,"CD2"), (6, "LiF"), (7, "propanediol"))
    pre_defined_mat_choice = forms.ChoiceField(required=False, label="Pre-defined Compound", choices=PRE_DEF_CHOICES, initial=1)

class EnergyLossPressureForm(forms.Form):
    name = "energy_loss_pressure_form"
    el_pressure = forms.DecimalField(required=False, label="P", help_text="Torr (Gas pressure) *required", validators=[MinValueValidator(0)])

class EnergyLossSelfDefinedElementForm(forms.Form):
    name = "energy_loss_target_element_form"
    target_element_proton_num = forms.IntegerField(required=False, label="Z", help_text="(Proton number) *required", validators=[MaxValueValidator(MAX_PROTON_NUM), MinValueValidator(1)])

class EnergyLossSelfDefinedCompoundForm(forms.Form):
    name = "energy_loss_target_compound_form"
    target_compound = forms.CharField(required=False,label="Self-defined Compound",help_text="using CaseLetterDigit format (ex: water would be H2O & salt would be NaCl)", max_length=15)

class EnergyLossDensityForm(forms.Form):
    name = "energy_loss_density_form"
    target_compound_density = forms.DecimalField(required=False,label="Compound Density",help_text="g/cm^3 *required", validators=[MinValueValidator(0)])

class EnergyLossTargetThicknessForm(forms.Form):
    name = "energy_loss_target_thickness_form"
    target_thickness = forms.DecimalField(required=True, label="Target Thickness", validators=[MinValueValidator(0)])

class EnergyLossTargetThicknessUnitChoiceForm(forms.Form):
    name = "energy_loss_thickness_unit_choice_form"
    UNIT_CHOICES = ((0,"ug/cm^2"),(1,"um"))
    thickness_unit_choice = forms.ChoiceField(required=True, label="", help_text=" *required", choices=UNIT_CHOICES, initial=0)



 










#
