from django.conf.urls import url
from g4emma import views

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^about/$', views.about, name='about'),
    url(r'^manual/$', views.manual, name='manual'),
    url(r'^simulation/$', views.simulation, name='simulation'),
    url(r'^tools/$', views.tools, name='tools'),
    url(r'^results/$', views.results, name='results'),
    url(r'^progress/$', views.progress, name='progress'),
    url(r'^tools/rigidity/$', views.rigidity, name='rigidity'),
    url(r'^tools/energy_loss/$', views.energy_loss, name='energy_loss'),
    url(r'^tools/charge_state/$', views.charge_state, name='charge_state'),
    url(r'^tools/charge_state_results/$', views.charge_state_results, name='charge_state_results'),
    url(r'^tools/multiple_scattering/$', views.multiple_scattering, name='multiple_scattering'),
]
