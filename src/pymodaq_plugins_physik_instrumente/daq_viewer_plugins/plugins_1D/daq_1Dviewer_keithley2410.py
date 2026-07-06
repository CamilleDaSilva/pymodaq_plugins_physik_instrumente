# -*- coding: utf-8 -*-
"""
Plugin PyMoDAQ DAQ_1DViewer pour le Keithley 2410.
Lit la tension et le courant mesuré par le Keithley en retour de la tension sourcée.
Affiche la courbe I-V en temps réel.

Testé avec :
- Keithley 2410
- PyMoDAQ 5.2.6
- Windows 10
- Connexion GPIB via NI-VISA

Prérequis : installer pyvisa et les drivers NI-VISA
"""

import numpy as np

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_physik_instrumente.hardware.keithley2410_wrapper import Keithley2410


class DAQ_1DViewer_Keithley2410(DAQ_Viewer_base):
    """Plugin PyMoDAQ pour lire la tension et le courant du Keithley 2410.

    Affiche la courbe I-V (courant en fonction de la tension) en temps réel.
    """

    params = comon_parameters + [
        {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str', 'value': 'GPIB0::24::INSTR'},
        {'title': 'Voltage Min (V):', 'name': 'volt_min', 'type': 'float', 'value': -20.0},
        {'title': 'Voltage Max (V):', 'name': 'volt_max', 'type': 'float', 'value': 20.0},
        {'title': 'N points:', 'name': 'n_points', 'type': 'int', 'value': 50},
        {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float', 'value': 700e-3},
        {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float', 'value': 20e-3},
    ]

    def ini_attributes(self):
        self.controller: Keithley2410 = None
        self.x_axis = None

    def commit_settings(self, param: Parameter):
        """Applique les changements de paramètres depuis l'interface."""
        pass

    def ini_detector(self, controller=None):
        """Initialisation de la communication avec le Keithley 2410."""
        if self.is_master:
            self.controller = Keithley2410(self.settings['visa_address'])
            initialized = True
        else:
            self.controller = controller
            initialized = True

        # Axe X = tensions du balayage
        volts = self.controller.init_balayage(
            voltMin=self.settings['volt_min'],
            voltMax=self.settings['volt_max'],
            NV=self.settings['n_points'],
            compliance=self.settings['compliance'],
            current_range=self.settings['current_range']
        )
        self.x_axis = Axis(data=volts, label='Tension', units='V', index=0)

        # Initialise le panneau d'affichage
        self.dte_signal_temp.emit(DataToExport(
            name='Keithley2410',
            data=[DataFromPlugins(
                name='Courbe I-V',
                data=[np.zeros(len(volts))],
                dim='Data1D',
                labels=['Courant [A]'],
                axes=[self.x_axis]
            )]
        ))

        info = f"Keithley 2410 connecté sur {self.settings['visa_address']}"
        return info, initialized

    def close(self):
        """Ferme la communication avec le Keithley."""
        if self.is_master:
            self.controller.close()

    def grab_data(self, Naverage=1, **kwargs):
        """Fait un balayage complet et renvoie la courbe I-V."""
        volts = self.x_axis.get_data()
        courants = []

        for volt in volts:
            _, courant = self.controller.measure(volt)
            courants.append(courant)

        self.controller.output_off()

        self.dte_signal.emit(DataToExport(
            name='Keithley2410',
            data=[DataFromPlugins(
                name='Courbe I-V',
                data=[np.array(courants)],
                dim='Data1D',
                labels=['Courant [A]'],
                axes=[self.x_axis]
            )]
        ))

    def stop(self):
        """Arrête l'acquisition et éteint la sortie."""
        self.controller.output_off()
        self.emit_status(ThreadCommand('Update_Status', ['Acquisition arrêtée']))
        return ''


if __name__ == '__main__':
    main(__file__)