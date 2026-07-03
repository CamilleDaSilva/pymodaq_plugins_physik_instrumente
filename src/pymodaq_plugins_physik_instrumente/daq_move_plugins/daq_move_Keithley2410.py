# -*- coding: utf-8 -*-
"""
Plugin PyMoDAQ pour le Keithley 2410 (sourcemètre).
Permet de sourcer une tension et de lire le courant mesuré en retour.

Testé avec :
- Keithley 2410
- PyMoDAQ 5.2.6
- Windows 10
- Connexion GPIB via NI-VISA

Prérequis : installer pyvisa et les drivers NI-VISA
"""

from typing import Union, List

from pymodaq.control_modules.move_utility_classes import (DAQ_Move_base, comon_parameters_fun,
                                                          main, DataActuatorType, DataActuator)
from pymodaq_utils.utils import ThreadCommand
from pymodaq_gui.parameter import Parameter

from pymodaq_plugins_physik_instrumente.hardware.keithley2410_wrapper import Keithley2410


class DAQ_Move_Keithley2410(DAQ_Move_base):
    """Plugin PyMoDAQ pour le Keithley 2410.

    Le Keithley 2410 est utilisé ici comme source de tension (actuateur).
    Il source une tension et mesure le courant en retour.
    La valeur "position" correspond à la tension appliquée en Volts.
    """

    is_multiaxes = False
    _axis_names: Union[List[str]] = ['Voltage']
    _controller_units: str = 'V'
    _epsilon: float = 0.01

    data_actuator_type = DataActuatorType.DataActuator

    params = [
        {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str', 'value': 'GPIB0::24::INSTR'},
        {'title': 'Voltage Min (V):', 'name': 'volt_min', 'type': 'float', 'value': -20.0},
        {'title': 'Voltage Max (V):', 'name': 'volt_max', 'type': 'float', 'value': 20.0},
        {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float', 'value': 0.7},
        {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float', 'value': 20e-3},
    ] + comon_parameters_fun(is_multiaxes, axis_names=_axis_names, epsilon=_epsilon)

    def ini_attributes(self):
        self.controller: Keithley2410 = None

    def ini_stage(self, controller=None):
        """Initialisation de la communication avec le Keithley 2410."""
        if self.is_master:
            self.controller = Keithley2410(self.settings['visa_address'])
            self.controller.init_balayage(
                voltMin=self.settings['volt_min'],
                voltMax=self.settings['volt_max'],
                NV=50,
                compliance=self.settings['compliance'],
                current_range=self.settings['current_range']
            )
            initialized = True
        else:
            self.controller = controller
            initialized = True

        info = f"Keithley 2410 connecté sur {self.settings['visa_address']}"
        return info, initialized

    def get_actuator_value(self) -> DataActuator:
        """Lit la tension actuellement sourcée par le Keithley."""
        try:
            voltage = self.controller.get_voltage()
        except Exception:
            voltage = 0.0
        pos = DataActuator(data=voltage, units=self.axis_unit)
        pos = self.get_position_with_scaling(pos)
        return pos

    def move_abs(self, value: DataActuator):
        """Applique une tension absolue sur le Keithley."""
        value = self.check_bound(value)
        self.target_value = value
        value = self.set_position_with_scaling(value)

        voltage = value.value(self.axis_unit)
        self.controller.set_voltage(voltage)
        self.emit_status(ThreadCommand('Update_Status', [f'Tension appliquée : {voltage} V']))

    def move_rel(self, value: DataActuator):
        """Applique une tension relative par rapport à la position actuelle."""
        value = self.check_bound(self.current_position + value) - self.current_position
        self.target_value = value + self.current_position
        value = self.set_position_relative_with_scaling(value)

        voltage = self.target_value.value(self.axis_unit)
        self.controller.set_voltage(voltage)
        self.emit_status(ThreadCommand('Update_Status', [f'Tension appliquée : {voltage} V']))

    def move_home(self):
        """Remet la tension à 0 V."""
        self.controller.set_voltage(0)
        self.emit_status(ThreadCommand('Update_Status', ['Retour à 0 V']))

    def stop_motion(self):
        """Éteint la sortie du Keithley."""
        self.controller.output_off()
        self.emit_status(ThreadCommand('Update_Status', ['Sortie éteinte']))
        self.move_done()

    def close(self):
        """Ferme la communication avec le Keithley."""
        if self.is_master:
            self.controller.close()

    def commit_settings(self, param: Parameter):
        """Applique les changements de paramètres depuis l'interface."""
        if param.name() == 'compliance':
            self.controller.instrument.write(f':SENS:CURR:PROT {param.value()}')
        elif param.name() == 'current_range':
            self.controller.instrument.write(f':SENS:CURR:RANG {param.value()}')


if __name__ == '__main__':
    main(__file__)