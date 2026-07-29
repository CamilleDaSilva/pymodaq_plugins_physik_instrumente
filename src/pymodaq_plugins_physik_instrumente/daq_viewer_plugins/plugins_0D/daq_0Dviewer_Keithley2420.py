# -*- coding: utf-8 -*-
"""
PyMoDAQ DAQ_0DViewer plugin for the Keithley 2420.
Reads a single collector current value on each grab_data() call.
Meant to be used as the detector in a DAQ_Scan, alongside a DAQ_Move
actuator (e.g. Keithley 2410) that steps the RPA grid voltage. DAQ_Scan
itself reconstructs the I-V curve point by point from the two modules.

Author: Camille Da Silva
ONERA DPHY/CSE — PICOMAX-E, 2026
"""

import numpy as np

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_physik_instrumente.hardware.keithley2420_wrapper import Keithley2420


class DAQ_0DViewer_Keithley2420(DAQ_Viewer_base):
    """PyMoDAQ plugin to read the collector current of the Keithley 2420 (RPA)."""

    params = comon_parameters + [

        {'title': 'Keithley Settings', 'name': 'keithley_settings', 'type': 'group', 'children': [
            {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str',
             'value': 'GPIB0::17::INSTR',
             'tip': 'VISA address of the Keithley 2420 (e.g. GPIB0::17::INSTR)'},
            {'title': 'Source Voltage (V):', 'name': 'source_voltage', 'type': 'float',
             'value': 0.0, 'min': -60.0, 'max': 60.0,
             'tip': 'Fixed voltage sourced on the collector (0 V for a pure ammeter usage)'},
            {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float',
             'value': 100e-3, 'min': 0.0, 'max': 3.0,
             'tip': 'Maximum current limit [A] (2420: 3 A max)'},
            {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float',
             'value': 20e-3, 'min': 1e-9, 'max': 3.0,
             'tip': 'Current measurement range [A]'},
            {'title': 'NPLC:', 'name': 'nplc', 'type': 'float',
             'value': 1.0, 'min': 0.01, 'max': 10.0,
             'tip': 'Number of power line cycles per measurement (precision vs speed)'},
        ]},
    ]

    def ini_attributes(self):
        self.controller: Keithley2420 = None
        self._last_current = None
        self._data_ready = False
        self._is_grabbing = False

    def commit_settings(self, param: Parameter):
        """Apply parameter changes from the interface."""
        if param.name() == 'source_voltage':
            self.controller.set_source_voltage(param.value())
        elif param.name() == 'compliance':
            self.controller.instrument.write(f':SENS:CURR:PROT {param.value()}')
        elif param.name() == 'current_range':
            self.controller.instrument.write(f':SENS:CURR:RANG {param.value()}')
        elif param.name() == 'nplc':
            self.controller.instrument.write(f':SENS:CURR:NPLC {param.value()}')

    def ini_detector(self, controller=None):
        """Initialize communication with the Keithley 2420."""
        if self.is_master:
            self.controller = Keithley2420(self.settings['keithley_settings', 'visa_address'])
            self.controller.init_mesure(
                source_voltage=self.settings['keithley_settings', 'source_voltage'],
                compliance=self.settings['keithley_settings', 'compliance'],
                current_range=self.settings['keithley_settings', 'current_range']
            )
            initialized = True
        else:
            self.controller = controller
            initialized = True

        self._data_ready = False
        self._last_current = None

        info = f"Keithley 2420 connected on {self.settings['keithley_settings', 'visa_address']}"
        return info, initialized

    def close(self):
        """Close communication with the Keithley."""
        if self.is_master:
            self.controller.close()

    def grab_data(self, Naverage=1, **kwargs):
        """Perform a single collector current measurement."""
        if self._is_grabbing:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Acquisition already in progress, ignoring.']))
            return

        self._is_grabbing = True
        self._data_ready = False

        try:
            current = self.controller.measure()

            self._last_current = current
            self._data_ready = True

            self.dte_signal.emit(DataToExport(
                name='Keithley2420',
                data=[DataFromPlugins(
                    name='Collector Current',
                    data=[np.array([current])],
                    dim='Data0D',
                    labels=['Current [A]'],
                )]
            ))
        except Exception as e:
            self._data_ready = False
            self.emit_status(ThreadCommand('Update_Status', [f'Grab error: {e}']))
        finally:
            self._is_grabbing = False

    def stop(self):
        """Stop acquisition (output stays on since 0 V default is harmless;
        turn off explicitly if needed for your RPA safety procedure)."""
        self.emit_status(ThreadCommand('Update_Status', ['Acquisition stopped']))
        return ''


if __name__ == '__main__':
    main(__file__)