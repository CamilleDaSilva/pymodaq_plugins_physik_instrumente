# -*- coding: utf-8 -*-
"""
Plugin PyMoDAQ DAQ_1DViewer pour le Keithley 2410.
Displays the I-V curve in real time with configurable parameters.
Includes Langmuir post-processing (linear regression, I0 calculation) and CSV export.
"""

import csv
import numpy as np
import statsmodels.api as sm

from qtpy.QtWidgets import QFileDialog

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_physik_instrumente.hardware.keithley2410_wrapper import Keithley2410


class DAQ_1DViewer_Keithley2410(DAQ_Viewer_base):
    """PyMoDAQ plugin to acquire and display the I-V curve of the Keithley 2410."""

    params = comon_parameters + [

        # ── Instrument connection ────────────────────────────────────────────
        {'title': 'Keithley Settings', 'name': 'keithley_settings', 'type': 'group', 'children': [
            {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str',
             'value': 'GPIB0::24::INSTR',
             'tip': 'VISA address of the Keithley (e.g. GPIB0::24::INSTR)'},
            {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float',
             'value': 700e-3, 'min': 0.0, 'max': 1.05,
             'tip': 'Maximum current limit [A]'},
            {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float',
             'value': 20e-3, 'min': 1e-9, 'max': 1.05,
             'tip': 'Current measurement range [A]'},
            {'title': 'Voltage Range (V):', 'name': 'voltage_range', 'type': 'float',
             'value': 210.0, 'min': 0.0, 'max': 210.0,
             'tip': 'Voltage source range [V] (20, 100 or 210)'},
            {'title': 'NPLC:', 'name': 'nplc', 'type': 'float',
             'value': 1.0, 'min': 0.01, 'max': 10.0,
             'tip': 'Number of power line cycles per measurement (precision vs speed)'},
             {'title': 'Export CSV', 'name': 'export_data', 'type': 'bool_push',
            'value': False, 'label': 'Export CSV',
            'tip': 'Save the current I-V curve to a CSV file'},
            {'title': 'Run regression', 'name': 'run_regression', 'type': 'bool_push',
            'value': False, 'label': 'Run regression',
            'tip': 'Run linear regression on the last acquired I-V curve'},
        ]},

        # ── Scan parameters ──────────────────────────────────────────────────
        {'title': 'Scan Settings', 'name': 'scan_settings', 'type': 'group', 'children': [
            {'title': 'Voltage Min (V):', 'name': 'volt_min', 'type': 'float',
             'value': -20.0, 'min': -210.0, 'max': 0.0,
             'tip': 'Minimum sweep voltage [V]'},
            {'title': 'Voltage Max (V):', 'name': 'volt_max', 'type': 'float',
             'value': 20.0, 'min': 0.0, 'max': 210.0,
             'tip': 'Maximum sweep voltage [V]'},
            {'title': 'N points:', 'name': 'n_points', 'type': 'int',
             'value': 50, 'min': 2, 'max': 2500,
             'tip': 'Number of measurement points'},
            {'title': 'Delay (ms):', 'name': 'delay', 'type': 'float',
             'value': 0.0, 'min': 0.0, 'max': 10000.0,
             'tip': 'Wait time between each measurement [ms]'},
        ]},

        # ── Langmuir probe physical parameters ───────────────────────────────
        {'title': 'Langmuir Probe', 'name': 'langmuir_settings', 'type': 'group', 'children': [
            {'title': 'Probe surface (cm²):', 'name': 'surface_sonde', 'type': 'float',
             'value': 6.25, 'min': 0.0,
             'tip': 'Langmuir probe surface area [cm²]'},
            {'title': 'Beam energy (eV):', 'name': 'energie_faisceau', 'type': 'float',
             'value': 400.0, 'min': 0.0,
             'tip': 'Ion beam energy [eV]'},
            {'title': 'Probe number:', 'name': 'numero_sonde', 'type': 'int',
             'value': 1, 'min': 1,
             'tip': 'Probe identifier number'},
        ]},

        # ── Post-processing ──────────────────────────────────────────────────
        {'title': 'Post-processing', 'name': 'postproc_settings', 'type': 'group', 'children': [
            {'title': 'Vmax ionic regression (V):', 'name': 'vmax_regression', 'type': 'float',
             'value': -5.0, 'max': 0.0,
             'tip': 'Max voltage for the linear regression of the ionic branch [V]'},
            {'title': 'I0 (mA):', 'name': 'i0_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Calculated ionic current I0 [mA]'},
            {'title': 'J (mA/cm²):', 'name': 'j_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Calculated ionic current density J [mA/cm²]'},
        ]},
    ]

    def ini_attributes(self):
        self.controller: Keithley2410 = None
        self.x_axis = None
        self._last_volts = None
        self._last_courants = None

    def commit_settings(self, param: Parameter):
        """Apply parameter changes from the interface."""
        print(f"commit_settings called: {param.name()}")
        if param.name() == 'compliance':
            self.controller.instrument.write(f':SENS:CURR:PROT {param.value()}')
        elif param.name() == 'current_range':
            self.controller.instrument.write(f':SENS:CURR:RANG {param.value()}')
        elif param.name() == 'nplc':
            self.controller.instrument.write(f':SENS:CURR:NPLC {param.value()}')
        elif param.name() == 'voltage_range':
            self.controller.instrument.write(f':SOUR:VOLT:RANG {param.value()}')
        elif param.name() == 'run_regression' and param.value():
            self.run_langmuir_regression()
            self.settings.child('postproc_settings', 'run_regression').setValue(False)
        elif param.name() == 'export_data' and param.value():
            self.export_csv_data()
            self.settings.child('postproc_settings', 'export_data').setValue(False)

    def ini_detector(self, controller=None):
        """Initialize communication with the Keithley 2410."""
        if self.is_master:
            self.controller = Keithley2410(self.settings['keithley_settings', 'visa_address'])
            initialized = True
        else:
            self.controller = controller
            initialized = True

        # Connect action buttons
        

        volts = self.controller.init_balayage(
            voltMin=self.settings['scan_settings', 'volt_min'],
            voltMax=self.settings['scan_settings', 'volt_max'],
            NV=self.settings['scan_settings', 'n_points'],
            compliance=self.settings['keithley_settings', 'compliance'],
            current_range=self.settings['keithley_settings', 'current_range']
        )
        self.x_axis = Axis(data=volts, label='Voltage', units='V', index=0)

        self.dte_signal_temp.emit(DataToExport(
            name='Keithley2410',
            data=[DataFromPlugins(
                name='I-V Curve',
                data=[np.zeros(len(volts))],
                dim='Data1D',
                labels=['Current [A]'],
                axes=[self.x_axis]
            )]
        ))

        info = f"Keithley 2410 connected on {self.settings['keithley_settings', 'visa_address']}"
        return info, initialized

    def close(self):
        """Close communication with the Keithley."""
        if self.is_master:
            self.controller.close()

    def grab_data(self, Naverage=1, **kwargs):
        """Perform a full voltage sweep and return the I-V curve."""
        import time
        volts = self.x_axis.get_data()
        courants = []
        delay = self.settings['scan_settings', 'delay'] / 1000.0

        for volt in volts:
            _, courant = self.controller.measure(volt)
            courants.append(courant)
            if delay > 0:
                time.sleep(delay)

        self.controller.output_off()

        self._last_volts = list(volts)
        self._last_courants = courants

        self.dte_signal.emit(DataToExport(
            name='Keithley2410',
            data=[DataFromPlugins(
                name='I-V Curve',
                data=[np.array(courants)],
                dim='Data1D',
                labels=['Current [A]'],
                axes=[self.x_axis]
            )]
        ))

    def export_csv_data(self):
        """Save the current I-V curve to a CSV file."""
        print("export_csv_data called")
        if self._last_volts is None or self._last_courants is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['No data available. Please run a Grab first.']))
            return

        filepath, _ = QFileDialog.getSaveFileName(
            None, 'Export CSV', 'IV_curve.csv', 'CSV files (*.csv)')

        if not filepath:
            return

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Voltage[V]', 'Current[A]'])
            for v, i in zip(self._last_volts, self._last_courants):
                writer.writerow([v, i])

        self.emit_status(ThreadCommand('Update_Status',
                                       [f'CSV exported: {filepath}']))
 
    def run_langmuir_regression(self):
        """Linear regression on the ionic branch of the I-V curve."""
        print("run_langmuir_regression called")
        print(f"last_volts: {self._last_volts}")  
        print(f"last_courants: {self._last_courants}")
        if self._last_volts is None or self._last_courants is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['No data available. Please run a Grab first.']))
            return

        vmax = self.settings['postproc_settings', 'vmax_regression']
        S = self.settings['langmuir_settings', 'surface_sonde']

        MV_ = []
        MI_ = []
        for v, i in zip(self._last_volts, self._last_courants):
            if v < vmax:
                MV_.append(v)
                MI_.append(i)
        print(f"vmax: {vmax}")
        print(f"MV_: {MV_}")
        print(f"MI_: {MI_}")
        if len(MV_) < 2:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Not enough points for regression. '
                                            'Adjust Vmax ionic regression.']))
            return

        try:
            MV_sm = sm.add_constant(MV_)
            lr = sm.OLS(MI_, MV_sm).fit()
            Constante = lr.params[0]
            print(f"Constante: {Constante}")
        except Exception as e:
            print(f"ERREUR regression: {e}")
            return
        I0_mA = Constante * 1e3
        J = Constante / S * 1e3
        print(f"I0_mA: {I0_mA}, J: {J}")

        try:
            self.settings.child('postproc_settings', 'i0_result').setValue(round(I0_mA, 4))
            self.settings.child('postproc_settings', 'j_result').setValue(round(J, 4))
            print("setValue OK")
        except Exception as e:
            print(f"ERREUR setValue: {e}")

        self.emit_status(ThreadCommand('Update_Status',
                                       [f'Regression OK — I0 = {I0_mA:.3f} mA, '
                                        f'J = {J:.3f} mA/cm²']))

    def stop(self):
        """Stop acquisition and turn off output."""
        self.controller.output_off()
        self.emit_status(ThreadCommand('Update_Status', ['Acquisition stopped']))
        return ''


if __name__ == '__main__':
    main(__file__)