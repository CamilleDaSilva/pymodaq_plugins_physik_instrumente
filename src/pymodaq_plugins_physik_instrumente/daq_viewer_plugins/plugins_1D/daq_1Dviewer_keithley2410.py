
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
            {'title': 'Vmin electron saturation (V):', 'name': 'vmin_saturation', 'type': 'float',
             'value': 15.0, 'min': 0.0,
             'tip': 'Min voltage above which the curve is considered in the electron saturation branch [V]'},
            {'title': 'Isat electron (mA):', 'name': 'isat_electron_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Calculated electron saturation current Isat [mA]'},
            {'title': 'Vf floating potential (V):', 'name': 'vf_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Floating potential Vf: intersection of ionic and transition branch regressions [V]'},
            {'title': 'I at Vf (mA):', 'name': 'i_intersection_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Current at the floating potential Vf [mA]'},
            {'title': 'Electron temperature Te (eV):', 'name': 'te_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Electron temperature from the semilog slope of the transition region [eV]'},
        ]},
    ]

    def ini_attributes(self):
        self.controller: Keithley2410 = None
        self.x_axis = None
        self._last_volts = None
        self._last_courants = None
        self.lcd_init = False

    def commit_settings(self, param: Parameter):
        """Apply parameter changes from the interface."""
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
            self.settings.child('keithley_settings', 'run_regression').setValue(False)
        elif param.name() == 'export_data' and param.value():
            self.export_csv_data()
            self.settings.child('keithley_settings', 'export_data').setValue(False)

    def ini_detector(self, controller=None):
        """Initialize communication with the Keithley 2410."""
        if self.is_master:
            self.controller = Keithley2410(self.settings['keithley_settings', 'visa_address'])
            initialized = True
        else:
            self.controller = controller
            initialized = True

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

        if not self.lcd_init:
            self.emit_status(ThreadCommand('init_lcd', dict(
                labels=['Vf (V)', 'I at Vf (mA)', 'Te (eV)'], Nvals=3, digits=4)))
            self.lcd_init = True
        self.emit_status(ThreadCommand('lcd', [
            np.array([0.0]), np.array([0.0]), np.array([0.0])]))

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

        self.emit_status(ThreadCommand('Update_Status', [f'CSV exported: {filepath}']))

    def run_langmuir_regression(self):
        """Compute ionic branch regression, electron saturation, transition branch,
        floating potential Vf (intersection of ionic and transition regressions),
        and electron temperature Te."""
        if self._last_volts is None or self._last_courants is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['No data available. Please run a Grab first.']))
            return

        S = self.settings['langmuir_settings', 'surface_sonde']
        vmax = self.settings['postproc_settings', 'vmax_regression']
        vmin_sat = self.settings['postproc_settings', 'vmin_saturation']

        # ── Ionic branch: linear regression below Vmax ───────────────────────
        MV_ion = [v for v, i in zip(self._last_volts, self._last_courants) if v < vmax]
        MI_ion = [i for v, i in zip(self._last_volts, self._last_courants) if v < vmax]

        I0_mA = None
        Constante = None
        Pente = None

        if len(MV_ion) < 2:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Not enough points for ionic regression.']))
        else:
            try:
                lr = sm.OLS(MI_ion, sm.add_constant(MV_ion)).fit()
                Constante = lr.params[0]
                Pente = lr.params[1] if len(lr.params) > 1 else 0.0
                I0_mA = Constante * 1e3
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Ionic regression error: {e}']))

        # ── Electron saturation branch: mean current above Vmin ─────────────
        MI_sat = [i for v, i in zip(self._last_volts, self._last_courants) if v > vmin_sat]
        Isat_mA = None

        if len(MI_sat) < 1:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Not enough points for electron saturation.']))
        else:
            try:
                Isat_mA = float(np.mean(MI_sat)) * 1e3
                self.settings.child('postproc_settings', 'isat_electron_result').setValue(
                    round(Isat_mA, 4))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Saturation error: {e}']))

        # ── Transition branch: linear regression between Vmax and Vmin ──────
        MV_trans = [v for v, i in zip(self._last_volts, self._last_courants)
                    if vmax <= v <= vmin_sat]
        MI_trans = [i for v, i in zip(self._last_volts, self._last_courants)
                    if vmax <= v <= vmin_sat]

        Vf = None
        Ix_mA = None
        Pente_trans = None
        Intercept_trans = None

        if len(MV_trans) < 2:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Not enough points for transition branch regression.']))
        elif I0_mA is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Ionic regression missing: cannot compute Vf.']))
        else:
            try:
                lr_trans = sm.OLS(MI_trans, sm.add_constant(MV_trans)).fit()
                Intercept_trans = lr_trans.params[0]
                Pente_trans = lr_trans.params[1] if len(lr_trans.params) > 1 else 0.0

                if Pente != Pente_trans:
                    # Vf = intersection of the two regression lines
                    Vf = (Intercept_trans - Constante) / (Pente - Pente_trans)
                    Ix = Constante + Pente * Vf
                    Ix_mA = Ix * 1e3

                    self.settings.child('postproc_settings', 'vf_result').setValue(round(Vf, 4))
                    self.settings.child('postproc_settings', 'i_intersection_result').setValue(
                        round(Ix_mA, 4))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Transition regression error: {e}']))

        # ── Electron temperature from semilog slope of transition region ─────
        MV_te = [v for v, i in zip(self._last_volts, self._last_courants)
                 if vmax < v < vmin_sat and i > 0]
        MI_te = [np.log(i) for v, i in zip(self._last_volts, self._last_courants)
                 if vmax < v < vmin_sat and i > 0]

        Te_eV = None
        if len(MV_te) >= 2:
            try:
                lr_te = sm.OLS(MI_te, sm.add_constant(MV_te)).fit()
                pente_te = lr_te.params[1]
                if pente_te > 0:
                    Te_eV = 1.0 / pente_te
                    self.settings.child('postproc_settings', 'te_result').setValue(round(Te_eV, 4))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Te error: {e}']))

        # ── Overlay on the plot ──────────────────────────────────────────────
        volts_full = np.array(self._last_volts)
        courants_full = np.array(self._last_courants)
        plot_data = [courants_full]
        plot_labels = ['Current [A]']

        if Constante is not None:
            plot_data.append(Constante + Pente * volts_full)
            plot_labels.append('Ionic regression')

        if Pente_trans is not None:
            plot_data.append(Intercept_trans + Pente_trans * volts_full)
            plot_labels.append('Transition regression')

        if Vf is not None:
            idx_x = int(np.argmin(np.abs(volts_full - Vf)))
            marker = np.full_like(volts_full, np.nan, dtype=float)
            marker[idx_x] = Ix_mA / 1e3
            plot_data.append(marker)
            plot_labels.append('Vf (intersection)')

        self.dte_signal.emit(DataToExport(
            name='Keithley2410',
            data=[DataFromPlugins(
                name='I-V Curve',
                data=plot_data,
                dim='Data1D',
                labels=plot_labels,
                axes=[self.x_axis]
            )]
        ))

        # ── Update LCD display ───────────────────────────────────────────────
        self.emit_status(ThreadCommand('lcd', [
            np.array([Vf if Vf is not None else 0.0]),
            np.array([Ix_mA if Ix_mA is not None else 0.0]),
            np.array([Te_eV if Te_eV is not None else 0.0]),
        ]))

        # ── Status message ───────────────────────────────────────────────────
        status_parts = []
        if Isat_mA is not None:
            status_parts.append(f'Isat = {Isat_mA:.3f} mA')
        if Vf is not None:
            status_parts.append(f'Vf = {Vf:.3f} V')
        if Ix_mA is not None:
            status_parts.append(f'I at Vf = {Ix_mA:.3f} mA')
        if Te_eV is not None:
            status_parts.append(f'Te = {Te_eV:.3f} eV')

        if status_parts:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Regression OK — ' + ', '.join(status_parts)]))

    def stop(self):
        """Stop acquisition and turn off output."""
        self.controller.output_off()
        self.emit_status(ThreadCommand('Update_Status', ['Acquisition stopped']))
        return ''


if __name__ == '__main__':
    main(__file__)

