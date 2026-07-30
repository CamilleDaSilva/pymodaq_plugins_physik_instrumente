# -*- coding: utf-8 -*-
"""
PyMoDAQ DAQ_1DViewer plugin for the Keithley 2410.
Acquires and displays the I-V curve with Langmuir post-processing.
"""

import csv
import numpy as np
import statsmodels.api as sm
from qtpy.QtWidgets import QFileDialog
from datetime import datetime

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
             'value': 700e-3, 'min': 0.0, 'max': 1.05},
            {'title': 'Current Range (A) (0=auto):', 'name': 'current_range', 'type': 'float',
             'value': 0.0, 'min': 0.0, 'max': 1.05,
             'tip': 'Current measurement range [A]. Set to 0 for autorange.'},
            {'title': 'Voltage Range (V):', 'name': 'voltage_range', 'type': 'float',
             'value': 210.0, 'min': 0.0, 'max': 210.0},
            {'title': 'NPLC:', 'name': 'nplc', 'type': 'float',
             'value': 1.0, 'min': 0.01, 'max': 10.0},
            {'title': 'Export CSV', 'name': 'export_data', 'type': 'bool_push',
             'value': False, 'label': 'Export CSV'},
            {'title': 'Run regression', 'name': 'run_regression', 'type': 'bool_push',
             'value': False, 'label': 'Run regression'},
        ]},

        # ── Scan parameters ──────────────────────────────────────────────────
        {'title': 'Scan Settings', 'name': 'scan_settings', 'type': 'group', 'children': [
            {'title': 'Voltage Min (V):', 'name': 'volt_min', 'type': 'float',
             'value': -20.0, 'min': -210.0, 'max': 0.0},
            {'title': 'Voltage Max (V):', 'name': 'volt_max', 'type': 'float',
             'value': 20.0, 'min': 0.0, 'max': 210.0},
            {'title': 'N points:', 'name': 'n_points', 'type': 'int',
             'value': 50, 'min': 2, 'max': 2500},
            {'title': 'Delay (ms):', 'name': 'delay', 'type': 'float',
             'value': 0.0, 'min': 0.0, 'max': 10000.0},
        ]},

        # ── Langmuir probe ───────────────────────────────────────────────────
        {'title': 'Langmuir Probe', 'name': 'langmuir_settings', 'type': 'group', 'children': [
            {'title': 'Probe surface (cm²):', 'name': 'surface_sonde', 'type': 'float',
             'value': 6.25, 'min': 0.0},
            {'title': 'Beam energy (eV):', 'name': 'energie_faisceau', 'type': 'float',
             'value': 400.0, 'min': 0.0},
            {'title': 'Probe number:', 'name': 'numero_sonde', 'type': 'int',
             'value': 1, 'min': 1},
        ]},

        # ── Post-processing ──────────────────────────────────────────────────
        {'title': 'Post-processing', 'name': 'postproc_settings', 'type': 'group', 'children': [
            {'title': 'Auto-detect bounds:', 'name': 'auto_detect', 'type': 'bool',
             'value': True,
             'tip': 'Automatically detect transition region bounds from dI/dV curve'},
            {'title': 'dI/dV threshold (A/V):', 'name': 'auto_detect_threshold', 'type': 'float',
             'value': 0.0001, 'min': 0.0, 'max': 1.0,
             'tip': 'dI/dV value where the Gaussian crosses the threshold on each side of the peak'},
            {'title': 'Vmax ionic regression (V):', 'name': 'vmax_regression', 'type': 'float',
             'value': -5.0, 'max': 0.0,
             'tip': 'Used only if Auto-detect is OFF'},
            {'title': 'Vmin electron saturation (V):', 'name': 'vmin_saturation', 'type': 'float',
             'value': 10.0, 'min': 0.0,
             'tip': 'Used only if Auto-detect is OFF'},
            {'title': 'Isat electron (mA):', 'name': 'isat_electron_result', 'type': 'float',
             'value': 0.0, 'readonly': True},
            {'title': 'Vf floating potential (V):', 'name': 'vf_result', 'type': 'float',
             'value': 0.0, 'readonly': True},
            {'title': 'I at Vf (mA):', 'name': 'i_intersection_result', 'type': 'float',
             'value': 0.0, 'readonly': True},
            {'title': 'Electron temperature Te (eV):', 'name': 'te_result', 'type': 'float',
             'value': 0.0, 'readonly': True},
        ]},
    ]

    def ini_attributes(self):
        self.controller: Keithley2410 = None
        self.x_axis = None
        self._last_volts = None
        self._last_currents = None
        self.lcd_init = False
        self._grabbing = False

    def commit_settings(self, param: Parameter):
        """Apply parameter changes from the interface."""
        if param.name() == 'compliance':
            self.controller.instrument.write(f':SENS:CURR:PROT {param.value()}')
        elif param.name() == 'current_range':
            if param.value() == 0.0:
                self.controller.instrument.write(':SENS:CURR:RANG:AUTO ON')
            else:
                self.controller.instrument.write(f':SENS:CURR:RANG {param.value()}')
        elif param.name() == 'nplc':
            self.controller.instrument.write(f':SENS:CURR:NPLC {param.value()}')
        elif param.name() == 'voltage_range':
            self.controller.instrument.write(f':SOUR:VOLT:RANG {param.value()}')

    def ini_detector(self, controller=None):
        """Initialize communication with the Keithley 2410."""
        if self.is_master:
            self.controller = Keithley2410(self.settings['keithley_settings', 'visa_address'])
            initialized = True
        else:
            self.controller = controller
            initialized = True

        # Connect buttons via sigValueChanged (runs in GUI thread → QFileDialog works)
        self.settings.child('keithley_settings', 'run_regression').sigValueChanged.connect(
            lambda param: self.run_langmuir_regression())
        self.settings.child('keithley_settings', 'export_data').sigValueChanged.connect(
            lambda param: self.export_csv_data())

        cr = self.settings['keithley_settings', 'current_range']
        self.controller.init_balayage(
            voltMin=self.settings['scan_settings', 'volt_min'],
            voltMax=self.settings['scan_settings', 'volt_max'],
            NV=self.settings['scan_settings', 'n_points'],
            compliance=self.settings['keithley_settings', 'compliance'],
            current_range=None if cr == 0.0 else cr
        )

        volts = np.linspace(
            self.settings['scan_settings', 'volt_min'],
            self.settings['scan_settings', 'volt_max'],
            self.settings['scan_settings', 'n_points']
        )
        self.x_axis = Axis(data=volts, label='Voltage', units='V', index=0)

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
        if self._grabbing:
            return
        self._grabbing = True

        import time
        volts = np.linspace(
            self.settings['scan_settings', 'volt_min'],
            self.settings['scan_settings', 'volt_max'],
            self.settings['scan_settings', 'n_points']
        )
        self.x_axis = Axis(data=volts, label='Voltage', units='V', index=0)
        currents = []
        delay = self.settings['scan_settings', 'delay'] / 1000.0

        for volt in volts:
            _, current = self.controller.measure(volt, stabilization_delay=delay if delay > 0 else 0.05)
            currents.append(current)

        self.controller.output_off()
        self._last_volts = list(volts)
        self._last_currents = currents

        self.dte_signal.emit(DataToExport(
            name='Keithley2410',
            data=[DataFromPlugins(
                name='I-V Curve',
                data=[np.array(currents)],
                dim='Data1D',
                labels=['Current [A]'],
                axes=[self.x_axis]
            )]
        ))

        self._grabbing = False
        self.emit_status(ThreadCommand('stop', []))

    def export_csv_data(self):
        """Save the current I-V curve to a CSV file (user chooses location)."""
        if self._last_volts is None or self._last_currents is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['No data available. Please run a Grab first.']))
            return

        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath, _ = QFileDialog.getSaveFileName(
            None, 'Export CSV', f'IV_curve_{date}.csv', 'CSV files (*.csv)')

        if not filepath:
            return

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['Voltage[V]', 'Current[A]'])
                for v, i in zip(self._last_volts, self._last_currents):
                    writer.writerow([v, i])
            self.emit_status(ThreadCommand('Update_Status', [f'CSV exported: {filepath}']))
        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [f'Export error: {e}']))

    def _auto_detect_bounds(self, volts_arr, currents_arr):
        """Detect Vmax and Vmin_sat from the dI/dV Gaussian curve."""
        seuil = self.settings['postproc_settings', 'auto_detect_threshold']
        dI = np.gradient(currents_arr, volts_arr)
        idx_peak = int(np.argmax(dI))

        idx_left = 0
        for k in range(idx_peak, 0, -1):
            if dI[k] < seuil:
                idx_left = k
                break
        vmax = float(volts_arr[idx_left])

        idx_right = len(dI) - 1
        for k in range(idx_peak, len(dI)):
            if dI[k] < seuil:
                idx_right = k
                break
        vmin_sat = float(volts_arr[idx_right])

        return vmax, vmin_sat

    def run_langmuir_regression(self):
        """Compute ionic regression, electron saturation, transition regression,
        floating potential Vf and electron temperature Te."""
        if self._last_volts is None or self._last_currents is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['No data available. Please run a Grab first.']))
            return

        S = self.settings['langmuir_settings', 'surface_sonde']
        volts_arr = np.array(self._last_volts)
        currents_arr = np.array(self._last_currents)

        # ── Bounds detection ─────────────────────────────────────────────────
        if self.settings['postproc_settings', 'auto_detect']:
            vmax, vmin_sat = self._auto_detect_bounds(volts_arr, currents_arr)
            self.emit_status(ThreadCommand('Update_Status',
                                           [f'Auto-detect: Vmax={vmax:.2f}V, '
                                            f'Vmin_sat={vmin_sat:.2f}V']))
        else:
            vmax = self.settings['postproc_settings', 'vmax_regression']
            vmin_sat = self.settings['postproc_settings', 'vmin_saturation']

        # ── Ionic branch regression ──────────────────────────────────────────
        MV_ion = [v for v, i in zip(self._last_volts, self._last_currents) if v < vmax]
        MI_ion = [i for v, i in zip(self._last_volts, self._last_currents) if v < vmax]

        Constante = None
        Pente = None

        if len(MV_ion) >= 2:
            try:
                lr = sm.OLS(MI_ion, sm.add_constant(MV_ion)).fit()
                Constante = lr.params[0]
                Pente = lr.params[1] if len(lr.params) > 1 else 0.0
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status',
                                               [f'Ionic regression error: {e}']))

        # ── Electron saturation: mean current above Vmin ─────────────────────
        MI_sat = [i for v, i in zip(self._last_volts, self._last_currents) if v > vmin_sat]
        Isat_electron_mA = None

        if len(MI_sat) >= 1:
            try:
                Isat_electron_mA = float(np.mean(MI_sat)) * 1e3
                self.settings.child('postproc_settings', 'isat_electron_result').setValue(
                    round(Isat_electron_mA, 4))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status',
                                               [f'Saturation error: {e}']))

        # ── Transition branch regression ─────────────────────────────────────
        MV_trans = [v for v, i in zip(self._last_volts, self._last_currents)
                    if vmax <= v <= vmin_sat]
        MI_trans = [i for v, i in zip(self._last_volts, self._last_currents)
                    if vmax <= v <= vmin_sat]

        Vf = None
        Ix_mA = None
        Pente_trans = None
        Intercept_trans = None

        if len(MV_trans) >= 2 and Constante is not None:
            try:
                lr_trans = sm.OLS(MI_trans, sm.add_constant(MV_trans)).fit()
                Intercept_trans = lr_trans.params[0]
                Pente_trans = lr_trans.params[1] if len(lr_trans.params) > 1 else 0.0

                if abs(Pente - Pente_trans) > 1e-12:
                    Vf = (Intercept_trans - Constante) / (Pente - Pente_trans)
                    Ix = Constante + Pente * Vf
                    Ix_mA = Ix * 1e3
                    self.settings.child('postproc_settings', 'vf_result').setValue(round(Vf, 4))
                    self.settings.child('postproc_settings', 'i_intersection_result').setValue(
                        round(Ix_mA, 4))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status',
                                               [f'Transition regression error: {e}']))

        # ── Electron temperature ─────────────────────────────────────────────
        MV_te = [v for v, i in zip(self._last_volts, self._last_currents)
                 if vmax < v < vmin_sat and i > 0]
        MI_te = [np.log(i) for v, i in zip(self._last_volts, self._last_currents)
                 if vmax < v < vmin_sat and i > 0]

        Te_eV = None
        if len(MV_te) >= 2:
            try:
                lr_te = sm.OLS(MI_te, sm.add_constant(MV_te)).fit()
                pente_te = lr_te.params[1] if len(lr_te.params) > 1 else 0.0
                if pente_te > 0:
                    Te_eV = 1.0 / pente_te
                    self.settings.child('postproc_settings', 'te_result').setValue(
                        round(Te_eV, 4))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Te error: {e}']))

        # ── Overlay on the plot ──────────────────────────────────────────────
        plot_data = [currents_arr]
        plot_labels = ['Current [A]']

        if Constante is not None:
            plot_data.append(Constante + Pente * volts_arr)
            plot_labels.append('Ionic regression')

        if Pente_trans is not None:
            plot_data.append(Intercept_trans + Pente_trans * volts_arr)
            plot_labels.append('Transition regression')

        if Vf is not None:
            idx_x = int(np.argmin(np.abs(volts_arr - Vf)))
            marker = np.full_like(volts_arr, np.nan, dtype=float)
            marker[idx_x] = Ix_mA / 1e3 if Ix_mA is not None else 0.0
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

        # ── Update LCD ───────────────────────────────────────────────────────
        self.emit_status(ThreadCommand('lcd', [
            np.array([Vf if Vf is not None else 0.0]),
            np.array([Ix_mA if Ix_mA is not None else 0.0]),
            np.array([Te_eV if Te_eV is not None else 0.0]),
        ]))

        # ── Status message ───────────────────────────────────────────────────
        status_parts = []
        if Vf is not None:
            status_parts.append(f'Vf={Vf:.3f}V')
        if Ix_mA is not None:
            status_parts.append(f'I@Vf={Ix_mA:.3f}mA')
        if Te_eV is not None:
            status_parts.append(f'Te={Te_eV:.3f}eV')
        if status_parts:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Regression OK — ' + ', '.join(status_parts)]))

    def stop(self):
        """Stop acquisition and turn off output."""
        self.controller.output_off()
        self._grabbing = False
        self.emit_status(ThreadCommand('Update_Status', ['Acquisition stopped']))
        return ''


if __name__ == '__main__':
    main(__file__)