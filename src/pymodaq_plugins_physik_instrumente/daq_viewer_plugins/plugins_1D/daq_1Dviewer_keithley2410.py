# -*- coding: utf-8 -*-
"""
PyMoDAQ DAQ_1DViewer plugin for the Keithley 2410.
Displays the I-V curve in real time with configurable parameters.
Includes Langmuir post-processing (linear regression, I0 calculation) and CSV export.
"""

import csv
import numpy as np
import statsmodels.api as sm
from datetime import datetime

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
             'value': -10.0, 'max': 0.0,
             'tip': 'Max voltage for the linear regression of the ionic branch [V]'},
            {'title': 'Vmin electron saturation (V):', 'name': 'vmin_saturation', 'type': 'float',
             'value': 15.0, 'min': 0.0,
             'tip': 'Min voltage above which the curve is considered in the electron saturation branch [V]'},
            {'title': 'Derivative threshold (fraction of peak):', 'name': 'seuil_derivee', 'type': 'float',
             'value': 0.3, 'min': 0.0, 'max': 1.0,
             'tip': 'Fraction of the max dI/dV used to cut the Gaussian-like derivative peak. '
                    'Defines the transition branch bounds automatically.'},
            {'title': 'V transition min (calc):', 'name': 'vmin_trans_calc', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Lower bound of the transition branch, computed from the derivative threshold'},
            {'title': 'V transition max (calc):', 'name': 'vmax_trans_calc', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Upper bound of the transition branch, computed from the derivative threshold'},
            {'title': 'Isat electron (mA):', 'name': 'isat_electron_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Calculated electron saturation current Isat [mA]'},
            {'title': 'Vf floating potential (V):', 'name': 'vf_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Floating potential Vf: intersection of ionic and transition branch regressions [V]'},
            {'title': 'I at Vf (mA):', 'name': 'i_intersection_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Current at the floating potential Vf [mA]'},
            {'title': 'J at Vf (mA/cm²):', 'name': 'j_intersection_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Current density J at the floating potential Vf: I(Vf) divided by the '
                    'probe surface area [mA/cm²]'},
            {'title': 'Electron temperature Te (eV):', 'name': 'te_result', 'type': 'float',
             'value': 0.0, 'readonly': True,
             'tip': 'Electron temperature from the semilog slope of the transition region [eV]'},
        ]},
    ]

    def ini_attributes(self):
        self.controller: Keithley2410 = None
        self.x_axis = None
        self._last_voltages = None
        self._last_currents = None
        self.lcd_init = False
        self._data_ready = False   # True only after a full, successful grab_data()
        self._is_grabbing = False  # prevents overlapping grabs

    def commit_settings(self, param: Parameter):
        """Apply parameter changes from the interface."""
        print(f"commit_settings called: {param.name()} = {param.value()}")
        if param.name() == 'compliance':
            self.controller.instrument.write(f':SENS:CURR:PROT {param.value()}')
        elif param.name() == 'current_range':
            self.controller.instrument.write(f':SENS:CURR:RANG {param.value()}')
        elif param.name() == 'nplc':
            self.controller.instrument.write(f':SENS:CURR:NPLC {param.value()}')
        elif param.name() == 'voltage_range':
            self.controller.instrument.write(f':SOUR:VOLT:RANG {param.value()}')
        elif param.name() == 'run_regression' and param.value():
            print("-> triggering run_langmuir_regression()")
            try:
                self.run_langmuir_regression()
            except Exception as e:
                print(f"EXCEPTION in run_langmuir_regression: {e}")
                import traceback
                traceback.print_exc()
            # 'bool_push' resets itself automatically — no manual setValue(False)
            # here, it would race with that auto-reset and cause repeated firing.
        elif param.name() == 'export_data' and param.value():
            print("-> triggering export_csv_data()")
            try:
                self.export_csv_data()
            except Exception as e:
                print(f"EXCEPTION in export_csv_data: {e}")
                import traceback
                traceback.print_exc()

    def ini_detector(self, controller=None):
        """Initialize communication with the Keithley 2410."""
        if self.is_master:
            self.controller = Keithley2410(self.settings['keithley_settings', 'visa_address'])
            initialized = True
        else:
            self.controller = controller
            initialized = True

        # NOTE: intentionally no manual sigValueChanged.connect() here.
        # commit_settings() above is the single, correct path for
        # run_regression / export_data — connecting both was causing
        # double/cascading calls in an earlier version.

        voltages = self.controller.init_balayage(
            voltMin=self.settings['scan_settings', 'volt_min'],
            voltMax=self.settings['scan_settings', 'volt_max'],
            NV=self.settings['scan_settings', 'n_points'],
            compliance=self.settings['keithley_settings', 'compliance'],
            current_range=self.settings['keithley_settings', 'current_range']
        )
        self.x_axis = Axis(data=voltages, label='Voltage', units='V', index=0)

        self._data_ready = False
        self._last_voltages = None
        self._last_currents = None

        self.dte_signal_temp.emit(DataToExport(
            name='Keithley2410',
            data=[DataFromPlugins(
                name='I-V Curve',
                data=[np.zeros(len(voltages))],
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
        """Perform a full voltage sweep and return the I-V curve.
        A single complete acquisition per call, whether triggered by Snap (1)
        or by the Continuous Grab (the repeated calling is decided by
        PyMoDAQ itself, not by this method)."""
        import time

        if self._is_grabbing:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Acquisition already in progress, ignoring.']))
            return

        self._is_grabbing = True
        self._data_ready = False

        try:
            voltages = self.x_axis.get_data()
            currents = []
            delay = self.settings['scan_settings', 'delay'] / 1000.0

            for voltage in voltages:
                _, current = self.controller.measure(voltage)
                currents.append(current)
                if delay > 0:
                    time.sleep(delay)

            self.controller.output_off()

            self._last_voltages = list(voltages)
            self._last_currents = currents
            self._data_ready = True
            print(f"grab_data done, _data_ready={self._data_ready}, "
                  f"len={len(self._last_voltages)}")

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
            self.emit_status(ThreadCommand('Update_Status',
                                           [f'Acquisition done: {len(voltages)} points.']))
        except Exception as e:
            self._data_ready = False
            print(f"EXCEPTION in grab_data: {e}")
            import traceback
            traceback.print_exc()
            self.emit_status(ThreadCommand('Update_Status', [f'Grab error: {e}']))
        finally:
            self._is_grabbing = False

    def export_csv_data(self):
        """Save the current I-V curve to a CSV file, chosen by the user,
        with all computed Langmuir parameters written as a header block
        above the voltage/current data table (single file, as requested)."""
        print("export_csv_data called")
        if not self._data_ready or self._last_voltages is None or self._last_currents is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['No completed acquisition available. Snap (1) first.']))
            return

        now = datetime.now()
        energy = self.settings['langmuir_settings', 'energie_faisceau']
        default_name = (f'langmuir_{energy:g}eV_'
                         f'{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv')

        filepath, _ = QFileDialog.getSaveFileName(
            None, 'Export CSV', default_name, 'CSV files (*.csv)')

        if not filepath:
            print("export_csv_data: cancelled by user (no path chosen)")
            return

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')

                writer.writerow(['# Langmuir probe acquisition'])
                writer.writerow(['# Date', now.strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['# Beam energy [eV]', energy])
                writer.writerow(['# Probe surface [cm2]',
                                  self.settings['langmuir_settings', 'surface_sonde']])
                writer.writerow(['# Probe number',
                                  self.settings['langmuir_settings', 'numero_sonde']])
                writer.writerow(['# Vmax ionic regression [V]',
                                  self.settings['postproc_settings', 'vmax_regression']])
                writer.writerow(['# Vmin electron saturation [V]',
                                  self.settings['postproc_settings', 'vmin_saturation']])
                writer.writerow(['# V transition min (calc) [V]',
                                  self.settings['postproc_settings', 'vmin_trans_calc']])
                writer.writerow(['# V transition max (calc) [V]',
                                  self.settings['postproc_settings', 'vmax_trans_calc']])
                writer.writerow(['# Isat electron [mA]',
                                  self.settings['postproc_settings', 'isat_electron_result']])
                writer.writerow(['# Vf floating potential [V]',
                                  self.settings['postproc_settings', 'vf_result']])
                writer.writerow(['# I at Vf [mA]',
                                  self.settings['postproc_settings', 'i_intersection_result']])
                writer.writerow(['# J at Vf [mA/cm2]',
                                  self.settings['postproc_settings', 'j_intersection_result']])
                writer.writerow(['# Electron temperature Te [eV]',
                                  self.settings['postproc_settings', 'te_result']])
                writer.writerow([])

                writer.writerow(['Voltage[V]', 'Current[A]'])
                for voltage, current in zip(self._last_voltages, self._last_currents):
                    writer.writerow([voltage, current])

            print(f"export_csv_data: file written to {filepath}")
            self.emit_status(ThreadCommand('Update_Status', [f'CSV exported: {filepath}']))
        except Exception as e:
            print(f"EXCEPTION in export_csv_data (write): {e}")
            self.emit_status(ThreadCommand('Update_Status', [f'Export error: {e}']))

    def compute_transition_bounds(self, voltages, currents, threshold_frac):
        """Determine the transition branch bounds by thresholding the peak of
        dI/dV (expected to look Gaussian-like around the transition region
        between the ionic and electron-saturation branches).

        Returns
        -------
        (v_min, v_max) : voltage bounds, or (None, None) if no clear peak is
            detected (e.g. a purely resistive load -> flat dI/dV).
        """
        voltages = np.asarray(voltages, dtype=float)
        currents = np.asarray(currents, dtype=float)

        if len(voltages) < 3:
            return None, None

        order = np.argsort(voltages)
        voltages_sorted = voltages[order]
        currents_sorted = currents[order]

        didv = np.gradient(currents_sorted, voltages_sorted)
        idx_peak = int(np.argmax(didv))
        peak_val = didv[idx_peak]

        if peak_val <= 0:
            return None, None

        threshold_val = threshold_frac * peak_val

        idx_left = idx_peak
        while idx_left > 0 and didv[idx_left - 1] >= threshold_val:
            idx_left -= 1
        idx_right = idx_peak
        while idx_right < len(didv) - 1 and didv[idx_right + 1] >= threshold_val:
            idx_right += 1

        v_min = float(voltages_sorted[idx_left])
        v_max = float(voltages_sorted[idx_right])
        return v_min, v_max

    def run_langmuir_regression(self):
        """Compute ionic branch regression, electron saturation, transition branch,
        floating potential Vf (intersection of ionic and transition regressions),
        current density J at Vf, and electron temperature Te."""
        print("run_langmuir_regression called")
        print(f"_data_ready: {self._data_ready}")
        if not self._data_ready or self._last_voltages is None or self._last_currents is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['No completed acquisition available. Snap (1) first.']))
            return

        vmax = self.settings['postproc_settings', 'vmax_regression']
        vmin_sat = self.settings['postproc_settings', 'vmin_saturation']
        deriv_threshold = self.settings['postproc_settings', 'seuil_derivee']

        # ── Ionic branch: linear regression below Vmax ───────────────────────
        voltages_ionic = [v for v, i in zip(self._last_voltages, self._last_currents) if v < vmax]
        currents_ionic = [i for v, i in zip(self._last_voltages, self._last_currents) if v < vmax]

        intercept_ionic = None
        slope_ionic = None

        if len(voltages_ionic) < 2:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Not enough points for ionic regression.']))
        else:
            try:
                fit_ionic = sm.OLS(currents_ionic, sm.add_constant(voltages_ionic)).fit()
                intercept_ionic = fit_ionic.params[0]
                slope_ionic = fit_ionic.params[1] if len(fit_ionic.params) > 1 else 0.0
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Ionic regression error: {e}']))

        # ── Electron saturation branch: mean current above Vmin ─────────────
        currents_saturation = [i for v, i in zip(self._last_voltages, self._last_currents) if v > vmin_sat]
        isat_ma = None

        if len(currents_saturation) < 1:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Not enough points for electron saturation.']))
        else:
            try:
                isat_ma = float(np.mean(currents_saturation)) * 1e3
                self.settings.child('postproc_settings', 'isat_electron_result').setValue(
                    round(isat_ma, 4))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Saturation error: {e}']))

        # ── Transition branch bounds: threshold on the dI/dV peak ────────────
        v_trans_min, v_trans_max = self.compute_transition_bounds(
            self._last_voltages, self._last_currents, deriv_threshold)
        print(f"v_trans_min={v_trans_min}, v_trans_max={v_trans_max}")

        if v_trans_min is None or v_trans_max is None:
            self.emit_status(ThreadCommand('Update_Status',
                ['No clear dI/dV peak detected (flat or decreasing derivative) — '
                 'transition branch bounds cannot be determined. Expected e.g. '
                 'with a purely resistive load.']))
            self.settings.child('postproc_settings', 'vmin_trans_calc').setValue(0.0)
            self.settings.child('postproc_settings', 'vmax_trans_calc').setValue(0.0)
            voltages_transition, currents_transition = [], []
        else:
            self.settings.child('postproc_settings', 'vmin_trans_calc').setValue(round(v_trans_min, 4))
            self.settings.child('postproc_settings', 'vmax_trans_calc').setValue(round(v_trans_max, 4))
            voltages_transition = [v for v, i in zip(self._last_voltages, self._last_currents)
                                    if v_trans_min <= v <= v_trans_max]
            currents_transition = [i for v, i in zip(self._last_voltages, self._last_currents)
                                    if v_trans_min <= v <= v_trans_max]

        vf = None
        current_at_vf_ma = None
        j_at_vf_ma_cm2 = None
        slope_transition = None
        intercept_transition = None

        if len(voltages_transition) < 2:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Not enough points for transition branch regression.']))
        elif intercept_ionic is None:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Ionic regression missing: cannot compute Vf.']))
        else:
            try:
                fit_transition = sm.OLS(currents_transition, sm.add_constant(voltages_transition)).fit()
                intercept_transition = fit_transition.params[0]
                slope_transition = fit_transition.params[1] if len(fit_transition.params) > 1 else 0.0

                slope_diff = slope_ionic - slope_transition
                slope_scale = max(abs(slope_ionic), abs(slope_transition), 1e-12)

                if abs(slope_diff) > 1e-9 * slope_scale:
                    vf = (intercept_transition - intercept_ionic) / slope_diff
                    current_at_vf = intercept_ionic + slope_ionic * vf
                    current_at_vf_ma = current_at_vf * 1e3

                    self.settings.child('postproc_settings', 'vf_result').setValue(round(vf, 4))
                    self.settings.child('postproc_settings', 'i_intersection_result').setValue(
                        round(current_at_vf_ma, 4))

                    # ── Current density J at Vf = I(Vf) / probe surface ──────
                    surface_sonde = self.settings['langmuir_settings', 'surface_sonde']
                    if surface_sonde > 0:
                        j_at_vf_ma_cm2 = current_at_vf_ma / surface_sonde
                        self.settings.child('postproc_settings', 'j_intersection_result').setValue(
                            round(j_at_vf_ma_cm2, 4))
                    else:
                        self.emit_status(ThreadCommand('Update_Status',
                            ['Probe surface is zero: J at Vf cannot be computed.']))
                else:
                    self.emit_status(ThreadCommand('Update_Status',
                        ['Ionic and transition slopes are equal (e.g. resistive load): '
                         'Vf cannot be computed.']))
            except Exception as e:
                self.emit_status(ThreadCommand('Update_Status', [f'Transition regression error: {e}']))

        # ── Electron temperature from semilog slope of transition region ─────
        te_ev = None
        if v_trans_min is not None and v_trans_max is not None:
            voltages_te = [v for v, i in zip(self._last_voltages, self._last_currents)
                           if v_trans_min < v < v_trans_max and i > 0]
            log_currents_te = [np.log(i) for v, i in zip(self._last_voltages, self._last_currents)
                               if v_trans_min < v < v_trans_max and i > 0]

            if len(voltages_te) >= 2:
                try:
                    fit_te = sm.OLS(log_currents_te, sm.add_constant(voltages_te)).fit()
                    slope_te = fit_te.params[1]
                    if slope_te > 0:
                        te_ev = 1.0 / slope_te
                        self.settings.child('postproc_settings', 'te_result').setValue(round(te_ev, 4))
                except Exception as e:
                    self.emit_status(ThreadCommand('Update_Status', [f'Te error: {e}']))

        # ── Overlay on the plot ──────────────────────────────────────────────
        voltages_full = np.array(self._last_voltages)
        currents_full = np.array(self._last_currents)
        plot_data = [currents_full]
        plot_labels = ['Current [A]']

        if intercept_ionic is not None:
            plot_data.append(intercept_ionic + slope_ionic * voltages_full)
            plot_labels.append('Ionic regression')

        if slope_transition is not None:
            plot_data.append(intercept_transition + slope_transition * voltages_full)
            plot_labels.append('Transition regression')

        if vf is not None:
            idx_x = int(np.argmin(np.abs(voltages_full - vf)))
            marker = np.full_like(voltages_full, np.nan, dtype=float)
            marker[idx_x] = current_at_vf_ma / 1e3
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
        print("run_langmuir_regression: plot emitted")

        # ── Update LCD display ───────────────────────────────────────────────
        self.emit_status(ThreadCommand('lcd', [
            np.array([vf if vf is not None else 0.0]),
            np.array([current_at_vf_ma if current_at_vf_ma is not None else 0.0]),
            np.array([te_ev if te_ev is not None else 0.0]),
        ]))

        # ── Status message ───────────────────────────────────────────────────
        status_parts = []
        if isat_ma is not None:
            status_parts.append(f'Isat = {isat_ma:.3f} mA')
        if vf is not None:
            status_parts.append(f'Vf = {vf:.3f} V')
        if current_at_vf_ma is not None:
            status_parts.append(f'I at Vf = {current_at_vf_ma:.3f} mA')
        if j_at_vf_ma_cm2 is not None:
            status_parts.append(f'J at Vf = {j_at_vf_ma_cm2:.3f} mA/cm²')
        if te_ev is not None:
            status_parts.append(f'Te = {te_ev:.3f} eV')

        print(f"status_parts: {status_parts}")
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