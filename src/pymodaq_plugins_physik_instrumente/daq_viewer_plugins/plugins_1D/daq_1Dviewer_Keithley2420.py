# -*- coding: utf-8 -*-
"""
PyMoDAQ DAQ_1DViewer plugin for the RPA (Retarding Potential Analyzer).
Sweeps the grid voltage on the Keithley 2410 and reads the collector
current on the Keithley 2420 at each step, directly building the I-V
curve — no PyMoDAQ DAQ_Scan involved (mirrors the Langmuir 1D viewer
approach).

Post-processing:
    - dI/dV (Savitzky-Golay derivative), computed on demand via a
      dedicated button (no automatic recompute on every acquisition,
      so settings can be tuned without re-running a sweep)
    - mean / std of the ion energy distribution
    - Gaussian fit superimposed on dI/dV (can be toggled independently)
    - Averaging of several full sweeps (n_sweeps), in addition to the
      existing point-by-point averaging (n_average)
    - CSV export of the last acquisition to a user-chosen path

Author: Camille Da Silva
ONERA DPHY/CSE — PICOMAX-E, 2026
"""

import csv
import time
import numpy as np

from qtpy.QtWidgets import QFileDialog

from pymodaq_utils.utils import ThreadCommand
from pymodaq_data.data import DataToExport, Axis
from pymodaq_gui.parameter import Parameter

from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.data import DataFromPlugins

from pymodaq_plugins_physik_instrumente.hardware.keithley2410_wrapper import Keithley2410
from pymodaq_plugins_physik_instrumente.hardware.keithley2420_wrapper import Keithley2420
from pymodaq_plugins_physik_instrumente.hardware.rpa_postprocessing import (
    derivative, weighted_mean_std, fit_gaussian
)


class DAQ_1DViewer_Keithley2420(DAQ_Viewer_base):
    """PyMoDAQ plugin acquiring the RPA I-V curve: grid voltage (2410) vs
    collector current (2420), plus post-processing (dI/dV, Gaussian fit)."""

    params = comon_parameters + [

        {'title': 'Keithley 2410 (Grid)', 'name': 'k2410_settings', 'type': 'group', 'children': [
            {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str',
             'value': 'GPIB0::24::INSTR'},
            {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float',
             'value': 0.02, 'min': 0.0, 'max': 1.05,
             'tip': 'Le 2410 est limite a 22W max. A 700V, le courant max '
                    'utilisable est ~21-31 mA -> ne pas depasser ~0.03 A '
                    'en haute tension.'},
            {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float',
             'value': 20e-3, 'min': 0.0, 'max': 1.05,
             'tip': 'Range of the 2410 (if its own current is also read). ~20 mA for a resistor test'},
            {'title': 'Voltage Range (V):', 'name': 'voltage_range', 'type': 'float',
             'value': 1200.0, 'min': 0.0, 'max': 1200.0,
             'tip': 'Must cover |Vmin| + |Vmax|'},
        ]},

        {'title': 'Keithley 2420 (Collector)', 'name': 'k2420_settings', 'type': 'group', 'children': [
            {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str',
             'value': 'GPIB1::25::INSTR',
             'tip': 'VISA address of the Keithley 2420 (collector current)'},
            {'title': 'Source Voltage (V):', 'name': 'source_voltage', 'type': 'float',
             'value': 0.0, 'min': -60.0, 'max': 60.0,
             'tip': 'Voltage imposed on the collector (0 V for pure ammeter use)'},
            {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float',
             'value': 100e-3, 'min': 0.0, 'max': 3.0},
            {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float',
             'value': 20e-3, 'min': 1e-9, 'max': 3.0,
             'tip': 'Measurement range — size it above the max expected current '
                    '(for 1.18 kOhm @ +/-20 V -> ~17 mA -> 20 mA is a good fit)'},
            {'title': 'NPLC:', 'name': 'nplc', 'type': 'float',
             'value': 1.0, 'min': 0.01, 'max': 10.0},
        ]},

        {'title': 'Scan Settings', 'name': 'scan_settings', 'type': 'group', 'children': [
            {'title': 'Voltage Min (V):', 'name': 'volt_min', 'type': 'float',
             'value': 500.0, 'min': -1100.0, 'max': 1100.0},
            {'title': 'Voltage Max (V):', 'name': 'volt_max', 'type': 'float',
             'value': 700.0, 'min': -1100.0, 'max': 1100.0},
            {'title': 'N points:', 'name': 'n_points', 'type': 'int',
             'value': 41, 'min': 2, 'max': 2500},
            {'title': 'N Average:', 'name': 'n_average', 'type': 'int',
             'value': 5, 'min': 1, 'max': 100,
             'tip': 'Number of samples averaged per voltage point (in-place temporal '
                    'averaging, at each step of the sweep).'},
            {'title': 'N Sweeps:', 'name': 'n_sweeps', 'type': 'int',
             'value': 1, 'min': 1, 'max': 100,
             'tip': 'Number of FULL sweeps to repeat and average. Complementary to '
                    'N Average: here the whole voltage sweep is repeated several '
                    'times and the resulting I-V curves are averaged together.'},
            {'title': 'Delay (ms):', 'name': 'delay', 'type': 'float',
             'value': 100.0, 'min': 0.0, 'max': 10000.0,
             'tip': 'Settling time after applying the grid voltage'},
        ]},

        {'title': 'Post-processing', 'name': 'postproc_settings', 'type': 'group', 'children': [
            {'title': 'Enable dI/dV:', 'name': 'enable_derivative', 'type': 'bool',
             'value': True,
             'tip': "Taken into account on the next click on 'Recompute post-processing', "
                    "no automatic recompute."},
            {'title': 'Smoothing window (pts, odd):', 'name': 'smooth_window', 'type': 'int',
             'value': 9, 'min': 5, 'max': 501,
             'tip': 'Width of the Savitzky-Golay window used for the derivative. '
                    'Increase if dI/dV is too noisy; keep it << N points. '
                    'Auto-adjusted (odd, > polynomial order) if needed.'},
            {'title': 'Smoothing polynomial order:', 'name': 'smooth_polyorder', 'type': 'int',
             'value': 2, 'min': 1, 'max': 5,
             'tip': 'Must stay < smoothing window (auto-adjusted otherwise).'},
            {'title': 'Enable Gaussian fit:', 'name': 'enable_fit', 'type': 'bool',
             'value': True,
             'tip': 'Fits a Gaussian on dI/dV (useless/unstable on a pure line, e.g. '
                    'a resistor test — uncheck in that case).'},
            {'title': 'Invert sign (-dI/dV):', 'name': 'invert_sign', 'type': 'bool',
             'value': False,
             'tip': 'The Keithley measures in receiver convention (negative current, '
                    'rising back to 0 as the grid repels ions). Check this to display '
                    '-dI/dV and get an energy distribution with a positive peak '
                    '(standard physical convention) instead of raw dI/dV. Does not '
                    'affect the mean/std (computed on |dI/dV|, sign-independent).'},
            {'title': 'Recompute post-processing:', 'name': 'recompute_postproc', 'type': 'bool',
             'value': False,
             'tip': 'Check to recompute dI/dV + fit from the LAST acquisition, without '
                    're-running a sweep — use this after changing the settings above. '
                    'Unchecks itself automatically after the computation.'},
            {'title': 'Weighted mean [V]:', 'name': 'stat_mean', 'type': 'float',
             'value': float('nan'), 'readonly': True,
             'tip': 'Mean of the energy distribution, weighted by |dI/dV|.'},
            {'title': 'Weighted sigma [V]:', 'name': 'stat_std', 'type': 'float',
             'value': float('nan'), 'readonly': True,
             'tip': 'Standard deviation of the energy distribution, weighted by |dI/dV|.'},
            {'title': 'Gaussian fit mu [V]:', 'name': 'stat_mean_fit', 'type': 'float',
             'value': float('nan'), 'readonly': True,
             'tip': 'Peak position from the Gaussian fit.'},
            {'title': 'Gaussian fit sigma [V]:', 'name': 'stat_sigma_fit', 'type': 'float',
             'value': float('nan'), 'readonly': True,
             'tip': 'Peak width from the Gaussian fit.'},
        ]},

        {'title': 'Export', 'name': 'export_settings', 'type': 'group', 'children': [
            {'title': 'Export CSV...:', 'name': 'export_csv', 'type': 'bool',
             'value': False,
             'tip': 'Check to open a save-file dialog and export the last acquisition '
                    '(voltage, current, dI/dV, Gaussian fit) to a CSV file. Unchecks '
                    'itself automatically after export.'},
            {'title': 'Last export path:', 'name': 'last_export_path', 'type': 'str',
             'value': '', 'readonly': True},
        ]},
    ]

    def ini_attributes(self):
        self.controller_2410: Keithley2410 = None
        self.controller_2420: Keithley2420 = None
        self.x_axis = None
        self._last_voltages = None
        self._last_currents = None
        self._last_all_sweeps = None
        self._last_div = None
        self._last_fitted = None
        self._data_ready = False
        self._is_grabbing = False

    def _apply_scan_range(self):
        if self.controller_2410 is None:
            return

        voltages = self.controller_2410.init_balayage(
            voltMin=self.settings['scan_settings', 'volt_min'],
            voltMax=self.settings['scan_settings', 'volt_max'],
            NV=self.settings['scan_settings', 'n_points'],
            compliance=self.settings['k2410_settings', 'compliance'],
            current_range=self.settings['k2410_settings', 'current_range']
        )

        vrange = self.settings['k2410_settings', 'voltage_range']
        with self.controller_2410._lock:
            self.controller_2410.instrument.write(f':SOUR:VOLT:RANG {vrange}')

        self.x_axis = Axis(data=voltages, label='Grid Voltage', units='V', index=0)
        self._last_voltages = None
        self._last_currents = None

        self.dte_signal_temp.emit(DataToExport(
            name='RPA',
            data=[
                DataFromPlugins(
                    name='I-V Curve',
                    data=[np.zeros(len(voltages))],
                    dim='Data1D',
                    labels=['Collector Current [A]'],
                    axes=[self.x_axis]
                ),
                DataFromPlugins(
                    name='dI-dV (Energy Distribution)',
                    data=[np.zeros(len(voltages)), np.zeros(len(voltages))],
                    dim='Data1D',
                    labels=['dI/dV [A/V]', 'Gaussian fit'],
                    axes=[self.x_axis]
                ),
            ]
        ))

    def commit_settings(self, param: Parameter):
        """Apply parameter changes from the interface."""

        if param.name() == 'recompute_postproc':
            if param.value():
                self._recompute_postprocessing()
                param.setValue(False)
            return

        if param.name() == 'export_csv':
            if param.value():
                self._export_csv()
                param.setValue(False)
            return

        if param.name() == 'smooth_window':
            wl = int(param.value())
            if wl % 2 == 0:
                wl += 1
            poly = self.settings['postproc_settings', 'smooth_polyorder']
            if wl <= poly + 1:
                wl = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
            if wl != param.value():
                param.setValue(wl)
            return

        if param.name() == 'smooth_polyorder':
            poly = int(param.value())
            wl = self.settings['postproc_settings', 'smooth_window']
            if wl <= poly + 1:
                new_wl = poly + 3 if (poly + 3) % 2 == 1 else poly + 4
                self.settings.child('postproc_settings', 'smooth_window').setValue(new_wl)
            return

        if param.name() in ('volt_min', 'volt_max', 'n_points') and param.parent().name() == 'scan_settings':
            self._apply_scan_range()
            return

        if not hasattr(self, 'controller_2410') or self.controller_2410 is None:
            return
        if param.name() == 'compliance' and param.parent().name() == 'k2410_settings':
            self.controller_2410.instrument.write(f':SENS:CURR:PROT {param.value()}')
        elif param.name() == 'current_range' and param.parent().name() == 'k2410_settings':
            self.controller_2410.instrument.write(f':SENS:CURR:RANG {param.value()}')
        elif param.name() == 'voltage_range':
            self.controller_2410.instrument.write(f':SOUR:VOLT:RANG {param.value()}')
        elif param.name() == 'source_voltage' and hasattr(self, 'controller_2420'):
            self.controller_2420.set_source_voltage(param.value())
        elif param.name() == 'compliance' and param.parent().name() == 'k2420_settings':
            self.controller_2420.instrument.write(f':SENS:CURR:PROT {param.value()}')
        elif param.name() == 'current_range' and param.parent().name() == 'k2420_settings':
            self.controller_2420.instrument.write(':SENS:CURR:RANG:AUTO OFF')
            self.controller_2420.instrument.write(f':SENS:CURR:RANG {param.value()}')
        elif param.name() == 'nplc':
            self.controller_2420.instrument.write(f':SENS:CURR:NPLC {param.value()}')

    def ini_detector(self, controller=None):
        """Initialize communication with both the 2410 (grid) and 2420 (collector)."""
        if self.is_master:
            self.controller_2410 = Keithley2410(self.settings['k2410_settings', 'visa_address'])
            self.controller_2420 = Keithley2420(self.settings['k2420_settings', 'visa_address'])
            self.controller_2420.init_mesure(
                source_voltage=self.settings['k2420_settings', 'source_voltage'],
                compliance=self.settings['k2420_settings', 'compliance'],
                current_range=self.settings['k2420_settings', 'current_range'],
                nplc=self.settings['k2420_settings', 'nplc']
            )
            self._apply_scan_range()
            initialized = True
        else:
            self.controller_2410, self.controller_2420 = controller
            voltages = np.linspace(
                self.settings['scan_settings', 'volt_min'],
                self.settings['scan_settings', 'volt_max'],
                self.settings['scan_settings', 'n_points']
            )
            self.x_axis = Axis(data=voltages, label='Grid Voltage', units='V', index=0)
            initialized = True

        self._data_ready = False
        self._last_voltages = None
        self._last_currents = None
        self._last_all_sweeps = None
        self._last_div = None
        self._last_fitted = None

        if not self.is_master:
            voltages = self.x_axis.get_data()
            self.dte_signal_temp.emit(DataToExport(
                name='RPA',
                data=[
                    DataFromPlugins(
                        name='I-V Curve',
                        data=[np.zeros(len(voltages))],
                        dim='Data1D',
                        labels=['Collector Current [A]'],
                        axes=[self.x_axis]
                    ),
                    DataFromPlugins(
                        name='dI-dV (Energy Distribution)',
                        data=[np.zeros(len(voltages)), np.zeros(len(voltages))],
                        dim='Data1D',
                        labels=['dI/dV [A/V]', 'Gaussian fit'],
                        axes=[self.x_axis]
                    ),
                ]
            ))

        info = (f"2410 grid on {self.settings['k2410_settings', 'visa_address']}, "
                f"2420 collector on {self.settings['k2420_settings', 'visa_address']}")
        return info, initialized

    def close(self):
        """Close communication with both instruments."""
        if self.is_master:
            if self.controller_2410 is not None:
                self.controller_2410.close()
            if self.controller_2420 is not None:
                self.controller_2420.close()

    def _postprocess(self, voltages, currents):
        """
        Compute dI/dV, optionally the Gaussian fit, and the statistics
        (weighted mean, weighted std) of the energy distribution.
        Never raises.
        """
        voltages = np.asarray(voltages, dtype=float)
        currents = np.asarray(currents, dtype=float)
        n = len(voltages)

        div = derivative(
            voltages, currents,
            window_length=self.settings['postproc_settings', 'smooth_window'],
            polyorder=self.settings['postproc_settings', 'smooth_polyorder']
        )

        if self.settings['postproc_settings', 'invert_sign']:
            div = -div

        mean_w, std_w = weighted_mean_std(voltages, div)

        if self.settings['postproc_settings', 'enable_fit']:
            params, fitted = fit_gaussian(voltages, div)
            if fitted is None:
                fitted = np.zeros(n)
                mean_fit, sigma_fit = float('nan'), float('nan')
            else:
                _, mean_fit, sigma_fit, _ = params
                sigma_fit = abs(sigma_fit)
        else:
            fitted = np.zeros(n)
            mean_fit, sigma_fit = float('nan'), float('nan')

        return div, fitted, mean_w, std_w, mean_fit, sigma_fit

    def _update_stats_display(self, mean_w, std_w, mean_fit, sigma_fit):
        """Show the (weighted and fit) mean/std as read-only numeric fields
        directly in the Post-processing settings."""
        self.settings.child('postproc_settings', 'stat_mean').setValue(float(mean_w))
        self.settings.child('postproc_settings', 'stat_std').setValue(float(std_w))
        self.settings.child('postproc_settings', 'stat_mean_fit').setValue(float(mean_fit))
        self.settings.child('postproc_settings', 'stat_sigma_fit').setValue(float(sigma_fit))

    def _recompute_postprocessing(self):
        """
        Recompute dI/dV (+ fit) from the LAST acquisition in memory, without
        re-running a hardware sweep. Called from commit_settings when the
        'Recompute post-processing' checkbox is checked.
        """
        if self._last_voltages is None or self._last_currents is None:
            self.emit_status(ThreadCommand(
                'Update_Status',
                ['No acquisition in memory — run a sweep first (Grab/Snap).']
            ))
            return

        voltages = np.array(self._last_voltages)
        currents = np.array(self._last_currents)

        data_export = [
            DataFromPlugins(
                name='I-V Curve',
                data=[currents],
                dim='Data1D',
                labels=['Collector Current [A]'],
                axes=[self.x_axis]
            )
        ]

        if self.settings['postproc_settings', 'enable_derivative']:
            div, fitted, mean_w, std_w, mean_fit, sigma_fit = self._postprocess(voltages, currents)
            self._last_div, self._last_fitted = div, fitted

            data_export.append(DataFromPlugins(
                name='dI-dV (Energy Distribution)',
                data=[div, fitted],
                dim='Data1D',
                labels=['dI/dV [A/V]', 'Gaussian fit'],
                axes=[self.x_axis]
            ))

            if self.settings['postproc_settings', 'enable_fit'] and not np.isnan(mean_fit):
                status = [f'dI/dV recomputed. Distribution: <V>={mean_w:.2f} V, sigma={std_w:.2f} V '
                          f'(fit: mu={mean_fit:.2f} V, sigma={sigma_fit:.2f} V)']
            else:
                status = [f'dI/dV recomputed. Distribution (weighted): <V>={mean_w:.2f} V, '
                          f'sigma={std_w:.2f} V (Gaussian fit disabled or not converged)']

            self._update_stats_display(mean_w, std_w, mean_fit, sigma_fit)
        else:
            self._last_div = np.zeros(len(voltages))
            self._last_fitted = np.zeros(len(voltages))
            data_export.append(DataFromPlugins(
                name='dI-dV (Energy Distribution)',
                data=[self._last_div, self._last_fitted],
                dim='Data1D',
                labels=['dI/dV [A/V]', 'Gaussian fit'],
                axes=[self.x_axis]
            ))
            self._update_stats_display(float('nan'), float('nan'), float('nan'), float('nan'))
            status = ['Post-processing disabled ("Enable dI/dV" unchecked).']

        self.dte_signal.emit(DataToExport(name='RPA', data=data_export))
        self.emit_status(ThreadCommand('Update_Status', status))

    def _export_csv(self):
        """
        Open a save-file dialog and export the last acquisition (voltage,
        current, dI/dV, Gaussian fit if available) to a CSV file at the
        chosen path. Called from commit_settings when the 'Export CSV...'
        checkbox is checked.
        """
        if self._last_voltages is None or self._last_currents is None:
            self.emit_status(ThreadCommand(
                'Update_Status',
                ['No acquisition in memory — run a sweep first (Grab/Snap).']
            ))
            return

        default_name = f'RPA_export_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        path, _ = QFileDialog.getSaveFileName(
            None, 'Export RPA data to CSV', default_name, 'CSV files (*.csv)'
        )
        if not path:
            self.emit_status(ThreadCommand('Update_Status', ['CSV export cancelled.']))
            return

        voltages = np.array(self._last_voltages)
        currents = np.array(self._last_currents)
        n = len(voltages)

        div = self._last_div if self._last_div is not None else np.full(n, np.nan)
        fitted = self._last_fitted if self._last_fitted is not None else np.full(n, np.nan)

        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Grid Voltage [V]', 'Collector Current [A]',
                                  'dI/dV [A/V]', 'Gaussian fit [A/V]'])
                for i in range(n):
                    writer.writerow([voltages[i], currents[i], div[i], fitted[i]])
        except OSError as e:
            self.emit_status(ThreadCommand('Update_Status', [f'CSV export failed: {e}']))
            return

        self.settings.child('export_settings', 'last_export_path').setValue(path)
        self.emit_status(ThreadCommand('Update_Status', [f'Data exported to {path}']))

    def grab_data(self, Naverage=1, **kwargs):
        """
        Sweep the grid voltage (2410) and read the collector current (2420,
        averaged over n_average samples) at each voltage step, repeat the
        whole sweep n_sweeps times, then average the n_sweeps I-V curves
        obtained.

        dI/dV (+ fit) is computed automatically at the end of the
        acquisition. The 'Recompute post-processing' button remains useful
        to recompute without re-running a hardware sweep, e.g. after
        changing the smoothing window or toggling the fit.
        """
        if self._is_grabbing:
            self.emit_status(ThreadCommand('Update_Status',
                                           ['Acquisition already in progress, ignoring.']))
            return

        self._is_grabbing = True
        self._data_ready = False

        try:
            voltages = self.x_axis.get_data()
            delay = self.settings['scan_settings', 'delay'] / 1000.0
            n_avg = self.settings['scan_settings', 'n_average']
            n_sweeps = self.settings['scan_settings', 'n_sweeps']

            all_sweeps = []

            for sweep_idx in range(n_sweeps):
                self.controller_2410.set_voltage(voltages[0])

                currents = []
                for voltage in voltages:
                    self.controller_2410.set_voltage(voltage)
                    if delay > 0:
                        time.sleep(delay)

                    samples = [self.controller_2420.measure() for _ in range(n_avg)]
                    currents.append(float(np.mean(samples)))

                all_sweeps.append(currents)

                if n_sweeps > 1:
                    self.emit_status(ThreadCommand(
                        'Update_Status', [f'Sweep {sweep_idx + 1}/{n_sweeps} done.']
                    ))

            self.controller_2410.output_off()

            all_sweeps = np.array(all_sweeps)
            currents_mean = np.mean(all_sweeps, axis=0)
            currents_std = np.std(all_sweeps, axis=0) if n_sweeps > 1 else None

            self._last_voltages = list(voltages)
            self._last_currents = list(currents_mean)
            self._last_all_sweeps = all_sweeps
            self._data_ready = True

            n = len(voltages)
            data_export = [
                DataFromPlugins(
                    name='I-V Curve',
                    data=[currents_mean],
                    dim='Data1D',
                    labels=['Collector Current [A]'],
                    axes=[self.x_axis]
                ),
            ]

            status_msgs = [f'Acquisition complete: {n} points x {n_sweeps} sweep(s).']
            if n_sweeps > 1:
                status_msgs.append(
                    f'Inter-sweep noise (mean std): {np.mean(currents_std):.3e} A.'
                )

            if self.settings['postproc_settings', 'enable_derivative']:
                div, fitted, mean_w, std_w, mean_fit, sigma_fit = self._postprocess(
                    voltages, currents_mean
                )
                self._last_div, self._last_fitted = div, fitted
                data_export.append(DataFromPlugins(
                    name='dI-dV (Energy Distribution)',
                    data=[div, fitted],
                    dim='Data1D',
                    labels=['dI/dV [A/V]', 'Gaussian fit'],
                    axes=[self.x_axis]
                ))
                if self.settings['postproc_settings', 'enable_fit'] and not np.isnan(mean_fit):
                    status_msgs.append(
                        f'Distribution: <V>={mean_w:.2f} V, sigma={std_w:.2f} V '
                        f'(fit: mu={mean_fit:.2f} V, sigma={sigma_fit:.2f} V)'
                    )
                else:
                    status_msgs.append(
                        f'Distribution (weighted): <V>={mean_w:.2f} V, sigma={std_w:.2f} V '
                        f'(Gaussian fit disabled or not converged)'
                    )
                self._update_stats_display(mean_w, std_w, mean_fit, sigma_fit)
            else:
                self._last_div = np.zeros(n)
                self._last_fitted = np.zeros(n)
                data_export.append(DataFromPlugins(
                    name='dI-dV (Energy Distribution)',
                    data=[self._last_div, self._last_fitted],
                    dim='Data1D',
                    labels=['dI/dV [A/V]', 'Gaussian fit'],
                    axes=[self.x_axis]
                ))
                self._update_stats_display(float('nan'), float('nan'), float('nan'), float('nan'))
                status_msgs.append('Post-processing disabled ("Enable dI/dV" unchecked).')

            self.dte_signal.emit(DataToExport(name='RPA', data=data_export))
            self.emit_status(ThreadCommand('Update_Status', status_msgs))

        except Exception as e:
            self._data_ready = False
            self.emit_status(ThreadCommand('Update_Status', [f'Grab error: {e}']))
            import traceback
            traceback.print_exc()
        finally:
            self._is_grabbing = False

    def stop(self):
        """Stop acquisition and turn off the grid voltage."""
        if self.controller_2410 is not None:
            self.controller_2410.output_off()
        self.emit_status(ThreadCommand('Update_Status', ['Acquisition stopped']))
        return ''


if __name__ == '__main__':
    main(__file__)