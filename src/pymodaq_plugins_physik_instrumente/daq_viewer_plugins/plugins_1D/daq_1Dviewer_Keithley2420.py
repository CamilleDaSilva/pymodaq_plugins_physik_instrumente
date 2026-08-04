# -*- coding: utf-8 -*-
"""
PyMoDAQ DAQ_1DViewer plugin for the RPA (Retarding Potential Analyzer).
Sweeps the grid voltage on the Keithley 2410 and reads the collector
current on the Keithley 2420 at each step, directly building the I-V
curve — no PyMoDAQ DAQ_Scan involved (mirrors the Langmuir 1D viewer
approach).

Post-processing (Étape 1 + 2):
    - dI/dV (Savitzky-Golay derivative)
    - mean / std of the ion energy distribution
    - Gaussian fit superimposed on dI/dV
    - Fine control of Savitzky-Golay parameters in the GUI
    - Independent enable/disable of the Gaussian fit
    - Option to plot -dI/dV (useful when current is negative)

Author: Camille Da Silva
ONERA DPHY/CSE — PICOMAX-E, 2026
"""

import time
import numpy as np

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

        # ── Keithley 2410 — grid voltage (retarding voltage) ─────────────────
        {'title': 'Keithley 2410 (Grille)', 'name': 'k2410_settings', 'type': 'group', 'children': [
            {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str',
             'value': 'GPIB0::24::INSTR'},
            {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float',
             'value': 700e-3, 'min': 0.0, 'max': 1.05},
            {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float',
             'value': 20e-3, 'min': 0.0, 'max': 1.05,
             'tip': 'Calibre du 2410 (si on lit aussi son courant). Pour test résistance ~20 mA'},
            {'title': 'Voltage Range (V):', 'name': 'voltage_range', 'type': 'float',
             'value': 40.0, 'min': 0.0, 'max': 210.0,
             'tip': 'Doit couvrir |Vmin| + |Vmax|'},
        ]},

        # ── Keithley 2420 — collector current (ammeter) ──────────────────────
        {'title': 'Keithley 2420 (Collecteur)', 'name': 'k2420_settings', 'type': 'group', 'children': [
            {'title': 'VISA Address:', 'name': 'visa_address', 'type': 'str',
             'value': 'GPIB1::25::INSTR',
             'tip': 'Adresse VISA du Keithley 2420 (courant collecteur)'},
            {'title': 'Source Voltage (V):', 'name': 'source_voltage', 'type': 'float',
             'value': 0.0, 'min': -60.0, 'max': 60.0,
             'tip': 'Tension imposée au collecteur (0 V pour usage ampèremètre pur)'},
            {'title': 'Compliance (A):', 'name': 'compliance', 'type': 'float',
             'value': 100e-3, 'min': 0.0, 'max': 3.0},
            {'title': 'Current Range (A):', 'name': 'current_range', 'type': 'float',
             'value': 20e-3, 'min': 1e-9, 'max': 3.0,
             'tip': 'Calibre de mesure — dimensionnez-le au-dessus du courant max attendu '
                    '(pour 1.18 kΩ @ ±20 V → ~17 mA → 20 mA est parfait)'},
            {'title': 'NPLC:', 'name': 'nplc', 'type': 'float',
             'value': 1.0, 'min': 0.01, 'max': 10.0},
        ]},

        # ── Scan parameters ──────────────────────────────────────────────────
        {'title': 'Scan Settings', 'name': 'scan_settings', 'type': 'group', 'children': [
            {'title': 'Voltage Min (V):', 'name': 'volt_min', 'type': 'float',
             'value': -20.0, 'min': -210.0, 'max': 210.0},
            {'title': 'Voltage Max (V):', 'name': 'volt_max', 'type': 'float',
             'value': 20.0, 'min': -210.0, 'max': 210.0},
            {'title': 'N points:', 'name': 'n_points', 'type': 'int',
             'value': 41, 'min': 2, 'max': 2500},
            {'title': 'N Average:', 'name': 'n_average', 'type': 'int',
             'value': 5, 'min': 1, 'max': 100,
             'tip': "Nombre d'échantillons moyennés par point de tension"},
            {'title': 'Delay (ms):', 'name': 'delay', 'type': 'float',
             'value': 100.0, 'min': 0.0, 'max': 10000.0,
             'tip': 'Temps de stabilisation après application de la tension de grille'},
        ]},

        # ── Post-processing (Étape 1 + 2) ─────────────────────────────────────
        {'title': 'Post-traitement', 'name': 'postproc_settings', 'type': 'group', 'children': [
            {'title': 'Activer dI/dV:', 'name': 'enable_derivative', 'type': 'bool',
             'value': True,
             'tip': 'Calcule et affiche la dérivée dI/dV (distribution en énergie)'},
            {'title': 'Afficher -dI/dV:', 'name': 'invert_derivative', 'type': 'bool',
             'value': False,
             'tip': 'Inverse le signe de la dérivée. Utile quand le courant est négatif '
                    '(convention récepteur) pour avoir une distribution positive.'},
            {'title': 'Activer fit gaussien:', 'name': 'enable_fit', 'type': 'bool',
             'value': True,
             'tip': 'Ajuste et superpose une gaussienne sur dI/dV (nécessite dI/dV activé)'},
            {'title': "Fenêtre de lissage (pts, impair):", 'name': 'smooth_window', 'type': 'int',
             'value': 9, 'min': 3, 'max': 501,
             'tip': "Largeur de la fenêtre Savitzky-Golay. "
                    "Doit rester << N points. Augmenter si dI/dV trop bruité."},
            {'title': 'Ordre polynôme lissage:', 'name': 'smooth_polyorder', 'type': 'int',
             'value': 2, 'min': 1, 'max': 5,
             'tip': 'Ordre du polynôme local (classiquement 2 ou 3). Doit être < fenêtre.'},
        ]},
    ]

    def ini_attributes(self):
        self.controller_2410: Keithley2410 = None
        self.controller_2420: Keithley2420 = None
        self.x_axis = None
        self._last_voltages = None
        self._last_currents = None
        self._data_ready = False
        self._is_grabbing = False

    def commit_settings(self, param: Parameter):
        """Apply parameter changes from the interface."""
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
        # Les paramètres 'postproc_settings' n'agissent qu'au prochain grab_data.

    def ini_detector(self, controller=None):
        """Initialize communication with both the 2410 (grid) and 2420 (collector)."""
        if self.is_master:
            self.controller_2410 = Keithley2410(self.settings['k2410_settings', 'visa_address'])
            voltages = self.controller_2410.init_balayage(
                voltMin=self.settings['scan_settings', 'volt_min'],
                voltMax=self.settings['scan_settings', 'volt_max'],
                NV=self.settings['scan_settings', 'n_points'],
                compliance=self.settings['k2410_settings', 'compliance'],
                current_range=self.settings['k2410_settings', 'current_range']
            )

            # Force the voltage range explicitly
            vrange = self.settings['k2410_settings', 'voltage_range']
            with self.controller_2410._lock:
                self.controller_2410.instrument.write(f':SOUR:VOLT:RANG {vrange}')

            self.controller_2420 = Keithley2420(self.settings['k2420_settings', 'visa_address'])
            self.controller_2420.init_mesure(
                source_voltage=self.settings['k2420_settings', 'source_voltage'],
                compliance=self.settings['k2420_settings', 'compliance'],
                current_range=self.settings['k2420_settings', 'current_range'],
                nplc=self.settings['k2420_settings', 'nplc']
            )
            initialized = True
        else:
            self.controller_2410, self.controller_2420 = controller
            voltages = np.linspace(
                self.settings['scan_settings', 'volt_min'],
                self.settings['scan_settings', 'volt_max'],
                self.settings['scan_settings', 'n_points']
            )
            initialized = True

        self.x_axis = Axis(data=voltages, label='Grid Voltage', units='V', index=0)

        self._data_ready = False
        self._last_voltages = None
        self._last_currents = None

        # Deux courbes dès l'init : I-V brute + dI/dV (vide pour l'instant)
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
                    labels=['dI/dV [A/V]', 'Fit gaussien'],
                    axes=[self.x_axis]
                ),
            ]
        ))

        info = (f"2410 grille sur {self.settings['k2410_settings', 'visa_address']}, "
                f"2420 collecteur sur {self.settings['k2420_settings', 'visa_address']}")
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
        Calcule dI/dV (éventuellement inversé), le fit gaussien et les
        statistiques (moyenne, écart-type) de la distribution en énergie.

        Ne lève jamais d'exception : en cas d'échec renvoie des tableaux de
        zéros pour que l'affichage ne casse pas.
        """
        n = len(voltages)
        div = derivative(
            voltages, currents,
            window_length=self.settings['postproc_settings', 'smooth_window'],
            polyorder=self.settings['postproc_settings', 'smooth_polyorder']
        )

        if self.settings['postproc_settings', 'invert_derivative']:
            div = -div

        mean_w, std_w = weighted_mean_std(voltages, div)

        fitted = np.zeros(n)
        mean_fit, sigma_fit = float('nan'), float('nan')

        if self.settings['postproc_settings', 'enable_fit']:
            params, fitted_curve = fit_gaussian(voltages, div)
            if fitted_curve is not None:
                fitted = fitted_curve
                _, mean_fit, sigma_fit, _ = params
                sigma_fit = abs(sigma_fit)

        return div, fitted, mean_w, std_w, mean_fit, sigma_fit

    def grab_data(self, Naverage=1, **kwargs):
        """Sweep the grid voltage (2410) and read the collector current
        (2420, averaged over n_average samples) at each step to build the
        full I-V curve, then run the post-processing (dI/dV + fit) if
        enabled."""
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
            n_avg = self.settings['scan_settings', 'n_average']

            # Make sure both outputs are ON before the sweep
            self.controller_2410.set_voltage(voltages[0])

            for voltage in voltages:
                self.controller_2410.set_voltage(voltage)
                if delay > 0:
                    time.sleep(delay)

                samples = [self.controller_2420.measure() for _ in range(n_avg)]
                current = float(np.mean(samples))
                currents.append(current)

            # Turn grid off after acquisition
            self.controller_2410.output_off()

            self._last_voltages = list(voltages)
            self._last_currents = currents
            self._data_ready = True

            data_export = [
                DataFromPlugins(
                    name='I-V Curve',
                    data=[np.array(currents)],
                    dim='Data1D',
                    labels=['Collector Current [A]'],
                    axes=[self.x_axis]
                )
            ]

            status_msgs = [f'Acquisition done: {len(voltages)} points.']

            if self.settings['postproc_settings', 'enable_derivative']:
                div, fitted, mean_w, std_w, mean_fit, sigma_fit = self._postprocess(
                    voltages, currents
                )

                labels = ['dI/dV [A/V]']
                data_list = [div]

                if self.settings['postproc_settings', 'enable_fit']:
                    labels.append('Fit gaussien')
                    data_list.append(fitted)

                data_export.append(
                    DataFromPlugins(
                        name='dI-dV (Energy Distribution)',
                        data=data_list,
                        dim='Data1D',
                        labels=labels,
                        axes=[self.x_axis]
                    )
                )

                msg = f'Distribution: <V>={mean_w:.2f} V, σ={std_w:.2f} V'
                if self.settings['postproc_settings', 'enable_fit'] and not np.isnan(mean_fit):
                    msg += f'  |  fit: μ={mean_fit:.2f} V, σ={sigma_fit:.2f} V'
                status_msgs.append(msg)

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
