# -*- coding: utf-8 -*-
"""
Python wrapper for the Keithley 2410 (sourcemeter).
Handles low-level communication via pyvisa (GPIB).

This file is meant to be placed in the hardware/ folder of the PyMoDAQ plugin.
It can also be used standalone in Langmuir scripts.

IMPORTANT: this class is a singleton indexed by VISA address. If two
PyMoDAQ plugins (the actuator and the detector) point to the same GPIB
address, they automatically share the same physical connection instead
of each opening their own (which caused a GPIB deadlock/blocking issue).
"""

import numpy as np
import pyvisa

_instances = {}
_ref_counts = {}
_initialized_addresses = set()


class Keithley2410:

    def __new__(cls, adresse: str):
        # Always return the same instance for a given VISA address
        if adresse in _instances:
            _ref_counts[adresse] += 1
            return _instances[adresse]

        instance = super().__new__(cls)
        _instances[adresse] = instance
        _ref_counts[adresse] = 1
        instance._adresse = adresse
        instance._connected = False
        return instance

    def __init__(self, adresse: str):
        # __init__ runs every time, even on the shared instance
        # -> only actually connect once
        if self._connected:
            return

        rm = pyvisa.ResourceManager()
        print("Connected devices:")
        [print('\t ->', element) for element in rm.list_resources()]
        self.instrument = rm.open_resource(adresse)
        self.instrument.timeout = 5000
        self.instrument.read_termination = '\n'
        self._connected = True

    def init_scan(self, voltMin: float, voltMax: float, NV: int,
                  compliance: float = 700e-3, current_range: float = 20e-3):
        """Initialize the Keithley for a voltage sweep with current measurement.

        Only performs the *rst / hardware configuration once per address,
        even if both the actuator and the detector plugins call this method
        during their own initialization. Always returns the sweep voltages.

        Parameters
        ----------
        voltMin : float - minimum sweep voltage [V]
        voltMax : float - maximum sweep voltage [V]
        NV : int - number of measurement points
        compliance : float - current limit [A] (default 700 mA)
        current_range : float - current measurement range [A] (default 20 mA)

        Returns
        -------
        volts : array - list of measurement voltages
        """
        volts = np.linspace(voltMin, voltMax, NV)

        if self._adresse in _initialized_addresses:
            # Already configured by the other plugin (actuator or detector):
            # don't redo the *rst so we don't wipe out the current config
            return volts

        voltRang = abs(voltMin) + abs(voltMax)
        self.instrument.write('*rst')
        self.instrument.write(':SYST:BEEP:STAT OFF')
        self.instrument.write(f':SENS:CURR:PROT {compliance}')
        self.instrument.write(':SOUR:FUNC VOLT')
        self.instrument.write(':SOUR:VOLT:MODE FIX')
        self.instrument.write(f':SOUR:VOLT:RANG {voltRang}')
        self.instrument.write(f':SOUR:VOLT:LEV {voltMin}')
        self.instrument.write(f':SENS:CURR:RANG {current_range}')

        _initialized_addresses.add(self._adresse)
        return volts

    def set_voltage(self, volt: float):
        """Apply a voltage on the Keithley output.

        Parameters
        ----------
        volt : float - voltage to apply [V]
        """
        self.instrument.write(':OUTP ON')
        self.instrument.write(f':SOUR:VOLT:LEV {volt}')

    def get_voltage(self) -> float:
        """Read the voltage currently being sourced.

        Returns
        -------
        float - sourced voltage [V]
        """
        response = self.instrument.query(':SOUR:VOLT:LEV?')
        return float(response.strip())

    def read_current(self) -> float:
        """Read the current measured by the Keithley.

        Returns
        -------
        float - measured current [A]
        """
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[1])

    def measure(self, volt: float) -> tuple:
        """Apply a voltage and return the measured (voltage, current).

        Parameters
        ----------
        volt : float - voltage to apply [V]

        Returns
        -------
        tuple : (voltage [V], current [A])
        """
        self.set_voltage(volt)
        response = self.instrument.query('READ?')
        values = response.split(',')
        return float(values[0]), float(values[1])

    def output_off(self):
        """Turn off the Keithley output."""
        self.instrument.write(':OUTP OFF')

    def close(self):
        """Turn off the output and close the connection.

        Only actually closes the physical connection once no plugin
        (actuator or detector) still uses this address. This way, closing
        the detector doesn't break a still-active actuator, and vice versa.
        """
        adresse = self._adresse
        _ref_counts[adresse] = max(0, _ref_counts.get(adresse, 1) - 1)

        if _ref_counts[adresse] > 0:
            # another plugin still uses this connection: don't close anything
            return

        self.output_off()
        self.instrument.close()
        self._connected = False
        _instances.pop(adresse, None)
        _initialized_addresses.discard(adresse)