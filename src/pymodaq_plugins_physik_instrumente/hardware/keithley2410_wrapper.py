# -*- coding: utf-8 -*-
"""
Python wrapper for the Keithley 2410 (sourcemeter).
Handles low-level communication via pyvisa (GPIB).
Singleton pattern: actuator and detector share the same GPIB connection.
"""

import time
import numpy as np
import pyvisa

_instances = {}
_ref_counts = {}
_initialized_addresses = set()


class Keithley2410:

    def __new__(cls, adresse: str):
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
        if self._connected:
            return
        rm = pyvisa.ResourceManager()
        print("Connected devices:")
        [print('\t ->', element) for element in rm.list_resources()]
        time.sleep(0.2)
        self.instrument = rm.open_resource(adresse)
        self.instrument.timeout = 5000
        self.instrument.read_termination = '\n'
        self.instrument.write_termination = '\n'
        print(self.instrument.query('*IDN?'))
        self._connected = True

    def init_balayage(self, voltMin: float, voltMax: float, NV: int,
                      compliance: float = 700e-3, current_range: float = None):
        """Initialize the Keithley for a voltage sweep with current measurement.

        Parameters
        ----------
        voltMin : float - minimum sweep voltage [V]
        voltMax : float - maximum sweep voltage [V]
        NV : int - number of measurement points
        compliance : float - current limit [A] (default 700 mA)
        current_range : float or None - current range [A], None = autorange
        """
        volts = np.linspace(voltMin, voltMax, NV)
        if self._adresse in _initialized_addresses:
            return volts
        voltRang = abs(voltMin) + abs(voltMax)
        self.instrument.write('*RST')
        self.instrument.write('*CLS')
        self.instrument.write(':SYST:BEEP:STAT OFF')
        self.instrument.write(':SENS:FUNC "CURR"')
        if current_range is None:
            self.instrument.write(':SENS:CURR:RANG:AUTO ON')
        else:
            self.instrument.write(f':SENS:CURR:RANG {current_range}')
        self.instrument.write(f':SENS:CURR:PROT {compliance}')
        self.instrument.write(':SOUR:FUNC VOLT')
        self.instrument.write(':SOUR:VOLT:MODE FIX')
        self.instrument.write(f':SOUR:VOLT:RANG {voltRang}')
        self.instrument.write(f':SOUR:VOLT:LEV {voltMin}')
        _initialized_addresses.add(self._adresse)
        return volts

    def set_voltage(self, volt: float):
        """Apply a voltage on the Keithley output."""
        self.instrument.write(':OUTP ON')
        self.instrument.write(f':SOUR:VOLT:LEV {volt}')

    def get_voltage(self) -> float:
        """Read the voltage currently being sourced."""
        response = self.instrument.query(':SOUR:VOLT:LEV?')
        return float(response.strip())

    def get_current(self) -> float:
        """Read the current measured by the Keithley."""
        response = self.instrument.query(':READ?')
        values = response.split(',')
        return float(values[1])

    def measure(self, volt: float, stabilization_delay: float = 0.05) -> tuple:
        """Apply a voltage and return the measured (voltage, current).

        Parameters
        ----------
        volt : float - voltage to apply [V]
        stabilization_delay : float - wait time before reading [s] (default 50ms)

        Returns
        -------
        tuple : (voltage [V], current [A])
        """
        self.set_voltage(volt)
        time.sleep(stabilization_delay)
        response = self.instrument.query(':READ?')
        values = response.split(',')
        return float(values[0]), float(values[1])

    def output_off(self):
        """Turn off the Keithley output."""
        self.instrument.write(':OUTP OFF')

    def close(self):
        """Turn off the output and close the connection."""
        adresse = self._adresse
        _ref_counts[adresse] = max(0, _ref_counts.get(adresse, 1) - 1)
        if _ref_counts[adresse] > 0:
            return
        self.output_off()
        self.instrument.close()
        self._connected = False
        _instances.pop(adresse, None)
        _initialized_addresses.discard(adresse)