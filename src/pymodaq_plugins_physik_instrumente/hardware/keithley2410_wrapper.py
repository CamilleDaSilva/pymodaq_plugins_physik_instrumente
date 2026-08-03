# -*- coding: utf-8 -*-
"""
Python wrapper for the Keithley 2410 (sourcemeter).
Handles low-level communication via pyvisa (GPIB).
Singleton pattern: actuator and detector share the same GPIB connection.
Thread-safe: a lock protects every write/query since the move plugin and
the 1D viewer can access the same instrument from different threads.
"""

import time
import threading
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
        instance._lock = threading.Lock()
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
        with self._lock:
            print(self.instrument.query('*IDN?'))
        self._connected = True

    def init_balayage(self, voltMin: float, voltMax: float, NV: int,
                      compliance: float = 700e-3, current_range: float = None):
        """Initialize the Keithley for a voltage sweep (source V, measure I optional).

        Parameters
        ----------
        voltMin : float - minimum sweep voltage [V]
        voltMax : float - maximum sweep voltage [V]
        NV : int - number of measurement points
        compliance : float - current limit [A] (default 700 mA)
        current_range : float or None - current range [A], None = autorange
        """
        volts = np.linspace(voltMin, voltMax, NV)

        # Always re-init (remove the early-return that prevented reconfiguration)
        voltRang = abs(voltMin) + abs(voltMax)
        if voltRang < 20:
            voltRang = 20.0  # minimum useful range for ±20 V tests

        with self._lock:
            self.instrument.write('*RST')
            self.instrument.write('*CLS')
            self.instrument.write(':SYST:BEEP:STAT OFF')

            # Source voltage, fixed mode
            self.instrument.write(':SOUR:FUNC VOLT')
            self.instrument.write(':SOUR:VOLT:MODE FIX')
            self.instrument.write(f':SOUR:VOLT:RANG {voltRang}')
            self.instrument.write(f':SOUR:VOLT:LEV {voltMin}')

            # Sense current (even if we mostly use the 2420 for the RPA)
            self.instrument.write(':SENS:FUNC "CURR"')
            self.instrument.write(f':SENS:CURR:PROT {compliance}')
            if current_range is None:
                self.instrument.write(':SENS:CURR:RANG:AUTO ON')
            else:
                self.instrument.write(':SENS:CURR:RANG:AUTO OFF')
                self.instrument.write(f':SENS:CURR:RANG {current_range}')

            # Format: voltage, current (useful if we ever read from 2410)
            self.instrument.write(':FORM:ELEM VOLT,CURR')

        _initialized_addresses.add(self._adresse)
        return volts

    def set_voltage(self, volt: float):
        """Apply a voltage on the Keithley output."""
        with self._lock:
            self.instrument.write(':OUTP ON')
            self.instrument.write(f':SOUR:VOLT:LEV {volt}')

    def get_voltage(self) -> float:
        """Read the voltage currently being sourced."""
        with self._lock:
            response = self.instrument.query(':SOUR:VOLT:LEV?')
        return float(response.strip())

    def get_current(self) -> float:
        """Read the current measured by the Keithley (if sensing on 2410)."""
        with self._lock:
            response = self.instrument.query(':READ?')
        values = response.split(',')
        # FORM:ELEM VOLT,CURR → index 1 is current
        return float(values[1]) if len(values) > 1 else float(values[0])

    def measure(self, volt: float, stabilization_delay: float = 0.05) -> tuple:
        """Apply a voltage and return the measured (voltage, current)."""
        self.set_voltage(volt)
        time.sleep(stabilization_delay)
        with self._lock:
            response = self.instrument.query(':READ?')
        values = response.split(',')
        return float(values[0]), float(values[1])

    def output_off(self):
        """Turn off the Keithley output."""
        with self._lock:
            self.instrument.write(':OUTP OFF')

    def close(self):
        """Turn off the output and close the connection."""
        adresse = self._adresse
        _ref_counts[adresse] = max(0, _ref_counts.get(adresse, 1) - 1)
        if _ref_counts[adresse] > 0:
            return
        self.output_off()
        with self._lock:
            self.instrument.close()
        self._connected = False
        _instances.pop(adresse, None)
        _initialized_addresses.discard(adresse)
