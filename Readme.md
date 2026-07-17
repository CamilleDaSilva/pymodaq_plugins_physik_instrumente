# PICOMAX-E environment setup (PyMoDAQ + Keithley 2410)

This folder lets you install everything needed to run the PyMoDAQ Keithley
2410 plugin (viewer and move) on a new computer.

## Step 1 — Manual prerequisites (once per computer)

These two installations go through external graphical installers, so they
are not automated by `setup.bat`.

### 1.1. Install Miniforge

- Download the Windows installer from:
  https://github.com/conda-forge/miniforge#download
- Run the installer, keep the default options.
- Then open the Start menu and launch **Miniforge Prompt** (not a regular
  `cmd.exe`) for all the commands below.

### 1.2. Install the NI-VISA drivers

Required for Python to communicate with the Keithley over GPIB.

- Download NI-VISA from the National Instruments website:
  https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html
- Run the installer, restart the computer if prompted.

### 1.3. Copy/clone the plugin repo

Place the `pymodaq_plugins_physik_instrumente` folder (or clone it from Git)
somewhere on the new computer, for example:

```
C:\Users\<user>\Desktop\local_repository\pymodaq_plugins_physik_instrumente
```

## Step 2 — Automatic installation

1. Copy `setup.bat` (provided next to this README) directly into the plugin
   folder, next to `pyproject.toml`.
2. Open **Miniforge Prompt**.
3. Move into the plugin folder:
   ```
   cd C:\Users\<user>\Desktop\local_repository\pymodaq_plugins_physik_instrumente
   ```
4. Run:
   ```
   setup.bat
   ```

The script:
- cleans up leftover mamba/conda locks,
- creates the `pymodaq_env` environment (Python 3.12),
- installs PyQt from conda-forge,
- installs the plugin in editable mode (`pip install -e . --no-deps`),
- installs the remaining dependencies (`bitstring`, `msl-loadlib`, `pipython`),
- checks that PyVISA properly detects the connected instruments.

## Step 3 — Day-to-day use

Once installation is complete, for every new session simply open
**Miniforge Prompt** then:

```
conda activate pymodaq_env
```

Then, you can type dashboard to get the PyMoDAQ dashboard.

## If you're behind a corporate proxy

If `mamba create` or `mamba install` fail with proxy errors
(`ConnectTimeoutError`, `proxy.companyname`...), you need to set the proxy in
the `%USERPROFILE%\.condarc` file:

```yaml
proxy_servers:
  http: http://proxy.companyname:<port>
  https: http://proxy.companyname:<port>
```

Replace `<port>` with the port actually used by your company's proxy.
You can also contact your IT department.

## If you get a mamba lockfile error

If `setup.bat` fails with a message containing `Could not open lockfile`,
relaunch **Miniforge Prompt as administrator**, or manually delete:

```
C:\ProgramData\miniforge3\pkgs\pkgs.lock
```

then rerun `setup.bat`.