# Changelog

## dev

### Changed

- `modelling` module got an overhaul. It now consists of smaller, self-consistent
  classes that are based on `attrs` and are read-only (but clonable). There is an
  explicit split between a `Model` and a `FixedLinearModel` which is defined at some
  coordinates `x`, allowing the latter to provide speedups for fitting lots of data at
  the same coordinates. They are also YAML-read/writable.


## 0.7.0

### Added

- Automatic notebook reports for calibration
- `xrfi_model_sweep` that actually works and is tested.
- Faster model evaluations

### Fixed

- xRFI doesn't assume that input spectrum is all positive (could be residuals, and
  therefore have negatives).

## Version 0.4.0

### Changed

- Much faster modeling via re-use of basis function evaluations

## Version 0.3.0

### Changed
- load_name now always an alias (hot_load, ambient, short, open)
- Load.temp_ave now always the correct one (even for hot load)

## Version 0.2.1

### Added

- Basic tests
- Travis/tox/codecov setup

## Version 0.2.0

### Added

- Many many many new features, and complete modularisation of code
- Now based on `edges-io` package to do the hard work.
- Refined most modules to remove redundant code
- Added better package structure

## Version 0.1.0

- First working version on github.
