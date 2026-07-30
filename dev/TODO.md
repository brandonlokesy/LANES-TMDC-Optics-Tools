# TODO

## Refactoring
- AttoCubePLVabScan needs a rename
- AttoCubePLVabScan needs a rewrite
    - accept PL, R, RC etc. spectra. 
    - check format/dimension of spectra files
    - Check if it's possible to verify only one gate is being used. E.g. bottom gate doping of device. Changes how statements about the scan are printed
    - able to accept any kind of independent parameter/variable. now electric field sweeps are assumed to be the only parameter sweep. power, position etc are possible too.
- AttoCubeRealSpace ...
    - Select the range of frames of interest -> work on a subset
- Position in scans (x,y).
    - for x (..) for y (...) can be flipped to for y (...) for x (..). reverse the nested loop
- Reference class
    - Format spectra to return array, not per value sweeps

## Fitting
- Multiple ROIs for diffusion E.g. 2-3 diffusion spots. Track each for area and centre of mass
- Multiple ROIs for dipole length. E.g. in hybrid intralayer excitons in homobilayers with switchable dipole lengths based on the transitions

## Documentation
- Add mathematical formulas to functions that assume certain physics. E.g. the calculation of the electric field within the heterostructure stack.
