# DESI Image Analysis for the Exposure Time Calculator

## Sky Camera

### Calibration

Collect calibration data using, e.g.
```
python calibrate.py -v -b 3 -T 10 --nzero 20 --ndark 50 --tdark 60 --nflat 4 --tflat 0.5,1,1.5,2 --outpath /Data/STXL/ref3
```
Analyze with:
```
etccalib -v --stxl-path /Data/STXL/ref3 --outpath ref3_calib.fits --binning 3
```

### Simulation

Simulate 
```
simskycam --bgraw /Data/STXL/ref3/dark_{N}.fits --calib ref3_calib.fits -v --outname docs/nb/ref3.csv -n 2000
```

### Measurement
