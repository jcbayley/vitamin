from gwpy.timeseries import TimeSeries

if __name__ == "__main__":
    ref_time = 1305029268
    real_noise_seg = [ref_time - 1, ref_time + 1]
    ts = TimeSeries.get('H1:GDS_CALIB_STRAIN', ref_time + 0.5, ref_time - 0.5)
    print(ts)
