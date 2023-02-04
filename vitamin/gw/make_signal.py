import numpy as np
import lal

speed_of_light = 3e8

def greenwich_mean_sidereal_time(time):
    time = float(time)
    return lal.GreenwichMeanSiderealTime(time)

def get_theta_phi(ra, dec, time):
    gmst = np.fmod(greenwich_mean_sidereal_time(time), 2*np.pi)

    phi = ra - gmst
    theta = np.pi/2 - dec
    return theta, phi

def time_delay_from_geocenter(detector, ra, dec, time):

    theta,phi = get_theta_phi(ra, dec, time)

    omega = [
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ]
    delta_d = detector["vertex"] - np.array([0,0,0])
    return np.dot(omega, delta_d)/speed_of_light

def get_polarisation_tensors(ra, dec, time, psi):

    theta,phi = get_theta_phi(ra, dec, time)

    u = np.array([
        np.cos(phi) * np.cos(theta), 
        np.cos(theta) * np.sin(phi), 
        -np.sin(theta)])

    v = np.array([
        -np.sin(phi), 
        np.cos(phi), 
        0])

    m = -u * np.sin(psi) - v * np.cos(psi)
    n = -u * np.cos(psi) + v * np.sin(psi)

    plus = np.einsum("i,j->ij", m, m) - np.einsum("i,j->ij", n, n)
    cross = np.einsum("i,j->ij", m, n) + np.einsum("i,j->ij", n, m)

    return {"plus":plus, "cross":cross}

def get_detector_response(detector, polarisations, parameters, start_time, frequencies, lal=False):

    signal = {}
    if lal:
        resp = {}
        resp["plus"],resp["cross"] = antenna_response(parameters["geocent_time"], parameters["ra"], parameters["dec"], parameters["psi"], "H1" )
    else:
        pol_tensors = get_polarisation_tensors(parameters["ra"], parameters["dec"], parameters["geocent_time"], parameters["psi"])

    for mode in polarisations.keys():
        if lal:
            signal[mode] = polarisations[mode]*resp[mode]
        else:
            response = np.einsum("ij, ij->", detector["detector_tensor"], pol_tensors[mode])
            signal[mode] = polarisations[mode]*response


    signal_ifo = sum(signal.values())

    time_shift = time_delay_from_geocenter(
        detector,
        parameters["ra"],
        parameters["dec"],
        parameters["geocent_time"]
    )

    dt = parameters["geocent_time"] - start_time + time_shift

    signal_ifo = signal_ifo * np.exp(-1j * 2 * np.pi * dt * frequencies)

    calibration_factor = 1
    signal_ifo *= calibration_factor

    return signal_ifo

def antenna_response( gpsTime, ra, dec, psi, det ):
    """
    Get the response of a detector to plus and cross polarisation signals.
    
    Args:
        gpsTime (float): the GPS time of the observations
        ra (float): the right ascension of the source (rads)
        dec (float): the declination of the source (rads)
        psi (float): the polarisation angle of the source (rads)
        det (str): a detector name (e.g., 'H1' for the LIGO Hanford detector)
    
    Returns:
        The plus and cross response.
    """
    
    gps = lal.LIGOTimeGPS( gpsTime )
    gmst_rad = lal.GreenwichMeanSiderealTime(gps)

    # create detector-name map
    detMap = {'H1': lal.LALDetectorIndexLHODIFF,
              'H2': lal.LALDetectorIndexLHODIFF,
              'L1': lal.LALDetectorIndexLLODIFF,
              'G1': lal.LALDetectorIndexGEO600DIFF,
              'V1': lal.LALDetectorIndexVIRGODIFF,
              'T1': lal.LALDetectorIndexTAMA300DIFF}

    try:
        detector=detMap[det]
    except KeyError:
        raise ValueError("ERROR. Key {} is not a valid detector name.".format(det))

    # get detector
    detval = lal.CachedDetectors[detector]

    response = detval.response

    # actual computation of antenna factors
    fp, fc = lal.ComputeDetAMResponse(response, ra, dec, psi, gmst_rad)

    return fp, fc


def get_detectors(dets):

    detmap = {'H1': lal.LALDetectorIndexLHODIFF,
              'L1': lal.LALDetectorIndexLLODIFF,
              'V1': lal.LALDetectorIndexVIRGODIFF}

    detectors = {}

    for i,det in enumerate(dets):
        detval = lal.CachedDetectors[detmap[det]]
        detectors[i] = {}
        detectors[i]["detector_tensor"] = detval.response

        detectors[i]["vertex"] = get_vertex_position_geocentric(
            detval.frDetector.vertexLongitudeRadians, 
            detval.frDetector.vertexLatitudeRadians, 
            detval.frDetector.vertexElevation
            )

    return detectors

def get_vertex_position_geocentric(latitude, longitude, elevation):
    """
    Calculate the position of the IFO vertex in geocentric coordinates in meters.

    Based on arXiv:gr-qc/0008066 Eqs. B11-B13 except for the typo in the definition of the local radius.
    See Section 2.1 of LIGO-T980044-10 for the correct expression

    Parameters
    ==========
    latitude: float
        Latitude in radians
    longitude:
        Longitude in radians
    elevation:
        Elevation in meters

    Returns
    =======
    array_like: A 3D representation of the geocentric vertex position

    """
    semi_major_axis = 6378137  # for ellipsoid model of Earth, in m
    semi_minor_axis = 6356752.314  # in m
    radius = semi_major_axis**2 * (semi_major_axis**2 * np.cos(latitude)**2 +
                                   semi_minor_axis**2 * np.sin(latitude)**2)**(-0.5)
    x_comp = (radius + elevation) * np.cos(latitude) * np.cos(longitude)
    y_comp = (radius + elevation) * np.cos(latitude) * np.sin(longitude)
    z_comp = ((semi_minor_axis / semi_major_axis)**2 * radius + elevation) * np.sin(latitude)
    return np.array([x_comp, y_comp, z_comp])
