""" Basics Module

This module contains basic functions that are used 
as building blocks in the RADEX fitting package.

This module requires that numpy are maptplotlib are installed.

It contains the following functions:

    * gaussian : Gaussian function
    * TB : Calculates brightness temperature from Jy/beam data
    * conf_int : Computes confidence interval of a 1-dimensional 
                probability distribution
    * compute_Pgrid : Computes probability grid with constant 
                filling factor and abundance ratios
    * compute_Pgrid_bffit : Computes probability grid with fixed 
                abundance ratios and fitted filling factor
    * compute_Pgrid_TBratiofit : Computes probability grid using 
                ratios of intensities
    * compute_Pgrid_bffratiofit : Computes probability grid, fitting 
                both filling factor and abundance ratios
    * compute_Pgrid_ratiofit : Computes probability grid with a fixed 
                filling factor and fitted abundance ratios
    * CalculateStats : Calculates statistics from a probability array
    * PlotCorner : Plots a corner plot of a probability array and its stats
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
from matplotlib.legend_handler import HandlerTuple


def gaussian(dat, A0, sigx, x0):
    """Gaussian function
    
    Parameters
    ----------
    dat : scalar or array
        data to compute Gaussian for
    A0 : scalar
        normalization factor
    sigx : scalar
        distribution width/variance
    x0 : scalar
        center of distribution
        
    Returns
    -------
    y : scalar or array
        Gaussian computed for dat, same type as dat
    """
    
    return A0 * np.exp(-(dat - x0)**2/(2*sigx**2))


def TB(dat, bmin, bmaj, freq):
    """Calculates brightness temperature
    
    Parameters
    ----------
    dat : scalar or array
        data in Jy/beam
    bmin : scalar
        beam minor axis in arcsec
    bmaj : scalar
        beam major axis in arcsec (interchangeable with bmin)
    freq : scalar
        frequency of dat in GHz
    
    Returns
    -------
    TB : scalar or array
        Computed brightness temperature for cube, same type as dat
    """
    
    return 1.222 * 10**3 * (dat * 1000) / (freq**2 * bmin * bmaj)


def conf_int(dist, CI):
    """Computes confidence interval of a 1-dimensional probability distribution
    
    Parameters
    ----------
    dist : array
        one-dimensional distribution to find confidence interval of
    CI : scalar
        value of confidence interval, must be less than 1
    
    Returns
    -------
    ij : list
        indices bounding the confidence interval in distribution array
    """
    
    ndist = dist / np.sum(dist, axis=None) # Normalize the distribution
    
    # Start with full distribution: 
    # Sum under curve is 1.0 and indices [i, j] are [0, len(dist)-1]
    Ptot = 1.0
    i, j = 0, len(dist)-1
    ilast = True # Track whether i or j was adjusted last
    
    while Ptot > CI:
        # Determine which end of the distribution is smaller to remove from interval
        # Update either i or j and subtract from total probability under the curve
        # Continue until target percentage is reached
        if ndist[i] < ndist[j]:
            Ptot -= ndist[i]
            ilast = True
            i+=1
        else:
            Ptot -= ndist[j]
            j-=1
            ilast = False
    
    # Return final i and j
    if ilast:
        return [i-1, j]
    else:
        return [i, j+1]

    
def compute_Pgrid(Mols, Arrays, Js, Ints, Delts, minmaxes, Rindex, Rshifts, Nshift0, grids, verbose=False, bff=1.0):
    """ Compute probability grid - constant filling factor and abundance ratios
    
    Parameters
    ----------
    Mols : list
        list of strings indicating molecules to match input intensities
    Arrays : list
        list of arrays of temps, densities, and columns to fit
    Js : list
        list of lower-level energy states to match input intensities
    Ints : list
        list of input intensities
    Delts : list
        list of rms error in input intensities
    minmaxes : list
        list of minimum and maximum temps, densities, and columns to consider
    Rindex : list
        list of which abundance ratio to use for each input intensity
    Rshifts : list
        list of abundance ratios quantified as number of indices to shift column
    Nshift0 : scalar or list
        shift in column index so model grids of each molecule start at same column
    grids : list
        list of pre-computed RADEX model grids
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    bff : scalar, optional
        filling factor to use in fitting, default is 1.0
    
    Returns
    -------
    PcubeFINAL : array
        output array containing probabilities, with shape determine by Arrays
    """
    
    Tarr, narr, Narr = Arrays
    Tmin, Tmax, nmin, nmax, Nmin, Nmax = minmaxes
    
    # Initiate probability array based on number of lines and lengths of parameter arrays
    Pcube = np.empty((len(Mols), len(Tarr), len(narr), len(Narr)))
    
    # For each emission line, get appropriate set of RADEX model outputs and compare to data
    for i, I in enumerate(Ints):
        if verbose:
            print(100.*float(i)/len(Ints), '% computing', datetime.datetime.now())
        if Rindex[i]==0:
            #modelInts = grids[Mols[i]][5][Tmin:Tmax, nmin:nmax, Nmin:Nmax, 0, Js[i]]
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, Nmin:Nmax, Js[i]]
        # If not the default abundance molecule, shift column indices to get correct range
        else:
            #modelInts = grids[Mols[i]][5][Tmin:Tmax, nmin:nmax, int(Nmin-Rshifts[Rindex[i]-1]+Nshift0):int(Nmax-Rshifts[Rindex[i]-1]+Nshift0), 0, Js[i]]
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, int(Nmin-Rshifts[Rindex[i]-1]+Nshift0):int(Nmax-Rshifts[Rindex[i]-1]+Nshift0), Js[i]]
        
        # Calculate probabilities
        Pcube[i] = (Delts[i])**(-1.) * np.exp(-0.5 * ((I - modelInts*bff)/(Delts[i]))**2)
        
    if verbose:
        print('Done computing grid!!!')
    
    # Convert cube to log scale to manage really small numbers better and convert nans to 0
    PC = np.log10(np.nan_to_num(Pcube))
    # Sum over all emission lines to get final pcube
    PcubeFINAL = np.nan_to_num(np.nansum(PC, axis=0)) 
    
    # Get max probability, normalize probability array, and convert back to linear scaling
    maxp = np.nanmax(PcubeFINAL)
    PcubeFINAL = 10.**(PcubeFINAL - maxp)
    
    return PcubeFINAL


def compute_Pgrid_bffit(Mols, Arrays, Js, Ints, Delts, minmaxes, Rindex, Rshifts, Nshift0, grids, verbose=False):
    """ Compute probability grid - fixed abundance ratios, fitted filling factor
    
    Parameters
    ----------
    Mols : list
        list of strings indicating molecules to match input intensities
    Arrays : list
        list of arrays of temps, densities, columns, and filling factors to fit
    Js : list
        list of lower-level energy states to match input intensities
    Ints : list
        list of input intensities
    Delts : list
        list of rms error in input intensities
    minmaxes : list
        list of minimum and maximum temps, densities, and columns to consider
    Rindex : list
        list of which abundance ratio to use for each input intensity
    Rshifts : list
        list of abundance ratios quantified as number of indices to shift column
    Nshift0 : scalar or list
        shift in column index so model grids of each molecule start at same column
    grids : list
        list of pre-computed RADEX model grids
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    
    Returns
    -------
    PcubeFINAL : array
        output array containing probabilities, with shape determine by Arrays
    """
    
    Tarr, narr, Narr, bffarr = Arrays
    Tmin, Tmax, nmin, nmax, Nmin, Nmax = minmaxes

    # Initiate probability array based on number of lines and lengths of parameter arrays
    Pcube = np.empty((len(Mols), len(bffarr), len(Tarr), len(narr), len(Narr)))

    # For each emission line, get appropriate set of RADEX model outputs and compare to data
    for i, I in enumerate(Ints):
        # If not the default abundance molecule, shift column indices to get correct range
        if Rindex[i]==0:
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, Nmin:Nmax, Js[i]]
        else:
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, int(Nmin-Rshifts[Rindex[i]-1]+Nshift0):int(Nmax-Rshifts[Rindex[i]-1]+Nshift0), Js[i]]
            
        # Add tiled dimension the size of filling factor array 
        modelInts = np.tile(modelInts, (len(bffarr), 1, 1, 1))
        
        # Calculate probabilities, multiplying model intensities by filling factors
        Pcube[i] = ((Delts[i])**(-1.) * np.exp(-0.5 * (((I).T - modelInts.T*bffarr)/(Delts[i]))**2)).T

    # Multiply together all emission lines to get final pcube and convert nans to 0
    PcubeFINAL = np.nan_to_num(np.prod(Pcube, axis=0))
    PcubeFINAL = np.moveaxis(PcubeFINAL, 0, -1)

    return PcubeFINAL


def compute_Pgrid_TBratiofit(Mols, Arrays, Js, Ints, Delts, minmaxes, Rindex, Rshifts, Nshift0, grids, verbose=False, denomi=0):
    """ Compute probability grid using ratios of intensities
    
    Parameters
    ----------
    Mols : list
        list of strings indicating molecules to match input intensities
    Arrays : list
        list of arrays of temps, densities, and columns to fit
    Js : list
        list of lower-level energy states to match input intensities
    Ints : list
        list of input intensities
    Delts : list
        list of rms error in input intensities
    minmaxes : list
        list of minimum and maximum temps, densities, and columns to consider
    Rindex : list
        list of which abundance ratio to use for each input intensity
    Rshifts : list
        list of abundance ratios quantified as number of indices to shift column
    Nshift0 : scalar or list
        shift in column index so model grids of each molecule start at same column
    grids : list
        list of pre-computed RADEX model grids
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    denomi : int, optional
        index of line to use as denominator in intensity ratios, default is 0
    
    Returns
    -------
    PcubeFINAL : array
        output array containing probabilities, with shape determine by Arrays
    """
    
    Tarr, narr, Narr = Arrays
    Tmin, Tmax, nmin, nmax, Nmin, Nmax = minmaxes
    Delts = np.array(Delts)
    Ints = np.array(Ints)
    
    # Initiate probability and model arrays based on number of lines and lengths of parameter arrays
    Pcube = np.empty((len(Mols)-1, len(Tarr), len(narr), len(Narr)))
    modelInts = np.empty((len(Ints), len(Tarr), len(narr), len(Narr)))
    
    # For each emission line, get appropriate set of RADEX model outputs and compare to data
    for i, I in enumerate(Ints):
        # If not the default abundance molecule, shift column indices to get correct range
        if Rindex[i]==0:
            modelInts[i] = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, Nmin:Nmax, Js[i]]
        else:
            modelInts[i] = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, int(Nmin-Rshifts[Rindex[i]-1]+Nshift0):int(Nmax-Rshifts[Rindex[i]-1]+Nshift0), Js[i]]
         
    # Create mask where denomi is True and the rest are False and an inverted one for numerators
    denom = np.full(len(Mols), False)
    denom[denomi] = True
    others = np.invert(denom)[:]
    
    # Calulate observed and modeled intensity ratios
    ratios = Ints[others]/Ints[denom]
    modelratios = modelInts[others]/modelInts[denom]
    # Propaget errors to get error in observed ratios
    rdelts = np.sqrt( (Delts[others]/Ints[denom])**2 + (Ints[others]*Delts[denom]/Ints[denom]**2)**2)
        
    # Calculate probabilities for each ratio
    for i in np.arange(len(Mols)-1, dtype=int):
        Pcube[i] = (rdelts[i])**(-1.) * np.exp(-0.5 * ((ratios[i] - modelratios[i])/(rdelts[i]))**2)
        
    #  Convert cube to log scale to manage really small numbers better and convert nans to 0
    PC = np.log10(np.nan_to_num(Pcube))
    # Sum over molecules to get final probability and convert nans to 0
    PcubeFINAL = np.nan_to_num(np.nansum(PC, axis=0)) 
    # Get max probability, normalize probability array, and convert back to linear scaling
    maxp = np.nanmax(PcubeFINAL)
    PcubeFINAL = 10.**(PcubeFINAL - maxp)
    
    return PcubeFINAL  


def compute_Pgrid_bffratiofit(Mols, Arrays, Js, Ints, Delts, minmaxes, Rindex, Rshifts, Nshift0, grids, verbose=False):
    """ Compute probability grid - fit both filling factor and abundance ratios
    
    Parameters
    ----------
    Mols : list
        list of strings indicating molecules to match input intensities
    Arrays : list
        list of arrays of temps, densities, columns, and filling factors to fit
    Js : list
        list of lower-level energy states to match input intensities
    Ints : list
        list of input intensities
    Delts : list
        list of rms error in input intensities
    minmaxes : list
        list of minimum and maximum temps, densities, and columns to consider
    Rindex : list
        list of which abundance ratio to use for each input intensity
    Rshifts : list
        list of abundance ratios quantified as number of indices to shift column
    Nshift0 : scalar or list
        shift in column index so model grids of each molecule start at same column
    grids : list
        list of pre-computed RADEX model grids
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    
    Returns
    -------
    PcubeFINAL : array
        output array containing probabilities, with shape determine by Arrays
    """
    
    Tarr, narr, Narr, bffarr, *_ = Arrays
    Tmin, Tmax, nmin, nmax, Nmin, Nmax = minmaxes
    
    # Initiate probability array based on number of lines and lengths of parameter arrays
    Pcube = np.empty((len(Mols),    len(bffarr), len(Tarr), len(narr), len(Narr)))
    
    # Add a fitting dimension to the probability array for every abundance ratio being fit
    # Then move added dimension to being after the emission line dimension
    for arr in Rshifts[::-1]:
        temp = np.ones(Pcube.ndim+1, dtype=int)
        temp[0] = len(arr)
        Pcube = np.moveaxis(np.tile(Pcube, temp), 0, 1)
    
    # Loop through all molecules and abundance ratios without needing to know how many dimensions are involved
    for idx, _ in np.ndenumerate(Pcube[...,0,0,0,0]):
        i = idx[0]
        # If not the default abundance molecule, shift column indices to get correct range
        if Rindex[i]==0:
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, Nmin:Nmax, Js[i]]
        else:
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, int(Nmin-Rshifts[Rindex[i]-1][idx[Rindex[i]]]+Nshift0):int(Nmax-Rshifts[Rindex[i]-1][idx[Rindex[i]]]+Nshift0), Js[i]]
        
        # Add tiled dimension for filling factor in modeled intensity array
        modelInts = np.tile(modelInts, (len(bffarr), 1, 1, 1))
        
        # Calculate probabilities, multiplying modeled intensities by filling factors
        Pcube[idx] = ((Delts[i])**(-1.) * np.exp(-0.5 * (((Ints[i]).T - modelInts.T*bffarr)/(Delts[i]))**2)).T
    
    # Multiply together all emission lines to get final pcube and convert nans to 0
    PcubeFINAL = np.nan_to_num(np.prod(Pcube, axis=0))
    
    # Shift filling factor dimension of probability array to the end
    PcubeFINAL = np.moveaxis(PcubeFINAL, -4, -1)
    # Shift abundance ratio dimensions of probability array to the end
    for arr in Rshifts:
        PcubeFINAL = np.moveaxis(PcubeFINAL, 0, -1)

    return PcubeFINAL


def compute_Pgrid_ratiofit(Mols, Arrays, Js, Ints, Delts, minmaxes, Rindex, Rshifts, Nshift0, grids, verbose=False, bff=1.0):
    """ Compute probability grid - fixed filling factor, fitted abundance ratios
    
    Parameters
    ----------
    Mols : list
        list of strings indicating molecules to match input intensities
    Arrays : list
        list of arrays of temps, densities, and columns to fit
    Js : list
        list of lower-level energy states to match input intensities
    Ints : list
        list of input intensities
    Delts : list
        list of rms error in input intensities
    minmaxes : list
        list of minimum and maximum temps, densities, and columns to consider
    Rindex : list
        list of which abundance ratio to use for each input intensity
    Rshifts : list
        list of abundance ratios quantified as number of indices to shift column
    Nshift0 : scalar or list
        shift in column index so model grids of each molecule start at same column
    grids : list
        list of pre-computed RADEX model grids
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    bff : scalar, optional
        filling factor to use in fitting, default is 1.0
    
    Returns
    -------
    PcubeFINAL : array
        output array containing probabilities, with shape determine by Arrays
    """
    
    Tarr, narr, Narr, *_ = Arrays
    Tmin, Tmax, nmin, nmax, Nmin, Nmax = minmaxes
    
    # Initiate probability array based on number of lines and lengths of parameter arrays
    Pcube = np.empty((len(Mols), len(Tarr), len(narr), len(Narr)))
    
    # Add a fitting dimension to the probability array for every abundance ratio being fit
    # Then move added dimension to being after the emission line dimension
    for arr in Rshifts[::-1]:
        temp = np.ones(Pcube.ndim+1, dtype=int)
        temp[0] = len(arr)
        Pcube = np.moveaxis(np.tile(Pcube, temp), 0, 1)

    # Loop through all molecules and abundance ratios without needing to know how many dimensions are involved
    for idx, _ in np.ndenumerate(Pcube[...,0,0,0]):
        i = idx[0]
        # If not the default abundance molecule, shift column indices to get correct range
        if Rindex[i]==0:
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, Nmin:Nmax, Js[i]]
        else:
            modelInts = grids[Mols[i]][4][Tmin:Tmax, nmin:nmax, int(Nmin-Rshifts[Rindex[i]-1][idx[Rindex[i]]]+Nshift0):int(Nmax-Rshifts[Rindex[i]-1][idx[Rindex[i]]]+Nshift0), Js[i]]
        
        # Calculate probabilities
        Pcube[idx] = ((Delts[i])**(-1.) * np.exp(-0.5 * ((Ints[i] - modelInts.T*bff)/(Delts[i]))**2)).T

    # Multiply together all emission lines to get final pcube and convert nans to 0
    PcubeFINAL = np.nan_to_num(np.prod(Pcube, axis=0))
    # Shift abundance ratio dimensions of probability array to the end
    for arr in Rshifts:
        PcubeFINAL = np.moveaxis(PcubeFINAL, 0, -1)
        
    return PcubeFINAL



def CalculateStats(Pcube, Arrays, verbose=False):
    """ Calculate statistics from probability array
    
    Parameters
    ----------
    Pcube : array
        array of probabilities output from a compute_Pgrid_* function
    Arrays : list
        list of arrays of temps, densities, and columns used in fitting
    
    Returns
    -------
    stats : array
        indices of highest probability parameters and statistical intervals
    """
    
    ndim = Pcube.ndim
    maxp = np.nanmax(Pcube)
    # Get number of dimensions and maximum probability to normalize distributions
    
    # If maxp is 0, no need to proceed further, return all nans
    if maxp==0.0 or np.isnan(Pcube).all():
        if verbose:
            print('cube is all nans or maxp is 0')
            print('maxp = ', maxp)
        res = np.full((ndim*12),(np.nan))
        return res
    
    totalP = np.nansum(Pcube)
    if verbose:
        print('max P: ', maxp)
        print('total P: ', totalP)
    
    # Get locations of best-fit parameter
    maxi = np.unravel_index(np.nanargmax(Pcube), Pcube.shape)
    # Initialize arrays to hold best-fit parameters and uncertainty ranges
    maxes = np.empty(ndim)
    maxes_p = np.empty(ndim)
    intervals = np.empty(ndim*10, dtype=int)
    
    # Calculate Bayes factor for probability cube then get 3D masks for each interval threshold
    BayesCube = np.abs(np.log(maxp / Pcube))
    mask5 = BayesCube < 5.0
    mask2 = BayesCube < 2.5
    mask1 = BayesCube < 1.0
    
    # Make array of indices for each element of Pcube to use in determining bayesian intervals
    indices = np.indices(Pcube.shape)
    
    # Loop over each dimension to calculate its stats
    for i in np.arange(ndim):
        # Try to get value of best-fit parameter and return nan if unsuccessful
        try:
            maxes[i] = Arrays[i][maxi[i]]
        except:
            maxes[i] = np.nan
        if verbose:
            print('i = ', maxi[i])
            print('max val = ', maxes[i])
        
        # Determine dimensions to sum over to get 1D distribution, then normalize it
        dims = np.arange(ndim, dtype=int)
        sumdims = dims[(dims!=i)]
        prof = np.nan_to_num(np.nansum(Pcube, axis=tuple(sumdims))) / totalP
        
        # Get best-fit parameter value of 1D probability distribution or return nan
        try:
            maxes_p[i] = Arrays[i][np.nanargmax(prof)]
        except:
            maxes_p[i] = np.nan
            
        # Add 95% and 67% confidence interval indices of 1D probability distribution to interval array
        intervals[10*i:10*i+2] = conf_int(prof, 0.95)
        intervals[10*i+2:10*i+4] = conf_int(prof, 0.67)
        if verbose:
            try:
                print('CIs: ', np.take(Arrays[i], np.array(intervals[10*i:10*i+4], dtype=int)))
            except:
                print('CIs (indices): ', intervals[10*i:10*i+4])

        # Add bayesian interval indices from 3D probability distribution to interval array
        intervals[10*i+4] = np.nanmin(indices[i][mask5])
        intervals[10*i+5] = np.nanmax(indices[i][mask5])
        intervals[10*i+6] = np.nanmin(indices[i][mask2])
        intervals[10*i+7] = np.nanmax(indices[i][mask2])
        intervals[10*i+8] = np.nanmin(indices[i][mask1])
        intervals[10*i+9] = np.nanmax(indices[i][mask1])
        if verbose:
            try:
                print('BIs: ', np.take(Arrays[i], np.array(intervals[10*i+4:10*i+10],dtype=int)))
            except:
                print('BISs (indices): ', intervals[10*i+4:10*i+10])
    
    return np.concatenate((maxes, maxes_p, intervals))


def PlotCorner(Arrays, Pcube, Stats, params, model=np.array([None]), verbose=False, cmap='Greys', scale='linear', plotintervals=[1,5], LTE=np.array([None]), title=None, save=False, fname='CornerPlot', fmt='png'):
    """ Plot a corner plot of a probability array and its stats
    
    Parameters
    ----------
    Arrays : list
        list of arrays of temps, densities, and columns used in fitting
    Pcube : array
        array of probabilities output from a compute_Pgrid_* function
    Stats : array
        indices of statistics output from the CalculateStats function
    params : list
        list of parameter name strings to print in plots
    model : array, optional
        list of modeled 'true' parameters, default is None
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    cmap : str, optional
        color map used in plotting 2D probability distributions, default is 'Greys'
    scale : str, optional
        whether to use linear or logarithmic scaling in all distribution plotting
    plotintervals : list, optional
        list of strings or ints indicating statistical intervals to plot, 
        default is [1, 5], which corresponds to CI95 and BI1
    LTE : array, optional
        list of LTE-derived temperature and columns, default is None
    title : str, optional
        title to print on plot, default is None
    save : bool, optional
        flag of whether to save plot or not, default is False
    fname : str, optional
        file name to use if saving plot, default is 'CornerPlot'
    fmt : str, optional
        file format to use if saving plot, default is 'png'
    """
    
    # Determine number of dimensions that will need to be plotted
    ndim = Pcube.ndim
    
    # Parse output of CalculateStats function
    maxes = Stats[:len(params)]
    maxes_p = Stats[len(params):2*len(params)]
    intervals = np.array(np.reshape(Stats[len(params)*2: len(params)*12], (len(params), 5, 2)), dtype=int)
    
    # Initiate lists needed to plot and label desired statistical intervals
    names = []
    intervalkwargs = []
    handles = []
    # Loop through requested statistical intervals to plot
    for n in np.arange(len(plotintervals)):
        # If strings given, convert to ints
        if isinstance(plotintervals[n], str):
            if plotintervals[n] == 'CI95':
                plotintervals[n] = 1
            elif plotintervals[n] == 'CI67':
                plotintervals[n] = 2
            elif plotintervals[n] == 'BI5':
                plotintervals[n] = 3
            elif plotintervals[n] == 'BI2.5':
                plotintervals[n] = 4
            elif plotintervals[n] == 'BI1':
                plotintervals[n] = 5
        else:
            plotintervals[n] = int(plotintervals[n])
        
        # Depending on desired stat interval, add its name and plotting kwargs
        if plotintervals[n]==1:
            names.append('95% Confidence Interval')
            intervalkwargs.append({'facecolor':'orange', 'alpha':0.2})
        elif plotintervals[n]==2:
            names.append('67% Confidence Interval')
            intervalkwargs.append({'facecolor':'None', 'edgecolor':'orange', 'alpha':0.5, 'hatch':'//'})
        elif plotintervals[n]==3:
            names.append('5.0 Bayesian Interval')
            intervalkwargs.append({'facecolor':'darkblue', 'alpha':0.2})
        elif plotintervals[n]==4:
            names.append('2.5 Bayesian Interval')
            intervalkwargs.append({'facecolor':'c', 'alpha':0.2})
        elif plotintervals[n]==5:
            names.append('1.0 Bayesian Interval')
            intervalkwargs.append({'facecolor':'None', 'edgecolor':'c', 'alpha':0.5, 'hatch':'//'})
    
    totalP = np.nansum(Pcube)
    
    # Initiate figure array with dimensions to match pcube
    f, axarr = plt.subplots(ndim, ndim, figsize=(8,6))#, sharex='col')
        
    dims = np.arange(ndim, dtype=int)
        
    # Loop through every pairing of parameter dimensions with i and j
    for i in range(ndim):
        # Get 1D probability distribution profile by summing over every other dimension
        profdims = dims[(dims!=i)]
        prof = np.nan_to_num(np.nansum(Pcube, axis=tuple(profdims)) / totalP)
        # Convert to log scale if desired
        if scale=='log':
            prof = np.nan_to_num(np.log10(prof))
        
        for j in range(ndim):
            if i<j:
                # For off-diagonal subplots : 
                
                # Get 2D probability distribution by summing over every other dimension but i and j
                sumdims = dims[(dims!=i) * (dims!=j)]
                # Plot 2D distribution with log or linear scaling
                if scale=='log': 
                    axarr[j][i].pcolormesh(Arrays[i], Arrays[j], np.log10(np.nansum(Pcube, axis=tuple(sumdims)).T / totalP), cmap=cmap, vmin=-10, vmax=0)
                else:
                    axarr[j][i].pcolormesh(Arrays[i], Arrays[j], np.nansum(Pcube, axis=tuple(sumdims)).T / totalP, cmap=cmap)
                
                # Mark locations of best-fit 3D and 1D parameters with xs
                h1 = axarr[j][i].scatter(maxes[i], maxes[j], c='darkcyan', marker='x')
                h2 = axarr[j][i].scatter(maxes_p[i], maxes_p[j], c='darkorange', marker='x')
                # Plot location of 'true' modeled values if given
                if np.array(model).any():
                    h3 = axarr[j][i].scatter(model[i], model[j], c='mediumvioletred', marker='x')
                # Plot location of LTE-derived values if given    
                if np.array(LTE).any():
                    if i==0 and j==2:
                        h4 = axarr[j][i].scatter(LTE[i], LTE[j-1], c='limegreen', marker='x')
                     
            elif j==i:
                # For diagonal subplots : 
                
                # Plot 1D probability profile
                axarr[j][i].plot(Arrays[i], prof, color='k')
                # Determine min and maxes of profile with limits in case of log scaling which could get super negative
                max1, min1 = np.nanmax([np.nanmax(prof), -9.9]), np.nanmax([np.nanmin(prof), -10.])
                # Try creating a y-axis range to use when plotting vertical lines of best-fit values
                try:
                    range1 = np.arange(min1, 2*max1 - min1, (max1-min1))[:2]
                except:
                    max1 = max1+0.5
                    min1 = min1-0.5
                    range1 = np.arange(min1, 2*max1 - min1, (max1-min1))[:2]
                if scale=='linear':
                    # If scaling is linear, range should always start at 0
                    range1[0] = 0
                # Plot vertical lines of locations of best-fit 3D and 1D value
                h5, = axarr[j][i].plot(np.ones(2)*maxes[i], range1, color='darkcyan', linestyle='--', label='3D Pmax')
                h6, = axarr[j][i].plot(np.ones(2)*maxes_p[i], range1, color='darkorange', linestyle='--', label='Profile Pmax')
                # Plot vertical lines of locations of 'true' modeled value if given
                if np.array(model).any():
                    h7, = axarr[j][i].plot(np.ones(2)*model[i], range1, color='mediumvioletred', linestyle='--')
                # Plot vertical lines of locations of LTE-derived value if given
                if np.array(LTE).any():
                    if i==0:
                        h8, = axarr[j][i].plot(np.ones(2)*LTE[i], range1, color='limegreen', linestyle=':')
                    if i==2:
                        h8, = axarr[j][i].plot(np.ones(2)*LTE[i-1], range1, color='limegreen', linestyle=':')
                
                # For each statistical interval requested, add shading to indicate its range
                # Limits are different if scaling is log vs linear
                for ni, n in enumerate(plotintervals):
                    if scale=='linear':
                        h = axarr[j][i].fill_between(Arrays[i][intervals[i,n-1,0]:intervals[i,n-1,1]+1], prof[intervals[i,n-1,0]:intervals[i,n-1,1]+1], label=names[ni], **intervalkwargs[ni])
                    else:
                        h = axarr[j][i].fill_between(Arrays[i][intervals[i,n-1,0]:intervals[i,n-1,1]+1], prof[intervals[i,n-1,0]:intervals[i,n-1,1]+1], -10., label=names[ni], **intervalkwargs[ni])
                    if i==0:
                        handles.append(h)
                    
                # Set y axis limits based on log vs linear scaling
                if scale=='linear':
                    axarr[j][i].set_ylim(0.0,max1)
                else:
                    axarr[j][i].set_ylim(min1,max1)
            
            # Turn off subplots above the diagonal
            elif i>j:
                axarr[j][i].axis('off')   
            
            # Add axis labels along bottom and left of subplot array
            if i==0 and j!=0:
                axarr[j][i].set_ylabel(params[j])
            if j==ndim-1:
                axarr[j][i].set_xlabel(params[i])
            
            # Set x axis limits based on parameter array size
            axarr[j][i].set_xlim(Arrays[i][0], Arrays[i][-1])
    
    # Add legend handlers to handles and names lists based on whether they were used
    # Only one of each type are added instead of one for each subplot
    handles.insert(0, (h2, h6))
    handles.insert(0, (h1, h5))
    names.insert(0, 'Profile Pmax')
    names.insert(0, '3D Pmax')
    if np.array(LTE).any():
        handles.insert(2, (h4, h8))
        names.insert(2, 'LTE Solution')
    if np.array(model).any():
        handles.insert(2, (h3, h7))
        names.insert(2, 'Model Value')
    
    # Add legend from handles and names above
    f.legend(handles, names, handler_map={tuple: HandlerTuple(ndivide=None)})
    
    # Add plot title if one is given
    if title:
        f.suptitle(title)
        
    plt.tight_layout()
    
    # Save or show final plot
    if save:
        plt.savefig(fname+'.'+fmt)
    else:
        plt.show()
    
    return  
 

    