"""Fitting Methods Module

This module contains classes and functions that
perform the fitting processes and plot or 
otherwise save the output.

This module requires that numpy, maptplotlib, astropy, 
spectralradex, and pathos are installed. It also imports the 
modules basics.py and fittingloopfuncs.py from this package.

It contains the following classes:
    * RADEXFit : an object containing attributes of the desired fitting 
        process and that can execute the fitting process on input data
    * Pcube : a probability array object with attributes about the 
        fitting process used to generate it and output statistics
    * MapData : a set of data cubes with attributes about which emission 
        lines are represented
    
It contains the following functions:
    * PlotFittedMaps : Plots the output of the fitting process into 
        maps of fitted parameters
    * CreateRadexGrids : Create grids of pre-computed RADEX models
        that are used to find the best-fit physical parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patches
import datetime
import pickle
import basics
from astropy.io import fits
from astropy.stats import mad_std
from fittingloopfuncs import defineloop, definefixedloop
from pathos.multiprocessing import Pool
from spectralradex import radex

np.seterr(all='ignore')
# Probability arrays frequently return very small numbers that get rounded to zero,
# then divide by those zeros.
# We don't need to receive error messages about that!

class RADEXFit:
    """Contains attributes about and executes the fitting process
    
    Attributes
    ----------
    mols : list
        list of strings indicating molecules of input emission lines
    js : list
        list of lower-level energy states of input emission lines
    mode : str
        indicates with fitting mode to use in fit* methods, default is 'default'
    bff : scalar
        filling factor to use in fitting, default is 1.0
    bffarr : array
        array of filling factor used if fitting the filling factor,
        default is a range between 0.05 and 1.0 with 0.05 spacing
    denomi : int
        index of line to use as denominator in intensity ratios, default is 0
    Rindex : list
        list of which abundance ratio to use for each emission line,
        default is each unique molecular species is numbered from 0
    radexgrids : list
        list of pre-computed RADEX model grids for each molecule
    Tarr : array
        array of temperatures to fit from radexgrids
    narr : array
        array of volume densities to fit from radexgrids
    Narr : array
        array of column densities to fit from radexgrids
    Nshift0 : scalar or list
        shift in column index so model grids of each molecule start at same column
    Rarr : array
        array of abundance ratio values used when fitting the ratios
    Rshifts : array
        array of abundance ratios quantified as number of indices to shift column
    Rs : list
        list of abundance ratio values used when ratios are fixed, default is 100
    minmaxes : list
        list of minimum and maximum temps, densities, and columns to consider
    arrays : list
        list of parameter arrays to be used in fitting
    plotparams : list
        list of fitted parameters formatted for labels when plotting
    printparams : list
        list of fitted parameters formatted for printing
    
    Methods
    -------
    fit(data, error, verbose=False, arrays=[[None]], Rshifts=[[None]], minmaxes=[None])
        Executes the fitting process on a single data point
    fitlist(data, error, verbose=False, save=False, interval=None, fname='RadexListFittingOutput.dat')
        Executes the fitting process on a list of data points
    fitmap(dat, nprocs=None, linefixed=True, sigma_threshold=5, line_threshold=None, verbose=False, 
    extraverbose=False, save=True, fname='FittingResults',  plotpeak=False, plotall=False, 
    plotname='CornerPlot', fmt='png', scale='log', savePC=False, ClumpMask=np.array([None]), 
    clumparray=np.array([None]))
        Executes the fitting process on a MapData object
    """
    
    def __init__(self, mols, js, mode='default', bff=1.0, bffarr=np.arange(0.05, 1.01, 0.05), denomi=0, Bounds={}, radexgrids=None, verbose=False, radexgridfile='examplegrid.pkl', Rindex=[None], Rvals=[None]):
        """
        Parameters
        ----------
        mols : list
            list of strings indicating molecules of input emission lines
        js : list
            list of lower-level energy states of input emission lines
        mode : str, optional
            indicates with fitting mode to use in fit* methods, default is 'default'
        bff : scalar, optional
            filling factor to use in fitting, default is 1.0
        bffarr : array, optional
            array of filling factor used if fitting the filling factor,
            default is a range between 0.05 and 1.0 with 0.05 spacing
        denomi : int, optional
            index of line to use as denominator in intensity ratios, default is 0
        Bounds : dict, optional
            dictionary of minimum and maximum values to use for parameters, default is None,
            which results in using full range from radexgrids
        radexgrids : list, optional
            list of pre-computed RADEX model grids for each molecule, default is None, 
            which then loads grids from files located at radexgridfile string
        verbose : bool, optional
            flag used to determine frequency of print statements, default is False
        radexgridfile : str, optional
            suffix of file location to load pre-computed RADEX model grids from, 
            default is 'reducedradexgrid.pkl'
        Rindex : list, optional
            list of which abundance ratio to use for each emission line,
            default is each unique molecular species is numbered from 0
        Rvals : scalar or list, optional
            target abundance ratios to use when abundance ratios are fixed, closest possible 
            values are used, default is 100
        """
        
        # Add emission line identifiers and fitting method attributes
        self.mols = mols
        self.js = js
        self.mode = mode
        
        if not isinstance(Rvals, list):
            Rvals = [Rvals]
        
        # Depending on fitting method, add appropriate filling factor and abundance ratio
        # attributes, and denomi if doing TB ratio fitting
        if mode=='default' or mode=='fitratio':
            self.bff = bff
            self.bffarr = np.array([bff])
            self.denomi = None
            # If Rindex array isn't specified, create list where each unique species
            # uses its own abundance ratio
            if not np.array(Rindex).any():
                self.Rindex = np.empty(len(mols), dtype=int)
                _, idx = np.unique(mols, return_index=True)
                for m, mol in enumerate(np.array(mols)[np.sort(idx)]):
                    self.Rindex = np.where(np.array(mols)==mol, int(m), self.Rindex)
            else:
                self.Rindex = Rindex
            # Default abundance ratio is 100 (based on 12co/13co ratio)
            if not np.array(Rvals).any():
                Rvals = np.full(len(np.unique(self.Rindex))-1, 100)
                
        elif mode=='fitbff' or mode=='fitbffratio':
            self.bff = 1.
            self.bffarr = bffarr
            self.denomi = None
            # If Rindex array isn't specified, create list where each unique species
            # uses its own abundance ratio
            if not np.array(Rindex).any():
                self.Rindex = np.empty(len(mols), dtype=int)
                _, idx = np.unique(mols, return_index=True)
                for m, mol in enumerate(np.array(mols)[np.sort(idx)]):
                    self.Rindex = np.where(np.array(mols)==mol, int(m), self.Rindex)
            else:
                self.Rindex = Rindex
            # Default abundance ratio is 100 (based on 12co/13co ratio)
            if not np.array(Rvals).any():
                Rvals = np.full(len(np.unique(self.Rindex))-1, 100)
                
        elif mode=='fitTBratio':
            self.bff = 1.
            self.bffarr = np.array([1.])
            self.denomi = denomi
            # If Rindex array isn't specified, create list where each unique species
            # uses its own abundance ratio
            if not np.array(Rindex).any():
                self.Rindex = np.empty(len(mols), dtype=int)
                _, idx = np.unique(mols, return_index=True)
                for m, mol in enumerate(np.array(mols)[np.sort(idx)]):
                    self.Rindex = np.where(np.array(mols)==mol, int(m), self.Rindex)
            else:
                self.Rindex = Rindex
            # Default abundance ratio is 100 (based on 12co/13co ratio)
            if not np.array(Rvals).any():
                Rvals = np.full(len(np.unique(self.Rindex))-1, 100)
            
        # If already-loaded radexgrids are not passed, load them from radexgridfile
        if not radexgrids:
            self.radexgrids = {}
            print('Loading pickle files', datetime.datetime.now())
            # Load a grid for each unique molecular species, file prefix should match mols string
            for mol in np.unique(mols):
                self.radexgrids[mol] = pickle.load(open(mol+'.'+radexgridfile, 'rb'), encoding='bytes')
            print('Done loading pickle files', datetime.datetime.now())
            print()
            # Change the print statements here to a loading animation thing?
            
        else:
            # Or use supplied already loaded grids
            self.radexgrids = radexgrids
            
        # Separate temp, density, and column arrays from first three elements of radexgrids list
        self.Tarr = self.radexgrids[mols[0]][0]
        self.narr = self.radexgrids[mols[0]][1]
        self.Narr = self.radexgrids[mols[0]][2]
        
        # If the smallest density is larger than 50., density is not in log scale and should be
        if self.narr[-1] > 50.:
            self.narr = np.log10(self.narr)
            
            ### I could probably remove these check once I create a function to make standard radex grid files ###
        
        # Initiate array of Nshift0's to make sure column grids are lined up between molecules
        self.Nshift0 = []
        # If the smallest column is larger than 1000., column arrays are not in log scale and should be
        # Also need to line up other grids assuming they are not in log scale
        if self.Narr[-1] > 1000.:
            self.Narr = np.log10(self.Narr)
            dN = self.Narr[1] - self.Narr[0]
            # For each molecule, make sure columns are lined up and add necessary shift to Nshift0 list
            for m in np.arange(1,len(np.unique(mols))):
                self.Nshift0.append(int((self.Narr[0] - np.log10(self.radexgrids[mols[m]][2][0])) / dN))
        else:
            dN = self.Narr[1] - self.Narr[0]
            for m in np.arange(1,len(np.unique(mols))):
                self.Nshift0.append(int((self.Narr[0] - self.radexgrids[mols[m]][2][0]) / dN))
        # Nshift0 is for if the N arrays of different molecules have different ranges
        
        # Make sure Nshift0 still exists if only one molecule is being used 
        if not self.Nshift0:
            self.Nshift0 = [0]

        # Initialize empty lists for determining allowable abundance ratio shifts
        shifts = []
        Nshmins = []
        Nshmaxs = []
        self.Rarr = []
        self.Rshifts = []
        self.Rs = []
        
        # For each unique abundance ratio from the Rindex input, determine the 
        # minimum and maximum index shift available for the given column arrays
        # Use bounds dictionary to limit min and max if a 'R*' key is given
        # where * is the number in Rindex
        for ri in np.arange(1,len(np.unique(self.Rindex))):
            if 'R'+str(ri) in Bounds.keys():
                Nshmin = int(np.floor(np.log10(Bounds['R'+str(ri)][0]) / dN))+1
                Nshmax = int(np.floor(np.log10(Bounds['R'+str(ri)][1]) / dN))+1
            else:
                Nshmin = 0
                rNmin = min([self.radexgrids[m][2][0] for m in np.unique(np.take(self.mols, np.argwhere(np.array(self.Rindex)==ri)))])
                Nshmax = int(np.floor((self.Narr[-1]-rNmin) / dN))
                # If no bounds are given, might be good to still limit it to 1-2 orders of magnitude by default...
            
            # Add index shifts and the corresponding abundance ratio physical value to lists
            shifts.append(np.arange(Nshmin, Nshmax, 1, dtype=int))
            self.Rarr.append(10.**(shifts[-1]*dN))
            
            # If abundance ratios are not being fit, add only the shift and ratio value that is the closest
            # match to the input Rval or its default value of 100
            if mode=='default' or mode=='fitbff' or mode=='fitTBratio':
                self.Rshifts.append(shifts[-1][np.argmin(np.abs(self.Rarr[-1] - Rvals[ri-1]))])
                self.Rs.append(round(10.**(self.Rshifts[-1]*dN)))
                Nshmins.append(self.Rshifts[-1])
                Nshmaxs.append(self.Rshifts[-1])
            # If ratios are being fit, add all possible shifts calculated above
            else:
                self.Rshifts.append(shifts[-1])
                self.Rs.append(10.**(self.Rshifts[-1]*dN))
                Nshmins.append(Nshmin)
                Nshmaxs.append(Nshmax)
            
            # Print final abundance ratio possibilities and the value being used if fixed
            if verbose:
                print('Available Rs : ', self.Rarr[-1])
            if mode=='default' or mode=='fitbff' or mode=='fitTBratio':
                print('Using R = ', self.Rs[-1])
                print()
        
        # Make sure Nshmins and Nshmaxs still exist as zeros if there are no abundance ratios
        if not Nshmins:
            Nshmins = [0]
            Nshmaxs = [0]
        
        # If a bound is given for column, limit column array to the bounds
        # Also account for if abundance ratio fitting or Nshift0 limits range of columns that can be fitted
        if 'N' in Bounds.keys():
            Nmin = max([np.where((self.Narr - Bounds['N'][0]) > 0)[0][0] - 1, max(Nshmaxs)-min(self.Nshift0)])
            Nmax = min([np.where((self.Narr - Bounds['N'][1]) >= 0)[0][0] + 1, len(self.Narr)+min(Nshmins)-max(self.Nshift0)])
        else:
            Nmin = max([0,max(Nshmaxs)-min(self.Nshift0)])
            Nmax = min([len(self.Narr), len(self.Narr)+min(Nshmins)-max(self.Nshift0)])
        self.Narr = self.Narr[Nmin:Nmax]
        if verbose:
            print('N min max: ', Nmin, Nmax)
        
        # If a bound is given for temp, limit temp array to the bounds
        if 'T' in Bounds.keys():
            Tmin = np.where((self.Tarr - Bounds['T'][0]) > 0)[0][0] - 1
            Tmax = np.where((self.Tarr - Bounds['T'][1]) >= 0)[0][0] + 1
        else:
            Tmin = 0
            Tmax = len(self.Tarr)
        if verbose:
            print('T min max: ', Tmin, Tmax)
        self.Tarr = self.Tarr[Tmin:Tmax]
        
        # If a bound is given for density, limit density array to the bounds
        if 'n' in Bounds.keys():
            nmin = np.where((self.narr - Bounds['n'][0]) > 0)[0][0] - 1
            nmax = np.where((self.narr - Bounds['n'][1]) >= 0)[0][0] + 1
        else:
            nmin = 0
            nmax = len(self.narr)
        if verbose:
            print('n min max: ', nmin, nmax)
        self.narr = self.narr[nmin:nmax]

        # Create list with the min and max indices for temp, density, and column
        self.minmaxes = [Tmin, Tmax, nmin, nmax, Nmin, Nmax]
        
        # Create lists of the arrays to be fit and the names of the parameters for each type of fitting method
        if mode=='default' or mode=='fitTBratio':
            self.arrays = [self.Tarr, self.narr, self.Narr]
            self.plotparams=[r'$T_{kin}$ (K)', r'log($n_{H_2}$ / cm$^{-3}$)', r'log($N_{CO}$ / cm$^{-2}$)']
            self.printparams=['T_kin', 'log_n', 'log_N']
        elif mode=='fitbff':
            self.arrays = [self.Tarr, self.narr, self.Narr, self.bffarr]
            self.plotparams=[r'$T_{kin}$ (K)', r'log($n_{H_2}$ / cm$^{-3}$)', r'log($N_{CO}$ / cm$^{-2}$)', r'$ff$']
            self.printparams=['T_kin', 'log_n', 'log_N', 'ff']
        elif mode=='fitratio':
            self.arrays = [self.Tarr, self.narr, self.Narr]
            self.plotparams=[r'$T_{kin}$ (K)', r'log($n_{H_2}$ / cm$^{-3}$)', r'log($N_{CO}$ / cm$^{-2}$)']
            self.printparams=['T_kin', 'log_n', 'log_N']
            for ri in np.arange(1,len(np.unique(self.Rindex))):
                self.plotparams.append(r'$R_{'+self.mols[np.argwhere(self.Rindex==ri)[0][0]]+'}$')
                self.printparams.append('R'+str(ri))
                self.arrays.append(self.Rarr[ri-1])
        elif mode=='fitbffratio':
            self.arrays = [self.Tarr, self.narr, self.Narr, self.bffarr]
            self.plotparams=[r'$T_{kin}$ (K)', r'log($n_{H_2}$ / cm$^{-3}$)', r'log($N_{CO}$ / cm$^{-2}$)', r'$ff$']
            self.printparams=['T_kin', 'log_n', 'log_N', 'ff']
            for ri in np.arange(1,len(np.unique(self.Rindex))):
                self.plotparams.append(r'$R_{'+self.mols[np.argwhere(self.Rindex==ri)[0][0]]+'}$')
                self.printparams.append('R'+str(ri))
                self.arrays.append(self.Rarr[ri-1])
        
    
    def fit(self, data, error, verbose=False, arrays=[[None]], Rshifts=[[None]], minmaxes=[None]):
        """ Executes the fitting process on a single data point
        
        If arrays, Rshifts, or minmaxes are not passed to the method, the corresponding 
        attributes of the RADEXFit object are used.
        
        Parameters
        ----------
        data : list
            brightness temperature for each emission line of a single pixel
        error : list
            corresponding rms errors in the emission line data
        verbose : bool, optional
            flag used to determine frequency of print statements, default is False
        arrays : list, optional
            list of parameter arrays to be used in fitting process, default is None
        Rshifts : list, optional
            array of abundance ratios quantified as number of indices to shift column, default is None
        minmaxes : list, optional
            list of minimum and maximum temps, densities, and columns to consider, default is None
        
        Returns
        -------
        pcube : Pcube object
            Pcube object with the resulting probability distribution and its statistics
        """
        
        # Initiating a Pcube object runs the fitting on a single data point as input here
        pcube = Pcube(self, data, error, verbose, arrays, Rshifts, minmaxes=minmaxes)
        return pcube
    
    
    def fitlist(self, data, error, verbose=False, save=False, interval=None, fname='RadexListFittingOutput.dat'):
        """Executes the fitting process on a list of data points
        
        Parameters
        ----------
        data : array
            2D array of brightness temps of each line for a list of data points
        error : list
            corresponding rms errors in the emission line data
        verbose : bool, optional
            flag used to determine frequency of print statements, default is False
        save : bool, optional
            flag indicating whether the output should be saved to a file, default is False
        interval : int or str, optional
            statistical interval to use when determining fitted range of parameters,
            default is None, which results in using the most constrained of BI1 an CI95
        fname : str, optional
            file name to be used if saving, default is 'RadexListFittingOutput.dat'
        
        Returns
        -------
        stattable : array
            array of best-fit parameters and their uncertainty range
        """
        
        # Initiate a list of final stats to return
        stattable = []
        
        # If not stat interval is requested, use the default preferred interval
        if not interval:
            # Run fitting on each data point and add results to stattable
            for dat in data:
                pcube = Pcube(self, dat, error, verbose)
                stattable.append(pcube.stats)
        # If a specific stat interval is requested, add that to stattable instead
        else:
            ndim = len(self.arrays)
            for dat in data:
                pcube = Pcube(self, dat, error, verbose)
                savestats = []
                # For each fitted parameter, get desired stat interval indices from pcube.allstats
                # and get parameter value from its array
                for p in np.arange(ndim):
                    savestats.append(np.round(pcube.allstats[p],2))
                    if interval==5 or interval=='BI1':
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+8])],2))
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+9])],2))
                    elif interval==4 or interval=='BI2.5':
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+6])],2))
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+7])],2))
                    elif interval==3 or interval=='BI5':
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+4])],2))
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+5])],2))
                    elif interval==2 or interval=='CI67':
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+2])],2))
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+3])],2))
                    elif interval==1 or interval=='CI95':
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10])],2))
                        savestats.append(np.round(self.arrays[p][int(pcube.allstats[ndim*2+p*10+1])],2))
                # Add list of all desired stats to stattable
                stattable.append(savestats)

        stattable = np.array(stattable)  
        
        # If saving is requested, create a header string using the printparams formatting then save
        # using the default or supplied filename
        if save:
            header = ''
            for p in np.arange(len(self.arrays)):
                header += str(self.printparams[p])+', '
                header += str(self.printparams[p]+'_low, ')
                header += str(self.printparams[p]+'_up, ')
            np.savetxt(fname, stattable, header=header[:-2], delimiter=',', fmt='%.2f')
            
        return stattable

    
    def fitmap(self, dat, nprocs=None, linefixed=True, sigma_threshold=5, line_threshold=None, verbose=False, extraverbose=False, save=True, fname='FittingResults',  plotpeak=False, plotall=False, plotname='CornerPlot', fmt='png', scale='log', savePC=False, ClumpMask=np.array([None]), clumparray=np.array([None])):
        """Executes the fitting process on a MapData object
        
        Parameters
        ----------
        dat : MapData object
            MapData object containing data cubes for each emission line
        nprocs : int, optional
            number of parallel processors to use if using, default is None
        linefixed : bool, optional
            flag of whether the temperature and density should be held
            fixed for a single line of sight, default is True
        sigma_threshold : int, optional
            threshold in multiples of sigma for a data point to be 
            considered a detection, default is 5
        line_threshold : int, optional
            number of lines that need to be detected to initiate the fitting process
            default is one less than the number of parameters being fit
        verbose : bool, optional
            flag used to determine frequency of print statements, default is False
        extraverbose : bool, optional
            flag used to determine frequency of print statements, default is False
        save : bool, optional
            flag indicating if the results and plots if plotting should be saved, default is True
        fname : str, optional
            file name prefix to use if saving results, default is 'FittingResults'
        plotpeak : bool, optional
            flag of whether to save corner plots of the line peak fit, default is False
        plotall : bool, optional
            flag of whether to save corner plots of all fits, default is False
        plotname : str, optional
            file name prefix to use if saving plots, default is 'CornerPlot'
        fmt : str, optional
            format of plots if saving, default is 'png'
        scale : str, optional
            log or linear scaling of plotted probability distribution, default is 'log'
        savePC : bool, optional
            flag of whether to save output Pcube objects, default is False
        ClumpMask : array, optional
            array of clump assignments in the style of clumpfind, default is None
            if None, then no clump assignments are used
        clumparray : array, optional
            array of clump identifiers to execute fits for, default is None
        
        Returns
        -------
        finalarrays : array
            arrays of fitted parameters and their uncertainty ranges in the 2D shape
            of the input maps
        """
        
        # If no ClumpMask is given, create array in shape of dat that is all ones
        if not ClumpMask.any():
            ClumpMask = np.ones(dat.dshape)
        
        # If no clumparray is given, create array of all identifiers in the given or default ClumpMask
        if not np.array(clumparray).any():
            clumparray = np.unique(ClumpMask)
            
        # If no line_threshold is given, use one less than the number of fitted parameters, 
        # or the number of emission lines if that is smaller
        if not line_threshold:
            line_threshold = np.min([len(self.arrays) - 1, len(self.mols)])
            
        # Define looping function with functions in fittingloopfunc.py module    
        if linefixed:
            func = definefixedloop(self, dat.error, sigma_threshold=sigma_threshold, line_threshold=line_threshold, verbose=verbose, extraverbose=extraverbose, save=save, plotpeak=plotpeak, plotall=plotall, plotname=plotname, fmt=fmt, scale=scale, savePC=savePC)
        else:
            func = defineloop(self, dat.error, sigma_threshold=sigma_threshold, line_threshold=line_threshold, verbose=verbose, extraverbose=extraverbose, save=save, plotall=plotall, plotname=plotname, fmt=fmt, scale=scale, savePC=savePC)
        
        # Create empty lists to add final results for each pixel and each clump
        finalarrays = []
        finaldicts = []
        
        # Loop through clump array (if no clumps are being used, the only value in this array is 1)
        for ncl in clumparray:
            if np.isnan(ncl) or ncl==0:
                continue 
            
            # Also possible for users to input ClumpMask that is a mask using True and False
            # In this case, convert ncl=True to ncl=1, and skip over ncl=False
            if isinstance(ncl, bool):
                if ncl:
                    ncl=1
                else:
                    continue
                
            # Apply ClumpMask to data
            Data = np.where(ClumpMask==ncl, dat.data, np.nan)
        
            # Find data points above the sigma_threshold error for all emission lines and pixels
            maskerr = np.sum((Data.T> sigma_threshold*np.array(dat.error)).T, axis=0)
            # Find pixels where the number of lines detected exceeds the line_threshold
            maskline = (np.nanmax(maskerr, axis=0)>=line_threshold)
            # Find velocity channels where the number of lines detected exceeds the line_threshold
            maskvs = (np.nanmax(maskerr, axis=(1,2))>=line_threshold)
            
            try:
                # Get indices of intensities to include both after flattening ra and dec axes (Ipixs)
                # and also for original data shape for putting results back in the right spot of the map (respixs)
                Ipixs = np.argwhere(maskline.flatten())[:,0]
                respixs = np.argwhere(maskline)

                # Reshape data to collapse ra and dec axes, then grab only pixels with enough detections
                drsh = np.reshape(Data, (dat.dshape[0], dat.dshape[1], dat.dshape[2]*dat.dshape[3]))
                Ints = np.take(drsh, Ipixs, axis=-1).T

                # Find the range of velocity channels to include 
                velmin, velmax = np.argwhere(maskvs)[0][0], np.argwhere(maskvs)[-1][0]
                Ints = Ints[:,velmin:velmax]
                
            except:
                # If the above doesn't work, it's because no pixels in this clump have enough detections
                print('No detections in Clump ', ncl)
                continue

            # If nprocs are given, use that number of parallel processors to compute
            if nprocs:
                pool = Pool(nprocs)
                resdict = np.array(pool.map(func, enumerate(Ints)))
                pool.close()
                pool.join()
                resarray = [rd['FinalResults'] for rd in resdict]

            # If nprocs not given, do not do parallel processing and just do a for loop
            else:
                resdict = []
                resarray = []
                for i, Is in enumerate(Ints):
                    resdict.append(func((i, Is)))
                    resarray.append(resdict[-1]['FinalResults'])

            # Initiate results array full of nans for pixels with no detections
            finalarray = np.full((len(self.printparams)*3, dat.dshape[2]*dat.dshape[3]), np.nan)
            finaldicts = np.full((dat.dshape[2]*dat.dshape[3]), None)
            
            # Put fitted results into the correct flatted map indices using Ipixs
            for i, ipix in enumerate(Ipixs):
                finalarray[:, ipix] = resarray[i]
                finaldicts[ipix] = resdict[i]
            
            # Make results array the right shape again with ra and dec axes
            finalarray = np.reshape(finalarray, (len(self.printparams)*3, dat.dshape[2], dat.dshape[3]))
            finaldicts = np.reshape(finaldicts, (dat.dshape[2], dat.dshape[3]))
            
            # If multiple clumps are being used, print and save with clump identifier
            if np.nanmax(ClumpMask>1):
                print('Done computing fits for clump', ncl)
                if save:
                    pickle.dump([finalarray, resdict], open(fname+'_Clump'+str(int(ncl))+'.pkl', 'wb'))
            else:
                print('Done computing fits!')
                if save:
                    pickle.dump([finalarray, resdict], open(fname+'.pkl', 'wb'))
            
            # add final array to list finalarrays that includes all clumps
            finalarrays.append(finalarray)

        return np.array(finalarrays)
        
        

    
    
class Pcube:
    """Probability distribution array with inputs and statistics as attributes
    
    Attributes
    ----------
    mols : list
        list of strings indicating molecules of input emission lines
    js : list
        list of lower-level energy states of input emission lines
    bff : scalar, optional
        filling factor to use in fitting, default is 1.0
    Rs : list
        list of abundance ratio values used when ratios are fixed
    plotparams : list
        list of fitted parameters formatted for labels when plotting
    printparams : list
        list of fitted parameters formatted for printing
    arrays : list
        list of parameter arrays to be used in fitting
    Rshifts : list
        array of abundance ratios quantified as number of indices to shift column
    pcube : array
        probability array output from a compute_Pgrid* function
    allstats : array
        array of all calculated statistics
    stats : array
        array of preferred statistics, best constrained of CI95 and BI1
    statindices : array 
        array of preferred statistics as indices in parameter arrays
    
    
    Methods
    -------
    plot(model=np.array([None]), verbose=False, cmap='Greys', scale='linear', 
    plotintervals=[1,5], LTE=np.array([None]), title=None, save=False, 
    fname='CornerPlot', fmt='png')
        Runs PlotCorner function from basics module
    printstats(interval=None)
        Prints statistics results of probability cube
    """
    
    def __init__(self, RADEXFit, data, error, verbose=False, arrays=[[None]], Rshifts=[[None]], minmaxes=[None]):
        """
        If arrays, Rshifts, or minmaxes are not passed to the method, the corresponding 
        attributes of the RADEXFit object are used.
        
        Parameters
        ----------
        RADEXFit : RADEXFit object
            RADEXFit object with attributes about fitting process input
        data : list
            brightness temperature for each emission line of a single pixel
        error : list
            corresponding rms errors in the emission line data
        verbose : bool, optional
            flag used to determine frequency of print statements, default is False
        arrays : list, optional
            list of parameter arrays to be used in fitting process, default is None
        Rshifts : list, optional
            array of abundance ratios quantified as number of indices to shift column, default is None
        minmaxes : list, optional
            list of minimum and maximum temps, densities, and columns to consider, default is None
        """
        
        # Grab relevant attributes from input RADEXFit object
        self.mols = RADEXFit.mols
        self.js = RADEXFit.js
        self.bff = RADEXFit.bff
        self.Rs = RADEXFit.Rs
        self.plotparams = RADEXFit.plotparams
        self.printparams = RADEXFit.printparams
        
        # If minmaxes, arrays, and Rshifts are not given, use those from the RADEXFit object
        if not np.array(minmaxes).any():
            minmaxes = RADEXFit.minmaxes
        if not np.array(arrays[0]).any():
            arrays = RADEXFit.arrays
        try:
            if not np.array(Rshifts[0]).any():
                Rshifts = RADEXFit.Rshifts
        except:
            if not np.array(Rshifts).any():
                Rshifts = RADEXFit.Rshifts
            
        self.arrays = arrays
        self.Rshifts = Rshifts
        
        # Run the relevant compute_Pgrid* function from basics.py module
        if RADEXFit.mode=='default':
            self.pcube = basics.compute_Pgrid(self.mols, self.arrays, self.js, data, error, minmaxes, RADEXFit.Rindex, self.Rshifts, RADEXFit.Nshift0, RADEXFit.radexgrids, verbose=verbose, bff=RADEXFit.bff)
        elif RADEXFit.mode=='fitbff':
            self.pcube = basics.compute_Pgrid_bffit(self.mols, self.arrays, self.js, data, error, minmaxes, RADEXFit.Rindex, self.Rshifts, RADEXFit.Nshift0, RADEXFit.radexgrids, verbose=verbose)
        elif RADEXFit.mode=='fitTBratio':
            self.pcube = basics.compute_Pgrid_TBratiofit(self.mols, self.arrays, self.js, data, error, minmaxes, RADEXFit.Rindex, self.Rshifts, RADEXFit.Nshift0, RADEXFit.radexgrids, verbose=verbose, denomi=RADEXFit.denomi)
        elif RADEXFit.mode=='fitratio':
            self.pcube = basics.compute_Pgrid_ratiofit(self.mols, self.arrays, self.js, data, error, minmaxes, RADEXFit.Rindex, self.Rshifts, RADEXFit.Nshift0, RADEXFit.radexgrids, verbose=verbose, bff=RADEXFit.bff)
        elif RADEXFit.mode=='fitbffratio':
            self.pcube = basics.compute_Pgrid_bffratiofit(self.mols, self.arrays, self.js, data, error, minmaxes, RADEXFit.Rindex, self.Rshifts, RADEXFit.Nshift0, RADEXFit.radexgrids, verbose=verbose)
            
        # Run CalculateStats on resulting pcube from basics.py module
        self.allstats = basics.CalculateStats(self.pcube, self.arrays, verbose)
        
        # If calculate stats returned all nans, the other stats are also all nans
        ndim = len(self.arrays)
        if np.isnan(self.allstats).all():
            self.stats = np.full((ndim*3),(np.nan))
            self.statindices = np.full((ndim*3),(np.nan))
        
        else:
            # Initiate empty lists of preferred stats' values and their indices
            self.stats = []
            self.statindices = []

            # Preferred stat interval is the better constrained of CI95 and BI1
            # with the exception of when the column is higher than 1e17.5, when BI1 is usually wrong
            # (See Finn+21 for explanation of this choice)
            # (This preference could be checked and updated based on other data sets or emission line sets)
            for p in np.arange(ndim):
                pmax = np.round(self.allstats[p],2)
                if self.printparams[p]=='log_N' and pmax > 17.5:
                    self.stats.append(pmax)
                    self.stats.append(np.round(self.arrays[p][int(self.allstats[ndim*2+p*10])],2))
                    self.stats.append(np.round(self.arrays[p][int(self.allstats[ndim*2+p*10+1])],2))
                    self.statindices.append(int(self.allstats[ndim*2+p*10]))
                    self.statindices.append(int(self.allstats[ndim*2+p*10+1]))
                elif (self.allstats[ndim*2+p*10+9] - self.allstats[ndim*2+p*10+8]) < (self.allstats[ndim*2+p*10+1] - self.allstats[ndim*2+p*10]):
                    self.stats.append(pmax)
                    self.stats.append(np.round(self.arrays[p][int(self.allstats[ndim*2+p*10+8])],2))
                    self.stats.append(np.round(self.arrays[p][int(self.allstats[ndim*2+p*10+9])],2))
                    self.statindices.append(int(self.allstats[ndim*2+p*10+8]))
                    self.statindices.append(int(self.allstats[ndim*2+p*10+9]))
                else:
                    self.stats.append(pmax)
                    self.stats.append(np.round(self.arrays[p][int(self.allstats[ndim*2+p*10])],2))
                    self.stats.append(np.round(self.arrays[p][int(self.allstats[ndim*2+p*10+1])],2))
                    self.statindices.append(int(self.allstats[ndim*2+p*10]))
                    self.statindices.append(int(self.allstats[ndim*2+p*10+1]))
                
            
    def plot(self, model=np.array([None]), verbose=False, cmap='Greys', scale='linear', plotintervals=[1,5], LTE=np.array([None]), title=None, save=False, fname='CornerPlot', fmt='png'):
        """Plots corner plot of probability array
        
        Parameters
        ----------
        model : array, optional
            list of modeled 'true' parameters, default is None
        verbose : bool, optional
            flag used to determine frequency of print statements, default is False
        cmap : str, optional
            color map used in plotting 2D probability distributions, default is 'Greys'
        scale : str, optional
            whether to use linear or logarithmic scaling in all distribution plotting
        plotintervals : list, optional
            statistical intervals to plot, default is [1, 5], which are CI95 and BI1
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
        
        # Run PlotCorner function from basics.py module
        basics.PlotCorner(self.arrays, self.pcube, self.allstats, self.plotparams, model, verbose, cmap, scale, plotintervals, LTE, title, save, fname, fmt)
        
        return
            
        
    def printstats(self, interval=None):
        """Prints statistics results of probability cube
        
        Parameters
        ----------
        interval : bool, optional
            statistical interval to print, default is None
            if None, prints preferred statistical interval
        """
        
        # For each parameter, print out best-fit value and either preferred or specified stat interval
        for p, par in enumerate(self.printparams):
            print(par, ' = ', np.round(self.allstats[p],2))
            if not interval:
                print('Uncertainty Range: ', str(self.stats[p*3+1:p*3+3]))
            elif interval==5 or interval=='BI1':
                print('Bayesian 1.0 Interval: '+str(np.take(self.arrays[p], np.array(self.allstats[len(self.arrays)*2+p*10+8:len(self.arrays)*2+p*10+10],dtype=int))))
            elif interval==4 or interval=='BI2.5':
                print('Bayesian 2.5 Interval: '+str(np.take(self.arrays[p], np.array(self.allstats[len(self.arrays)*2+p*10+6:len(self.arrays)*2+p*10+8],dtype=int))))
            elif interval==3 or interval=='BI5':
                print('Bayesian 5.0 Interval: '+str(np.take(self.arrays[p], np.array(self.allstats[len(self.arrays)*2+p*10+4:len(self.arrays)*2+p*10+6],dtype=int))))
            elif interval==2 or interval=='CI67':
                print('67% Confidence Interval: '+str(np.take(self.arrays[p], np.array(self.allstats[len(self.arrays)*2+p*10+2:len(self.arrays)*2+p*10+4],dtype=int))))
            elif interval==1 or interval=='CI95':
                print('95% Confidence Interval: '+str(np.take(self.arrays[p], np.array(self.allstats[len(self.arrays)*2+p*10:len(self.arrays)*2+p*10+2],dtype=int))))
        
        return
    

class MapData:
    """Set of data cubes with attributes about observational metadata
    
    Attributes
    ----------
    mols : list
        list of strings indicating molecules of input emission lines
    js : list
        list of lower-level energy states of input emission lines
    data : array
        array of 3D data cubes for each emission line
    freqs : list
        rest frequency of each observed emission line
    error : list
        rms error in each emission line
    bmin : scalar
        beam minor axis in arcseconds, should match all data
    bmaj : scalar
        beam major axis in arcseconds, should match all data
    bpa : scalar
        beam position angle in degrees, should match all data
    arcsec_per_pixel : scalar
        arcseconds per pixel, should match all data
    dv : scalar
        velocity/frequency channel size in km/s, should match all data
    dnu : scalar
        velocity/frequency channel size in GHz
    dshape : tuple
        shape of data attribute
    """
    
    def __init__(self, mols, js, fnames, error=None, ralim=[0,None], declim=[0,None], vellim=[0,None]):
        """
        Parameters
        ----------
        mols : list
            list of strings indicating molecules of input emission lines
        js : list
            list of lower-level energy states of input emission lines
        fnames : list
            list of strings with file locations for each data set to load
        error : list, optional
            rms error in each emission line, default is None, which results 
            in the error being calculated for each line from loaded data cubes
        ralim : list, optional
            indices of min and max RA pixels to include, default is all
        declim : list, optional
            indices of min and max Dec pixels to include, default is all
        vellim : list, optional
            indices of min and max velocity channels to include, default is all
        """
        
        # Save emission line information from input
        self.mols = mols
        self.js = js
        
        # Initiate empty lists to add data and their rest frequencies to
        self.data = []
        self.freqs = []
        # If error is not input, initiate empty list to calculated it
        if not error:
            self.error = []
        
        for i, f in enumerate(fnames):
            # Loop through files and get the data and header, using ra, dec, and velocity ranges
            dat = fits.getdata(f)[...,vellim[0]:vellim[1],declim[0]:declim[1],ralim[0]:ralim[-1]]
            hdr = fits.getheader(f)
            # Somtimes data still has a stokes axis, if so get rid of it
            if np.ndim(dat)==4:
                dat = dat[0]
            
            # Get metadata about obs from fits header
            freq = hdr['RESTFRQ'] / 1e9
            bmin = np.round(hdr['bmin'] * 3600,3)
            bmaj = np.round(hdr['bmaj'] * 3600,3)
            bpa = np.round(hdr['bpa'], 3)
            arcsec_per_pixel = np.round(hdr['CDELT2']*3600,3)
            dnu = hdr['CDELT3'] / 1e9
            dv = np.round(np.abs(3e5 * (hdr['CDELT3']/hdr['CRVAL3'])),2)
            # Check that each subsequent dataset has the same overall shape, beam size, pixel size, and velocity channel size
            # If they don't, warn the user but continue anyways
            # Final values of these parameters in the MapData object will be those matching the last file opened
            if i>0:
                if bmin!=self.bmin or bmaj!=self.bmaj or arcsec_per_pixel!=self.arcsec_per_pixel or dv!=self.dv or dat.shape!=shape:
                    print('Data files should all have the same beam, pixel, and channel size and array shape before fitting')
            self.freqs.append(freq)
            self.bmin = bmin
            self.bmaj = bmaj
            self.bpa = bpa
            self.arcsec_per_pixel = arcsec_per_pixel
            self.dv = dv
            self.dnu = dnu
            shape = dat.shape
            # If the intensity units are not already in brightness temperature, convert data now
            # Assumes that if not in K, data is in Jy/beam
            if not hdr['BUNIT']=='K':
                print('Converting ', mols[i], js[i], ' from Jy/beam to brightness temperature')
                dat = basics.TB(dat, bmin, bmaj, freq)
            # If error isn't given, calculate it using median absolute deviation from astropy
            if not error:
                self.error.append(mad_std(dat[~np.isnan(dat)]))
            self.data.append(dat)
            
        self.data = np.array(self.data)
        self.dshape = self.data.shape
           
        return
        
        
def PlotFittedMaps(params, resarray, clumparray=[None], ploterror=True, errorthresholds=[None], MapData=None, contours=[None], contourmap=0, plotbeam=True, plotobsfootprint=True, plotcontour=True):
    """Plot output of the fitmap method into maps of best-fit parameters
    
    Parameters
    ----------
    resarray : array or str
        results array or a file location from RADEXFit.fitmap method
    clumparray : array, optional
        array of clump identifiers to plot, default is None
    ploterror : bool, optional
        flag of whether to plot maps of the error as well
    errorthresholds : array, optional
        fractional error that is required to consider fit good enough to plot
        default is 0.5 for temp, 2.0 for density, and 0.8 for column
    MapData : MapData object, optional
        MapData object to plot contours, footprints, or beam size, default is None
    contours : array, optional
        contour levels to plot for moment 0 map in fraction of the peak value
    contourmap : int, optional
        index of emission line which contours should be plotted for, default is 0 
    plotbeam : bool, optional
        flag of whether to plot beam size in lower left corner, default is True
    plotobsfootprint : bool, optional
        flag of whether to plot shared observation footprint, default is True
    plotcontour : bool, optional
        flag of whether to plot contours, default is True
    """
    
    # If errorthresholds are not given, add default values for fractional error limits of
    # 0.5, 2.0, and 0.8 for temp, density, and column, and 1.0 for any additional fitted params
    if not np.array(errorthresholds).any():
        errorthresholds = np.ones(len(params))
        errorthresholds[0] = 0.5
        errorthresholds[1] = 2.0
        errorthresholds[2] = 0.8
    
    # Check if the resarray has clumps it needs to combine
    if len(resarray)==1:
        # If given resarray is a string, load resarray from file
        if isinstance(resarray, str):
            resarray = pickle.load(open(resarray))[0]
        # If there are no clumps, drop the degenerative axis in resarray
        resarray = resarray[0]
        clumps = False
    else:
        clumps = True
        # If there are clumps but no clumparray is given, make a default one
        if not np.array(clumparray).any():
            clumparray = np.arange(len(resarray), dtype=int)

        # If given resarray is a string, load resarray from file for each clump
        if isinstance(resarray, str):
            maps = []
            for ncl in clumparray:
                maps.append(pickle.load(open(resarray+'_Clump'+str(int(ncl))+'.pkl', 'rb'))[0])
            maps = np.array(maps)
        else:
            maps = resarray

    for p, par in enumerate(params):
        # If no clumparray is given, don't need to combine clumps into one map
        if not clumps:
            # Get best-fit params, then calculate fractional error from upper and lower uncertainty range values
            pmap = resarray[3*p]
            # If parameter is density or column, switch to linear scaling to calculate fractional error
            if p==1 or p==2:
                errmap = np.sqrt(np.abs(10**resarray[3*p] - 10**resarray[3*p+1]) * np.abs(10**resarray[3*p+2] - 10**resarray[3*p])) / 10**pmap
            # All other parameters are already in linear scaling
            else:
                errmap = np.sqrt(np.abs(resarray[3*p] - resarray[3*p+1]) * np.abs(resarray[3*p+2] - resarray[3*p])) / pmap
            
            # Apply error thresholds based on fractional error map
            pmap = np.where(errmap < errorthresholds[p], pmap, np.nan)
            errmap = np.where(errmap < errorthresholds[p], errmap, np.nan)
            
        else:
            # Get column map first for weighting all other parameter averages between clumps where they overlap
            Nmaps = 10**maps[:,6] 
            # p==2 is always column parameter, need to sum over column instead of average it 
            # and also account for it being in log scaling
            if p==2:
                errmaps = np.sqrt(np.abs(Nmaps-10**maps[:,3*p+1]) * np.abs(10**maps[:,3*p+2]-Nmaps))
                # Apply error thresholds based on fractional error map before combining
                Nmaps = np.where(errmaps < errorthresholds[p]*Nmaps, Nmaps, np.nan)

                # Get final summed column map and its error then put back in log scaling
                pmap = np.log10(np.nansum(Nmaps, axis=0))
                lowmap = np.nansum(10**maps[:,7], axis=0)
                upmap = np.nansum(10**maps[:,8], axis=0)
                errmap = (np.sqrt(np.abs(10**pmap-lowmap) * np.abs(upmap-10**pmap)))/10**pmap
            # p==1 is always density parameter, need to account for it being in log scaling
            elif p==1:
                pmaps = 10**maps[:,3*p]
                errmaps = np.sqrt(np.abs(10**maps[:,3*p]-10**maps[:,3*p+1]) * np.abs(10**maps[:,3*p+2]-10**maps[:,3*p]))
                # Apply error thresholds based on fractional error map before combining
                pmaps = np.where(errmaps < errorthresholds[p]*pmaps, pmaps, np.nan)

                # Take column-weighted average of physical value then put best-fit map back in log scaling
                pmap = np.log10(np.nansum(Nmaps*pmaps, axis=0) / np.nansum(Nmaps, axis=0))
                lowmap = np.nansum(Nmaps*10**maps[:,4], axis=0) / np.nansum(Nmaps, axis=0)
                upmap = np.nansum(Nmaps*10**maps[:,5], axis=0) / np.nansum(Nmaps, axis=0)
                errmap = (np.sqrt(np.abs(10**pmap-lowmap) * np.abs(upmap-10**pmap)))/10**pmap
            # all other parameters are not in log scaling
            else:
                pmaps = maps[:,3*p]
                errmaps = np.sqrt(np.abs(maps[:,3*p]-maps[:,3*p+1]) * np.abs(maps[:,3*p+2]-maps[:,3*p]))
                # Apply error thresholds based on fractional error map before combining
                pmaps = np.where(errmaps < errorthresholds[p]*pmaps, pmaps, np.nan)

                # Take column-weighted average of physical value
                pmap = np.nansum(Nmaps*pmaps, axis=0) / np.nansum(Nmaps, axis=0)
                lowmap = np.nansum(Nmaps*maps[:,3*p+1], axis=0) / np.nansum(Nmaps, axis=0)
                upmap = np.nansum(Nmaps*maps[:,3*p+2], axis=0) / np.nansum(Nmaps, axis=0)
                errmap = (np.sqrt(np.abs(pmap-lowmap) * np.abs(upmap-pmap)))/pmap
                
        # Create new figure for each map
        f = plt.figure(figsize=(10,8))
        ax1 = plt.subplot(111)

        # If MapData object is given, plot contours, beam, or obs footprint
        if MapData:
            if plotcontour:
                # Get total intesity map (moment 0)
                mom0 = np.nansum(MapData.data[contourmap], axis=0)
                # If desired contours are not given, create contours at 25%, 50%, and 75% of peak
                if not np.array(contours).any():
                    contours = np.array([0.25,0.5,0.75])
                plt.contour(mom0, np.nanmax(mom0)*np.array(contours), linewidths=0.75, colors='k') 
            if plotbeam:
                # Beam is placed in the lower left corner of the map
                beam = patches.Ellipse((min(MapData.data[0,0].shape)*0.05,(min(MapData.data[0,0].shape)*0.05)), MapData.bmaj/MapData.arcsec_per_pixel, MapData.bmin/MapData.arcsec_per_pixel, MapData.bpa, fill=True, color='k', linewidth=0.5)
                ax1.add_artist(beam)
            
            if plotobsfootprint:
                # Common observation footprint of where all lines are observed and have non-nan data
                mom8s = np.nanmax(MapData.data, axis=1)
                masks = mom8s > 0.
                obsfootprint = np.prod(masks, axis=0) # 1s where all observations are non-nan, else 0s
                # A contour value of 0.5 draws line between 0s and 1s
                plt.contour(obsfootprint, [0.5], linewidth=0.5, colors='k', linestyles='--', alpha=0.5)

        # Get colorbar limits and also limits from error for scaling map opacity
        vmin = np.nanmin(pmap[pmap!=-np.inf])
        vmax = np.nanmax(pmap[pmap!=np.inf])
        emin = np.nanmin((errmap).flatten()[(errmap).flatten()!=-np.inf])
        emax = max([np.nanmax((errmap).flatten()[(errmap).flatten()!=np.inf]), 0.5])
        # Threshold value where larger errors start to have lower opacity
        ethr = 0.5*(emin+emax)

        plt.imshow(pmap,origin='lower',cmap='plasma',vmax=vmax,vmin=vmin)
        plt.colorbar(label=par) # Set the colorbar with the first plt.imshow without the opacities
        # Clear to white before showing the version with variable opacity
        plt.imshow(0*pmap,cmap='Greys',vmin=0,vmax=1)
        cmap=plt.cm.plasma
        # Create map of colors based on pmap
        cols = cmap(colors.Normalize(vmin, vmax, clip=True)(pmap))
        # Set the opacity variable of those colors to range between ethr and emax based on fractional error
        cols[...,-1]= 1 - colors.Normalize(ethr, emax, clip=True)(errmap)
        # Show new version of the map
        plt.imshow(cols,origin='lower')

        # Remove x and y axis ticks
        ax1.set_xticks([],'')
        ax1.set_yticks([],'')
        
        plt.show()

        if ploterror:
            f = plt.figure(figsize=(10,8))
            ax1 = plt.subplot(111)
            # Show fractional error map as a percentage map
            im = plt.imshow(errmap*100., origin='lower', cmap='plasma', vmin=emin*100., vmax=min([emax,errorthresholds[p]])*100.)
            
            # Add contours, beam, and/or observation foorprint as above
            if MapData:
                if plotcontour:
                    mom0 = np.nansum(MapData.data[contourmap], axis=0)
                    if not np.array(contours).any():
                        contours = np.array([0.25,0.5,0.75])
                    plt.contour(mom0, np.nanmax(mom0)*np.array(contours), linewidths=0.75, colors='k') 
                if plotbeam:
                    beam = patches.Ellipse((min(MapData.data[0,0].shape)*0.05,(min(MapData.data[0,0].shape)*0.05)), MapData.bmaj/MapData.arcsec_per_pixel, MapData.bmin/MapData.arcsec_per_pixel, MapData.bpa, fill=True, color='k', linewidth=0.5)
                    ax1.add_artist(beam)

                if plotobsfootprint:
                    mom8s = np.nanmax(MapData.data, axis=1)
                    masks = mom8s > 0.
                    obsfootprint = np.prod(masks, axis=0)
                    plt.contour(obsfootprint, [0.5], linewidth=0.5, colors='k', linestyles='--', alpha=0.5)
            
            # Remove x and y axis ticks
            ax1.set_xticks([],'')
            ax1.set_yticks([],'')
            plt.colorbar(im, label='% Error in '+par)
            plt.show()
    
    return


def CreateRadexGrids(fname, mols, Tarr, narr, Narr, dv, freqmin=0., freqmax=500., tbg=2.73, geometry=1, nprocs=None, verbose=False):
    """Function to create RADEX grids in the right format for fitting
    
    Parameters
    ----------
    fname : str
        name of file to save pickle, molecule name will be added
    mols : list
        list of strings of molecule names matching collision data files
        files must be in the same directory
    Tarr : array
        array of temperatures to include
    narr : array
        array of H2 densities to include, in log scale
    Narr : array
        array of emitter column densitites to include, in log scale
    dv : scalar
        linewidth of observations to be fit
    freqmin : scalar, optional
        minimum frequency to return in GHz, default is 0
    freqmax : scalar, optional
        maximum frequency to return in GHz, default is 500
    tbg : scalar, optional
        background temperatue in K, default is 2.73
    geometry : int, optional
        modeled RADEX geometry, default is 1, which is a sphere
        2 is LVG and 3 is slab
    nprocs : int, optional
        number of parallel processors to use in computation, default is None
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    """
    
    for mol in np.unique(mols):
        if verbose:
            print('Computing radex grid for molecule ', mol)
        # Get default parameters to fill in parameter dict
        parameters = radex.get_example_grid_parameters()
        
        # Replace default parameters with input or my own default values
        parameters['molfile'] = './'+mol+'.dat'
        parameters['tkin'] = Tarr
        # narr and Narr should be in log
        parameters['cdmol'] = 10**Narr
        parameters['h2'] = 10**narr
        parameters['linewidth'] = dv
        parameters['tbg'] = tbg
        parameters['fmin'] = freqmin
        parameters['fmax'] = freqmax
        parameters['geometry'] = geometry
        
        # If nprocs are given, run the grid creation with that many parallel processors
        if nprocs:
            pool = Pool(nprocs)
            grid = radex.run_grid(parameters, target_value='T_R (K)', pool=pool).to_numpy()
            pool.join()
            pool.close()
        else:
            grid = radex.run_grid(parameters, target_value='T_R (K)').to_numpy()
        
        # Reformat grid to have the right shape and axis order that later functions expect
        TRgrid = np.reshape(grid[:,3:], (len(narr), len(Tarr), len(Narr), len(grid[0])-3))
        TRgrid = np.moveaxis(TRgrid, 1, 0)
        if verbose:
            print('Grid shape:', TRgrid.shape)
        
        # Dump list of arrays
        pickle.dump([Tarr,narr,Narr,dv,TRgrid], open(str(mol)+'.'+fname+'.pkl', 'wb'))             

    return
    
    
    
    
    
    
    
    
    