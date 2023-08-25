"""Fitting Loop Funcs Module

This module contains functions that are used todefine the looping 
functions used in the RADEXFit.fitmap method. These functions
loop through the velocity channels along a line of sight to fit
each voxel and return a summed column, and column-weighted 
average temperature, density, filling factor, and/or abundance
ratios for an input pixel

This module requires that numpy is installed.
    
It contains the following functions:
    * definefixedloop : Fit parameters along a line of sight with
        all but the column fixed to the line peak uncertainty range
    * defineloop : Fit parameters along a line of sight with all
        parameters varying along the line of sight
"""

import numpy as np


def definefixedloop(RADEXFit, delts, sigma_threshold=5, line_threshold=None, verbose=False, extraverbose=False, save=True, plotpeak=False, plotall=False, plotname='CornerPlot', fmt='png', scale='log', savePC=False):
    """Fit params along a line of sight fixed to line peak uncertainty range
    
    Parameters
    ----------
    RADEXFit: RADEXFit object
        RADEXFit object with with attributes about the fitting process input
    delts : list
        corresponding rms errors in the emission line data
    sigma_threshold : int, optional
        threshold in multiples of sigma for a data point to be 
        considered a detection, default is 5
    line_threshold : int, optional
        number of lines that need to be detected to initiate the fitting process
        default is one less than the total number of emission lines
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    extraverbose : bool, optional
        flag used to determine frequency of print statements, default is False
    save : bool, optional
        flag indicating if the plots should be saved, default is True
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
    
    Returns
    -------
    fixedloopfunc : function
        Function that will fit params along a line of sight with all parameters
        except column fixed to line peak uncertainty range
    """
    
    # Make local instances of arrays and minmaxes and apply fixed filling factor if applicable
    Arrays = RADEXFit.arrays[:]
    Rshifts = RADEXFit.Rshifts[:]
    if RADEXFit.mode=='default' or RADEXFit.mode=='fitratio':
        Arrays[1] = Arrays[1]+np.log10(RADEXFit.bff)
        Arrays[2] = Arrays[2]+1.5*np.log10(RADEXFit.bff)
    cminmaxes = np.array(RADEXFit.minmaxes)
    
    if not line_threshold:
        line_threshold = np.min([len(Arrays) - 1, len(RADEXFit.mols)])
        
    verbose=True if extraverbose else verbose
    
    def fixedloopfunc(Ints_i):
        """Fit params along a line of sight fixed to line peak uncertainty range
        
        Parameters
        ----------
        Ints_i : tuple
            tuple of the flattened map index of the pixel and its list of 
            brightness temperatures to fit (formatted as from enumerate())
        """
        
        pix, Is = Ints_i
        
        resdict = {'ncomp': 0}
        resdict['FinalResults'] = np.full(len(RADEXFit.printparams)*3, np.nan)
        
        # make array length of vels with number of detected lines in each channel
        Imask = np.sum(Is > sigma_threshold * np.reshape(np.tile(delts, len(Is)), (len(Is), len(RADEXFit.mols))), axis=1)
        if not (Imask >= line_threshold).any():
            return resdict
        
        if verbose:
            print()
            print('Computing pixel ', pix)
            print()
            
        # initiate dictionaries, results array, and counters
        PCdict = {}
        ncomp = 0
        line = False

        # Find where lines are and separate different velocity components
        for v in np.arange(len(Is)):
            
            if Imask[v]<line_threshold:
                line=False

            elif Imask[v]>=line_threshold:
                if line==False:
                    ncomp += 1
                    resdict['indices_'+str(ncomp)] = []
                    resdict['ncomp'] = ncomp

                line = True
                resdict['indices_'+str(ncomp)].append(v)


        if verbose:
            print('Fitting', ncomp, 'velocity component(s)')
            print()
            
        # Find vmax in line profile or skip pixel if no vmax is found
        for comp in np.arange(1,ncomp+1):
            
            vlims = [resdict['indices_'+str(comp)][0], resdict['indices_'+str(comp)][-1]]
            
            vmax = None
            for i in np.arange(len(RADEXFit.mols))[::-1]:
                try:
                    vmax = np.nanargmax(Is[vlims[0]:vlims[1],i]) + vlims[0]
                except:
                    pass
            if not vmax:
                if verbose:
                    print('No vmax in pixel', i, ', component', comp)
                continue
            # Should tell user to put the most reliable line as the first line in each of the arrays (mols, js, data, etc), but also only after making sure it isn't hard coded that 13co is the second line.....
            
            results = np.empty((len(resdict['indices_'+str(comp)]), len(RADEXFit.printparams)*3))
            
            if True:
            #try:
                # First fit the line peak based on vmax above
                Vmax = Is[vmax,:]
                if verbose:
                    print('vmax=', vmax)
                    print('max ints: ', Vmax)
                    print()

                PC = RADEXFit.fit(Vmax, delts, verbose=extraverbose)

                if plotpeak:
                    PC.plot(verbose=extraverbose, scale=scale, fname=plotname+'_Peak_pixel'+str(pix)+'.'+fmt, save=save)

                finalresults = np.array(PC.stats)

                # Cut all parameter arrays to the fitted range of the line peak fit, update minmaxes array for T and n
                # (except for column, which should not be shared along the line)
                cut_Arrays = []
                cut_Rshifts = []
                for p in np.arange(len(RADEXFit.printparams)):
                    r=0
                    if RADEXFit.printparams[p][0]=='R':
                        arr = Arrays[p][PC.statindices[p*2]:PC.statindices[p*2+1]+1]
                        rshift = Rshifts[r][PC.statindices[p*2]:PC.statindices[p*2+1]+1]
                        cut_Rshifts.append(rshift)
                        r+=1
                    if not RADEXFit.printparams[p]=='log_N':
                        arr = Arrays[p][PC.statindices[p*2]:PC.statindices[p*2+1]+1]
                        if p<2:
                            cminmaxes[p*2:p*2+2] = PC.statindices[p*2:p*2+2] + np.array([0,1])
                    else:
                        arr = Arrays[p]
                    if verbose:
                        print()
                        print(RADEXFit.printparams[p], 'new array length: ', len(arr))
                    cut_Arrays.append(arr)
                
                # Step along line profile and fit the column density with constrained parameters above
                # Involves also only fitting where enough lines are detected 
                # and determining if there are multiple velocity components along the line of sight
                for i, v in enumerate(resdict['indices_'+str(comp)]):
                    Vs = Is[v]

                    if verbose:
                        print('v now=', v)
                        print('Ints: ', Vs)
                        
                    PC = RADEXFit.fit(Vs, delts, verbose=extraverbose, arrays = cut_Arrays, Rshifts=cut_Rshifts, minmaxes = cminmaxes)

                    if plotall:
                        PC.plot(verbose=extraverbose, scale=scale, fname=plotname+'_pixel'+str(pix)+'_velchan'+str(v)+'.'+fmt, save=save)

                    results[i] = PC.stats 
                    PCdict['PC'+str(v)] = PC.pcube
                    if verbose:
                        print('fitted values:', PC.stats)
                        print()

                # Get fitted N range along the line profile and sum for total N, Nlow, and Nup   
                Ns = 10**results[:,6:9] / RADEXFit.bff

                finalresults[6:9] = np.round(np.array(np.log10(np.nansum(Ns, axis=0))), 3)
                finalresults[3:6] = np.round(np.array(np.log10(10**finalresults[3:6]/RADEXFit.bff**1.5)), 3)
                
                if comp==1:
                    resdict['FinalResults'] = finalresults*1.
                else:
                    for p in np.arange(len(RADEXFit.printparams)):
                        if p!=2:
                            if p==1:
                                resdict['FinalResults'][3:6] = np.round(np.log10((10**resdict['FinalResults'][3:6]*10**resdict['FinalResults'][6:9] + 10**finalresults[3:6]*10**finalresults[6:9]) / (10**resdict['FinalResults'][6:9] + 10**finalresults[6:9])), 3)
                            else:
                                resdict['FinalResults'][3*p:3*p+3] = np.round((resdict['FinalResults'][3*p:3*p+3]*10**resdict['FinalResults'][6:9] + finalresults[3*p:3*p+3]*10**finalresults[6:9]) / (10**resdict['FinalResults'][6:9] + 10**finalresults[6:9]), 3)
                    resdict['FinalResults'][6:9] = np.round(np.log10(10**resdict['FinalResults'][6:9]+10**finalresults[6:9]), 3)
                    
                
                if verbose:
                    print('final stats array for velocity component', comp)
                    print(finalresults)
                    print()
                resdict['results_comp'+str(comp)] = finalresults

                if savePC:
                    print('Saving pixel ', i)
                    pickle.dump(PCdict, open('Pixel'+str(i)+'_PCdict.pkl', 'w'))

            else:
            #except Exception:
                print()
                print('ERROR WITH PIXEL ', pix, 'COMPONENT', comp)
                print()
            
        return resdict

    return fixedloopfunc



def defineloop(RADEXFit, delts, sigma_threshold=5, line_threshold=None, verbose=False, extraverbose=False, save=True, plotall=False, plotname='CornerPlot', fmt='png', scale='log', savePC=False):
    """Fit parameters along a line of sight with all parameters varying
    
    Parameters
    ----------
    RADEXFit: RADEXFit object
        RADEXFit object with with attributes about the fitting process input
    delts : list
        corresponding rms errors in the emission line data
    sigma_threshold : int, optional
        threshold in multiples of sigma for a data point to be 
        considered a detection, default is 5
    line_threshold : int, optional
        number of lines that need to be detected to initiate the fitting process
        default is one less than the total number of emission lines
    verbose : bool, optional
        flag used to determine frequency of print statements, default is False
    extraverbose : bool, optional
        flag used to determine frequency of print statements, default is False
    save : bool, optional
        flag indicating if the plots should be saved, default is True
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
    
    Returns
    -------
    loopfunc : function
        Function that will fit parameters along a line of sight with 
        all parameters varying
    """
    
    # Make local instances of arrays and minmaxes and apply fixed filling factor if applicable
    Arrays = RADEXFit.arrays[:]
    if RADEXFit.mode=='default' or RADEXFit.mode=='fitratio':
        Arrays[1] = Arrays[1]+np.log10(RADEXFit.bff)
        Arrays[2] = Arrays[2]+1.5*np.log10(RADEXFit.bff)
    cminmaxes = np.array(RADEXFit.minmaxes)
    
    if not line_threshold:
        line_threshold = np.min([len(Arrays) - 1, len(RADEXFit.mols)])
        
    verbose=True if extraverbose else verbose
    
    def loopfunc(Ints_i):
        """Fit params along a line of sight with all parameters varying
        
        Parameters
        ----------
        Ints_i : tuple
            tuple of the flattened map index of the pixel and its list of 
            brightness temperatures to fit (formatted as from enumerate())
        """
        
        pix, Is = Ints_i
        
        resdict = {'ncomp': 0}
        resdict['FinalResults'] = np.full(len(RADEXFit.printparams)*3, np.nan)
        
        # make array length of vels with number of detected lines in each channel
        Imask = np.sum(Is > sigma_threshold * np.reshape(np.tile(delts, len(Is)), (len(Is), len(RADEXFit.mols))), axis=1)
        if not (Imask >= line_threshold).any():
            return resdict
        
        if verbose:
            print()
            print('Computing pixel ', pix)
            print()
            
        # initiate dictionaries, results array, and counters
        PCdict = {}
        ncomp = 0
        line = False

        # Find where lines are and separate different velocity components
        for v in np.arange(len(Is)):
            
            if Imask[v]<line_threshold:
                line=False

            elif Imask[v]>=line_threshold:
                if line==False:
                    ncomp += 1
                    resdict['indices_'+str(ncomp)] = []
                    resdict['ncomp'] = ncomp

                line = True
                resdict['indices_'+str(ncomp)].append(v)

        if verbose:
            print('Fitting', ncomp, 'velocity component(s)')
            print()
            
        # Find vmax in line profile or skip pixel if no vmax is found
        for comp in np.arange(1,ncomp+1):
            vlims = [resdict['indices_'+str(comp)][0], resdict['indices_'+str(comp)][-1]]
            
            results = np.empty((len(resdict['indices_'+str(comp)]), len(RADEXFit.printparams)*3))
            
            #if True:
            try:

                # Step along line profile and fit the column density with constrained parameters above
                # Involves also only fitting where enough lines are detected 
                # and determining if there are multiple velocity components along the line of sight
                for i, v in enumerate(resdict['indices_'+str(comp)]):
                    Vs = Is[v]

                    if verbose:
                        print('v now=', v)
                        print('Ints: ', Vs)

                    PC = RADEXFit.fit(Vs, delts, verbose=extraverbose, arrays=Arrays)

                    if plotall:
                        PC.plot(verbose=extraverbose, scale=scale, fname=plotname+'_pixel'+str(pix)+'_velchan'+str(v)+'.'+fmt, save=save)

                    results[i] = PC.stats 
                    PCdict['PC'+str(v)] = PC.pcube
                    if verbose:
                        print('fitted values:', PC.stats)
                        print()

                finalresults = np.empty(len(RADEXFit.printparams)*3)
                
                # Get fitted N range along the line profile and sum for total N, Nlow, and Nup        
                Ns = 10**results[:,6:9] / RADEXFit.bff
                finalresults[6:9] = np.round(np.array(np.log10(np.nansum(Ns, axis=0))), 3)
                
                for p in np.arange(len(RADEXFit.printparams)):
                    if p!=2:
                        if p==1:
                            res = 10**results[:,3:6]/(RADEXFit.bff**1.5)
                            finalresults[3:6] = np.round(np.log10(np.nansum(Ns * res) / np.nansum(Ns)),3)
                            if comp>1:
                                resdict['FinalResults'][3:6] = np.round(np.log10((10**resdict['FinalResults'][3:6]*10**resdict['FinalResults'][6:9] + 10**finalresults[3:6]*10**finalresults[6:9]) / (10**resdict['FinalResults'][6:9] + 10**finalresults[6:9])), 3)
                        else:
                            res = results[:,3*p:3*p+3]
                            finalresults[3*p:3*p+3] = np.round(np.nansum(Ns * res) / np.nansum(Ns),3)
                            if comp>1:
                                resdict['FinalResults'][3*p:3*p+3] = np.round((resdict['FinalResults'][3*p:3*p+3]*10**resdict['FinalResults'][6:9] + finalresults[3*p:3*p+3]*10**finalresults[6:9]) / (10**resdict['FinalResults'][6:9] + 10**finalresults[6:9]), 3)

                
                if verbose:
                    print()
                    print('final stats array for velocity component', comp)
                    print(finalresults)
                    print()
                resdict['results_comp'+str(comp)] = finalresults
                
                if comp==1:
                    resdict['FinalResults'] = finalresults*1.
                else:
                    resdict['FinalResults'][6:9] = np.round(np.log10(10**resdict['FinalResults'][6:9]+10**finalresults[6:9]), 3)

                if savePC:
                    print('Saving pixel ', i)
                    pickle.dump(PCdict, open('Pixel'+str(i)+'_PCdict.pkl', 'w'))

            #else:
            except Exception:
                print()
                print('ERROR WITH PIXEL ', pix, 'COMPONENT', comp)
                print()
            
        return resdict

    return loopfunc



