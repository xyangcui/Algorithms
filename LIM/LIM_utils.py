
def LIM(xDat,lag): 
    '''
File Name    : LIM_analysis
Function Name: LIM  

Author       : Meg D. Fowler 
Date         : 8 Jan 2020 

Summary      : This function builds a linear inverse model (LIM) given a certain set of data. 
               That data should be supplied without an annual cycle. NOTE: there is no 
               check included here to make sure LIM is an appropriate choice for the dataset 
               supplied - that is left to the user. It is useful to check, for example, that
               the tau test is passed and that there are no Nyquist modes present. 

Inputs       : xDat - data to use in building LIM. Order should be [variables, time]. For example, 
                      if the goal is to use a year's worth of daily surface temperaure at 6 weather 
                      stations, the data should have dimensions of [6, 365]. 
               lag - the value supplied for Tau_0, the lag used in the lagged covariance matrix.  

Outputs      : b_alpha - values of Beta (not in a diagonal matrix as used in the calculation of L) 
               L       - the matrix of L values 
               Q       - the matrix of Q values 
               G       - the Green function 
               c0      - the contemporaneous covariance matrix 
               cT      - the lagged covariance matrix 
               normU   - the modes   of G, normalized 
               v       - the adjoints of G 
               g       - the eigenvalues of G 
               periods - the periods of oscillations in LIM 
               decayT  - the decay times of the periods above 
               * NOTE: These can be easily reduced (or expanded) by altering the return statement at the bottom of the script) 
''' 
    import numpy as np 
    from numpy import linalg as LA 

    # Take transpose of input data matrix 
    xDat_T = np.transpose(xDat) 

    # ------------------------------------------------------------------
    # STEP 1: Compute the lagged and contemporaneous covariance matrices 
    sizes = np.shape(xDat)    #Get size of matrix to determine how many data points and how many time records to consider 
    nDat = sizes[0]
    nT   = sizes[1]
  
    #Get the value of the data (xDat) at the specified lag to use in computing the lagged covariance matrix 
    xLagged = np.full([nDat,nT-lag],np.nan)  #Initialize matrix full of NaNs
    for iT in range(nT-lag):                 #Get the value of the data at the specified lag
        xLagged[:,iT] = xDat[:,iT+lag]

    # Initialize matrices full of NaNs 
    c0 = np.full([nDat, nDat], np.nan)    #Initialize matrix full of NaNs
    cT = np.full([nDat, nDat], np.nan)    #Initialize matrix full of NaNs
    
    # Compute covariance matrices for each data point 
    for iR in range(nDat):
        for iC in range(nDat):
            # Contemporaneous covariance matrix:
            c0[iR,iC] = np.nansum(xDat[iR,:]*xDat_T[:,iC]) / np.nansum(np.isfinite(xDat[iR,:]*xDat_T[:,iC]))
            # Lagged covariance matrix:
            cT[iR,iC] = np.nansum(xLagged[iR,:]*xDat_T[:-lag,iC]) / np.nansum(np.isfinite((xLagged[iR,:]*xDat_T[:-lag,iC])))     
    # --------------------------------------------------------------------
    # STEP 2: Compute the Green function, defining its eigen values and vectors 
    
    G = cT.dot(LA.inv(c0))    #The Green function is defined as the product between covariance matrices 

    # Define the modes (u) and eigen-values (g) of G
    g, u = LA.eig(G)

    iSort = g.argsort()[::-1]    #Sort the eigen values and vectors in order 
    g     = g[iSort]
    u     = u[:,iSort] 

    # Define the adjoints (v) based on the transpose of G 
    eigVal_T, v = LA.eig(np.transpose(G))
    iSortT      = eigVal_T.argsort()[::-1]
    eigVal_T    = eigVal_T[iSortT]
    v           = v[:,iSortT] 
   
    # But modes should ultimately be sorted by decreasing decay time (i.e., decreasing values of 1/beta.real) 

    # Compute Beta  
    b_tau   = np.log(g)
    b_alpha = b_tau/lag  #L's eignvalue

    # Sort data by decreasing decay time 
    sortVal = -1/b_alpha.real              #Decay time 

    iSort2 = sortVal.argsort()[::-1]      #Sorted indices 
    u      = u[:,iSort2]
    v      = v[:,iSort2]
    g      = g[iSort2]
    b_alpha = b_alpha[iSort2]

    # Make diagonal array of Beta (values should be negative)
    beta = np.zeros((nDat, nDat), complex)
    np.fill_diagonal(beta, b_alpha)

 
    #Need to normalize u so that u_transpose*v = identitity matrix, and u*v_transpose = identity matrix as well 
    normFactors = np.dot(np.transpose(u),v)
    normU       = np.dot(u,LA.inv(normFactors))

    # --------------------------------------------------------------------
    # STEP 3: Compute L and Q matrices 

    # Compute L matrix as normU * beta * v_transpose 
    L = np.dot(normU, np.dot(beta, np.transpose(v)))

    # Compute Q matrix 
    Q_negative = np.dot(L, c0) + np.dot(c0, np.transpose(L))
    Q = -Q_negative 

    # Also define the periods and decay times 
    #periods = (2 * np.pi) / b_alpha.imag 
    decayT    = -1 / b_alpha.real 
    frequency = b_alpha.imag / (2* np.pi)
    # --------------------------------------------------------------------
    # RETURN statement 
    return {'b_alpha':b_alpha, 'L':L, 'Q':Q, 'G':G, \
            'c0': c0, 'cT':cT, 'normU':normU, 'v':v, 'g':g, 'frequency':frequency, 'decayT':decayT }


def Nyquist_check(data,tau0):
    import numpy as np
    from numpy import linalg as LA

    if tau0<=5:
        lags = np.arange(1,tau0+4,1) # Test lags from 1 to tau0+3 (need to give end point as +4 to get proper value)
    else:
        lags = np.arange(tau0-4,tau0+4,1) # Test lags from tau0-3 to tau0+3
    
    # Check each lag in lags array to see if/where a Nyquist mode arises 
    for iLag in range(len(lags)): 
        output = LIM(data,lags[iLag]) #Compute LIM at this lag 
        #b_alpha_fnct,L_fnct,Q_fnct,G_fnct,c0_fnct,cT_fnct,u_fnct,v_fnct,g_fnct,periods_fnct,decayT_fnct = 
        b_alpha_fnct = output['b_alpha']
        nyq_check = b_alpha_fnct.imag*lags[iLag]    #Make array of beta_imaginary * lags, used to check for Nyquist mode 
    
        # Print out results of this check  for each lag 
        print('Tau_0 = ', lags[iLag])

        if (tau0==lags[iLag]): 
            print(' ** This is the tau0 you entered! **')

        if np.any((nyq_check>=3.141) & (nyq_check<=3.142))==True: 
            print('!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--')
            print('   WARNING: Nyquist mode encountered at this T0')
            print('!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--')  
            print('  Problem value of b_alpha.imag*T0 = ', nyq_check[np.where((nyq_check>=3.141) & (nyq_check<=3.142))])
        else: 
            print('No Nyquist mode here.')
        print()


def Q_test(Q,Q_plot="no"):
    '''
Q test - checks to make sure eigenvalues of Q are positive.
If Q_plot is set to "yes", a plot of the diagonal matrix of Q eigenvalues will be produced.

Inputs
    Q      - the Q matrix from LIM
    Q_plot - "yes" or "no" - whether or not to plot the diagonal matrix of Q eigenvalues
Outputs
    None - prints out whether or not eigenvalues of Q are positive, and optionally plots diagonal
              matrix of Q eigenvalues.
              
remember: If Q has negative eigenvalues, it does not mean the LIM is unusable. The negative values 
mainly for some reasons, including noise, nonlinearly and so on. 
You can remove the negative eigenvalues and referred eigenvectors. But it may affect the accuracy of LIM.
    
'''
    import numpy 
    from numpy import linalg as LA
    import matplotlib.pyplot as plt

    # Eigenanalysis of G 
    Q_eigval,Q_eigvec = LA.eig(Q)
    
    if numpy.any(Q_eigval<0)==True: 
        print('!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--')
        print('        WARNING: Eigenvalue of Q is negative! ')
        print('!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--!--') 
    else:
        print('Eigenvalue of Q is positive.')

    #plot diagonal matrix of Q
    if Q_plot=='yes':
        #Make diagonal matrix of g
        Q_diag = numpy.zeros((numpy.shape(Q)[0], numpy.shape(Q)[0]))
        numpy.fill_diagonal(Q_diag, Q_eigval.real)
        fig = plt.figure(1,figsize=(5, 5))
        ax  = fig.add_subplot()
        ax.set_xticks([]) 
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_title("Q_diag",fontsize=16)
        im = ax.pcolor(Q_diag, vmin=-0.3, vmax=0.3,cmap='bwr')
        cbar = fig.colorbar(
            im, 
            ax=ax,
            orientation='vertical',
            shrink=0.6,                
            aspect=20,                       
            )
        plt.show()

def tau_test1(data,tau0):
    '''
    tau_test: if diff tau' error_variance are close enough, 
                the tau test is passed.

    Inputs
        data  - initial state vector, numpy array
        tau0  - used tau0

    Outputs
        None - a plot of lines of error covariance.
    '''
    import numpy as np
    from numpy import linalg as LA
    import matplotlib.pyplot as plt
    from   matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
    import matplotlib.ticker as ticker

    # Error will be computed out to 31 days (assuming data is supplied in daily form)
    tau_arr     = np.arange(32)
    # Define expected error for range of lags defined in main part of Section 5 above. 
    if tau0<=5:
        lags = np.arange(1,tau0+5,1) # Test lags from 1 to tau0+3 (need to give end point as +4 to get proper value)
    else:
        lags = np.arange(tau0-4,tau0+4,1) # Test lags from tau0-3 to tau0+3

    all_epsilon = np.full([len(lags),len(tau_arr)],np.nan)  #Empty array to store error in 

    for iT0 in range(len(lags)):
        T0 = lags[iT0]
    
        #Carry out LIM with this value of lag
        output = LIM(data,T0) 
        c0_fnct = output['c0']; u_fnct = output['normU']; v_fnct = output['v']; g_fnct = output['g']
        #b_alpha_fnct,L_fnct,Q_fnct,G_fnct,c0_fnct,cT_fnct,u_fnct,v_fnct,g_fnct,periods_fnct,decayT_fnct = LIM(data,T0)

        #Make diagonal matrix of g
        g_diag = np.zeros((np.shape(data)[0], np.shape(data)[0]), complex)
        np.fill_diagonal(g_diag, g_fnct)
    
        # Compute the Green function for various values of Tau given T0
        epsilon_tau = np.full([len(tau_arr)],np.nan)
        for iT in range(len(tau_arr)):
            G_tau  = np.dot(u_fnct,np.dot((g_diag)**(tau_arr[iT]/T0),np.transpose(v_fnct))).real

            #Expected error 
            all_epsilon[iT0, iT] = (c0_fnct - np.dot(G_tau, np.dot(c0_fnct,np.transpose(G_tau)) ) ).trace()

    # Expected error on day 0, when conditions are known, is set to zero 
    new_epsilon        = np.full([len(lags),len(tau_arr)],np.nan)
    new_epsilon[:,0]   = 0                    
    new_epsilon[:,1::] = all_epsilon[:,1::]

    # Normalize by variance 
    norm_epsilon = new_epsilon/np.nansum(np.nanvar(data,axis=1))

    # Set up plot 
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12,8)

    # Plot each line for various choices of tau0 
    for iT0 in range(len(lags)):
        ax1.plot(tau_arr, norm_epsilon[iT0,:],label = r'$\tau_{0} = $'+str(lags[iT0]))

    ax1.set_xlabel('Lead time (days)',fontsize=16)
    ax1.set_ylabel('Normalized error variance',fontsize=16)
    ax1.legend(fontsize=14)
    ax1.set_title('Tau Test',fontsize=16)
    ax1.set_xlim([0,31])
    ax1.set_ylim([0,1])
    ax1.tick_params(labelsize=14)
    #ax1.grid()
    plt.show()

def tau_test2(data,tau0,yupper):
    '''
    tau_test: if diff_dims L's norms are close enough, 
                the tau test is passed.

    Inputs
        data  - initial state vector, numpy array
        tau0  - used tau0

    Outputs
        None - a plot of lines of error covariance.
    '''

    import numpy as np
    from numpy import linalg as LA
    import matplotlib.pyplot as plt
    from   matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
    import matplotlib.ticker as ticker

    lags = np.arange(1,21,1)

    all_epsilon = np.full([len(lags),len(data[:,0])],np.nan)  #Empty array to store error in 

    for iT0 in range(len(lags)):
        T0 = lags[iT0]
    
        #Carry out LIM with this value of lag
        output = LIM(data,T0) 
        L = output['L']

        #Calculate L2 norm.
        for i in np.arange(0,data.shape[0],1):
            L_sub = L[0:i,0:i]
            all_epsilon[iT0,i] = np.linalg.norm(L_sub)

    #all_epsilon = all_epsilon[all_epsilon[:, 1].argsort()]
    # Set up plot 
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12,8)

    # Plot each line for various choices of tau0 
    for iT0 in range(len(data[:,0])):
        ax1.plot(lags, all_epsilon[:,iT0],color="black")
    ax1.set_xlabel(r'$\tau_{0}$',fontsize=16)
    ax1.set_ylabel('Norm',fontsize=16)
    ax1.set_title('Tau test',fontsize=16)
    ax1.set_xlim([1,20])
    ax1.set_xticks(ticks=[1,2,4,6,8,10,12,14,16,18,20])
    ax1.set_ylim([0,yupper])
    ax1.tick_params(labelsize=14)
    #ax1.grid()
    plt.show()


def tau_test3(data,tau0):
    '''
    tau_test: if diff tau's L are close enough, 
                the tau test is passed.

    Inputs
        data  - initial state vector, numpy array
        tau0  - used tau0

    Outputs
        None - a plot of heatmaps of L.
    ''' 

    import numpy as np
    from numpy import linalg as LA
    import matplotlib.pyplot as plt
    from   matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
    import matplotlib.ticker as ticker
    from LIM_utils import LIM

    # Define expected error for range of lags defined in main part of Section 5 above. 
    if tau0<=5:
        lags = np.arange(1,tau0+5,1) # Test lags from 1 to tau0+3 (need to give end point as +4 to get proper value)
    else:
        lags = np.arange(tau0-4,tau0+4,1) # Test lags from tau0-3 to tau0+3

    L_list = []

    for iT0 in range(len(lags)):
        T0 = lags[iT0]
        #Carry out LIM with this value of lag
        output = LIM(data,T0) 
        L_list.append(output['L'].real)

    titles = [f"tau={i+1}" for i in range(6)]

    #set up plots
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    plt.rcParams.update({'font.size': 12})

    for i, ax in enumerate(axes.flat):
        im = ax.pcolor(
            L_list[i], 
            vmin=-0.3, 
            vmax=0.3,
            cmap='bwr',                                      
        )
        ax.set_title(titles[i], fontweight="bold")
        # Make plot look more like a table
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticks([])  # remove x tickers
        ax.set_yticks([])  # remove y tickers
     
    cbar = fig.colorbar(
        im, 
        ax=axes.ravel().tolist(),
        orientation='vertical',
        shrink=0.6,                
        aspect=20,                       
    )

    # keep image
    plt.show()

def tau_test4(x,L,c0):
    '''
    tau_test: if traces of observation and LIM are close enough, 
                the tau test is passed.

    Inputs
        x   - state vector, numpy array, [n,nt]
        L   - the matrix of L values, numpy array, [n,n]
        c0  - lag-0 covariance matrix, numpy array, [n,n]

    Outputs
        None - a plot of trace of lagged covariance matrix from observations and LIM.
    '''
    import numpy
    import matplotlib.pyplot as plt
    from scipy.linalg import expm

    Tau = numpy.arange(1,31,1)
    tr  = numpy.zeros([2,Tau.shape[0]])

    for i,tau in enumerate(Tau):
        #original
        x0 = numpy.copy(x[:,:-tau])
        xt = numpy.copy(x[:,tau:])
        ct = numpy.dot(xt, x0.T)/(x0.shape[1]-1)
        tr[0,i] = numpy.trace(ct)
        #LIM prediction
        F = numpy.dot(expm(L.real*tau),c0)
        tr[1,i] = numpy.trace(F)

    # Set up plot 
    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    # Plot each line for various choices of tau0 
    colors = ["red","blue"]
    labels = ["OBS-based","LIM-based"]
    for iT0 in range(2):
        ax.plot(Tau, tr[iT0,:],label=labels[iT0],color=colors[iT0])
        ax.set_xlabel(r'$\tau_{0}$',fontsize=16)
        ax.set_ylabel('Norm',fontsize=16)
        ax.set_title('tau_test',fontsize=16)
        ax.set_xlim([1,30])
        ax.set_xticks(ticks=[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30])
        ax.tick_params(labelsize=14)

    plt.legend()
    plt.show()   


def Error_test(datat,tau0,unit_days,split_num):
    #pass this test if the expected error is nearly identical with real error.
    #may add an extra AR1 error and persistent forecast.
    import numpy as np
    import matplotlib.pyplot as plt
    from LIM_utils import LIM

    #got expected error.  43years.
    nt = datat.shape[1]
    split_days = unit_days*split_num
    data = datat[:,0:split_days]

    output = LIM(data,tau0)
    u = output['normU']
    v = output['v']
    c0= output['c0']
    g_fnct = output['g']

    #Make diagonal matrix of g
    g_diag = np.zeros((np.shape(data)[0], np.shape(data)[0]), complex)
    np.fill_diagonal(g_diag, g_fnct)

    #Lags to determine error over 
    tau_arr = np.arange(35)
    all_epsilon = np.full([len(tau_arr)],np.nan)  #Empty array to store error in 

    #Make diagonal matrix of g
    g_diag = np.zeros((np.shape(data)[0], np.shape(data)[0]), complex)
    np.fill_diagonal(g_diag, g_fnct)

    # Compute the Green function for various values of Tau given tau0
    epsilon_tau = np.full([len(tau_arr)],np.nan)
    for iT in range(len(tau_arr)):
        G_tau   = np.dot(u,np.dot((g_diag)**(tau_arr[iT]/tau0),np.transpose(v))).real

        #Expected error 
        all_epsilon[iT] = (c0 - np.dot(G_tau, np.dot(c0,np.transpose(G_tau)) ) ).trace()
    # Expected error on day 0, when conditions are known, is set to zero 
    new_epsilon      = np.full([len(tau_arr)],np.nan)
    new_epsilon[0]   = 0                    
    new_epsilon[1::] = all_epsilon[1::]

    # Normalize by variance 
    expected_error = new_epsilon/np.nansum(np.nanvar(data,axis=1))


    # --- ERROR IN THE LIM FORECAST --- #
    dat = datat[:,split_days+1:]
    nDat = dat.shape[1]
    #   Create empty arrays to store values in 
    meanErr_fcst  = np.full([len(tau_arr),nDat],np.nan)  

    for iTau in range(len(tau_arr)):   #Loop over number of lags to be computed   

        G_tau  = np.dot(u,np.dot((g_diag)**(tau_arr[iTau]/tau0),np.transpose(v))).real

        for iT in range(nDat-tau_arr[iTau]):    #Loop over timesteps 
            # --- Get forecast data --- #
            #If there is any missing data at this time...
            if ( np.isnan(data[:,iT]).any() ): 
                #Indices where NaNs are *not*
                iNan = np.where(~np.isnan(dat[:,iT]))
                
                #Make replica matrix replacing NaNs with zeros so that it won't count for other stations (but is computed)
                replZeros          = np.zeros(np.shape(dat[:,iT]))
                replZeros[iNan[0]] = data[iNan[0],iT]
                
                #Get forecast using zeros array so that stations w/o NaN are still computed properly 
                fcst = np.dot(G_tau, replZeros)
            
                #Replacing stations that had missing data with a NaN (no forecast if no data)
                fcst = np.nan
            else:     
                #If no data is missing, just compute forecast as normal - G(Tau) * X(:,t)
                fcst = np.dot(G_tau, dat[:,iT])
            # -- Build up mean squared error 
            meanErr_fcst[iTau,iT] = np.linalg.norm(fcst-dat[:,iT+tau_arr[iTau]])**2
    
    # Set first error estimate to zero: 
    meanErr_fcst[0,:] = 0
    actual_error = np.nanmean(meanErr_fcst, axis=1)
    #Normalize by the variance at each station 
    actual_error  = actual_error/np.nansum(np.nanvar(dat,axis=1))

    # --- ERRORS THAT ARISE FROM A SIMPLE AR1 PROCESS --- #
    from statsmodels.tsa.api import VAR
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    #train a VAR model.
    model = VAR(data.T)
    var = model.fit(maxlags=1, ic='aic') 
    result = breaks_cusumolsresid(var.resid)
    #get lag days
    p   = var.k_ar

    meanErr_fcst  = np.full([len(tau_arr),nDat],np.nan)    
    for iT in np.arange(p-1,nDat-tau_arr[-1],1):    #Loop over timesteps 
        #if iT <=nDat-tau_arr[-1]
        #If no data is missing, just compute forecast as normal 
        fcst = var.forecast(dat[:,iT+1-p:iT+1].T,steps=tau_arr[-1])
        for iTau in np.arange(1,len(tau_arr),1):    
            # -- Build up mean squared error 
            meanErr_fcst[iTau,iT] = np.linalg.norm(fcst[iTau-1,:]-dat[:,iT+tau_arr[iTau]])**2
    
    # Set first error estimate to zero: 
    meanErr_fcst[0,:] = 0
    AR1pro_error = np.nanmean(meanErr_fcst, axis=1)
    #Normalize by the variance at each station 
    AR1pro_error  = AR1pro_error/np.nansum(np.nanvar(dat,axis=1))


    # --- ERRORS THAT ARISE FROM A SIMPLE Persistent forecast --- #
    meanErr_fcst  = np.full([len(tau_arr),nt],np.nan)  
    for iTau in range(len(tau_arr)):   #Loop over number of lags to be computed   
        for iT in range(nt-tau_arr[iTau]):    #Loop over timesteps 
            fcst = datat[:,iT]
            # -- Build up mean squared error 
            meanErr_fcst[iTau,iT] = np.linalg.norm(fcst-datat[:,iT+tau_arr[iTau]])**2
    
    # Set first error estimate to zero: 
    meanErr_fcst[0,:] = 0
    persis_error = np.nanmean(meanErr_fcst, axis=1)
    #Normalize by the variance at each station 
    persis_error  = persis_error/np.nansum(np.nanvar(datat,axis=1))


    # Set up plot 
    fig, ax1 = plt.subplots()
    fig.set_size_inches(12,8)
    #Plot expected error 
    ax1.plot(tau_arr, expected_error,color='green',linestyle='solid',label='Expected error')
    ax1.plot(tau_arr, actual_error,color='orange',linestyle='solid',label='LIM error')
    ax1.plot(tau_arr, AR1pro_error,color='blue',linestyle='solid',label='AR(1) error')
    ax1.plot(tau_arr, persis_error,color='red',linestyle='solid',label='Persis error')
    ax1.set_xlabel('Lead time (days)',fontsize=16)
    ax1.set_ylabel('Normalized error variance',fontsize=16)
    ax1.legend(fontsize=14)
    ax1.set_xlim([0,max(tau_arr)])
    ax1.set_ylim([0,2.5])
    ax1.tick_params(labelsize=14)
    ax1.grid(False)
    plt.show()


def optimal_state(g,u,v,N,tau,tau0):
    '''
    calculate optimal state and growth rate.(Breeden et al. 2020 MWR)
    Inputs
        g: eigenvalues of G.
        u: eigenvector of L or G.
        v: adjoint eivector of L or G.
        N: norm. the expected growth direction.
        tau: tau.
        tau0: tau used to train this LIM.

    Outputs
        p: optimal state.
        g_rate: growth rate.
    '''
    import numpy
    from numpy import linalg as LA

    g_diag = numpy.zeros((numpy.shape(N)[0], numpy.shape(N)[0]), complex)
    numpy.fill_diagonal(g_diag, g)
    G_tau = numpy.dot(u,numpy.dot((g_diag)**(tau/tau0),numpy.transpose(v)))
    # Define the modes (u) and eigen-values (g) of G
    g, u = LA.eig(G_tau.T@N@G_tau)
    iSort = g.argsort()[::-1]    #Sort the eigen values and vectors in order 
    g     = g[iSort]
    u     = u[:,iSort].real 

    # --------------------------------------------------------------------
    # RETURN statement
    p = u[:,0] #optimal state.
    g_rate = g[0]  #grwowth rate.
    return {'p':p, 'g_rate':g_rate}


def forecast(g,u,v,tau0,x0,tau):
    '''
    deterministic forecast by LIM.
    Inputs
        g: eigenvalues of G.
        u: eigenvector of L or G.
        v: adjoint eivector of L or G.
        tau: tau.
        tau0: tau used to train this LIM.
        x0: initial state.

    Outputs
        x: forecast state.
    '''

    import numpy as np
    from numpy import linalg as LA

    #Make diagonal matrix of g (again, if not done before)
    g_diag = np.zeros((np.shape(x0)[0], np.shape(x0)[0]), complex)
    np.fill_diagonal(g_diag, g)

    #Define Green's function at the peak of the max amplification growth
    G_tau = np.dot(u,np.dot((g_diag)**(tau/tau0*1.0),np.transpose(v)))
    #G_tau = G**(tau/tau0)
    return (np.dot(G_tau.real,x0))

def ens_fcst(L, Q, delta, X, freq='m'):
    '''
    LIM ensemble forecast.(Perkins and Hakim, 2021 P&P  Penland and Matrosova 1994 JC)
    Inputs
        delta - a time step. (hours for instance, delta=12 means 12 hours)
        X - state vector and Y is intermediate vector.
        L - a dynamical calculator.
        Q - error-martrix.
        freq - frequency used by your LIM (daily? d; monthly? m; yearly? y).

    firstly, we have
        Y(t+delta) = X(t) + sigma(L[:,j]*X[j]*delta) + sigma(q@sqrt[yi*delta])
    then 
        X(t+delta/2) = (Y(t+delta) + X(t))/2   
    '''
    import numpy
    from numpy import linalg as LA
    from numpy import random
    # EVD for Q.
    g, q = LA.eig(Q)

    # which delta.
    if freq == 'd':
        delt = delta/24
        n = 2 * 24//delta
    elif freq == 'm':
        delt = delta/720
        n = 2 * 720//delta
    else:
        delt = delta/8640
        n = 2 * 8640//delta

    for t in range(n):
        Y = X
        # deterministic 
        term1 = numpy.zeros(Y.shape[0])
        for j in range(L.shape[1]):
            term1 += L[:, j].real * Y[j] * delt

        # stochastic
        term2 = numpy.zeros(Y.shape[0])
        R = random.normal(0, 1, Y.shape[0])
        for i in range(q.shape[1]):
            term2 += q[:, i].real * numpy.sqrt(g[i].real * delt) * R[i].real
        # temporary
        Y = X + term1 + term2
        # final value for half-step forward.
        X = (X + Y) * 0.5

    # --------------------------------------------------------------------
    # RETURN statement 
    alpha = int(2 / delt)
    return X[:, 0::alpha]

def long_run(L, Q, delta, X, freq='m'):
    '''
    Linear Inverse Modeling long-time run.(Henderson et al. 2020 JC  Penland and Matrosova 1994 JC)
    Free run.
    Inputs
        delta - a time step. (integrated delta; is not the tau0 used to train LIM)
        X - state vector and Y is intermediate vector.
        L - a dynamical calculator.
        q - error-martrix eig vector with yi is eig value.

    firstly, we have
        Y(t+delta) = Y(t) + sigma(L[:,j]*Y[j]*delta) + sigma(q@sqrt[yi*delta])
    then 
        X(t+delta./2) = (Y(t+delta) + Y(t))/2   
    '''
    import numpy
    from numpy import linalg as LA
    from numpy import random
    # EVD for Q.
    g, q = LA.eig(Q)

    # which delta.
    if freq == 'd':
        delt = delta/24
        n = 2 * 24//delta
    elif freq == 'm':
        delt = delta/720
        n = 2 * 720//delta
    else:
        delt = delta/8640
        n = 2 * 8640//delta

    X = numpy.zeros((L.shape[0], n))
    Y = X[:,0]
    for t in range(n):
        term1 = numpy.zeros(Y.shape[0])
        for j in range(L.shape[1]):
            term1 += L[:, j] * Y[j] * delt

        term2 = numpy.zeros(Y.shape[0])
        R = random.normal(0, 1, Y.shape[0])
        for i in range(q.shape[1]):
            term2 += q[:, i] * numpy.sqrt(g[i] * delt) * R[i]

        Y = X + term1 + term2
        # final value for half-step forward.
        X = (X + Y) * 0.5

    # --------------------------------------------------------------------
    # RETURN statement
    alpha = int(2 / delt)
    return X[:, 0::alpha]

def dynamical_filter(X,v,u):
    '''
      Description: a method likes POP
      by using eginvectors of L, x can be decomposed as x = sigma(u*alpha)
      ref:
        Penland, C., and L. Matrosova, 2006: Studies of El Niño and Interdecadal Variability in Tropical Sea Surface Temperatures Using a Nonnormal Filter. 
                Journal of Climate, 19, 5796–5815, https://doi.org/10.1175/JCLI3951.1.
      Clac: alpha = v.T @ x
      Input:
        X: LIM state [space,time]
        v: selected eignvectors of adjoint L. [space, number]
      Output:
        RC: reconstructed state vectors. [space,time]

    '''
    return u @ (v.T @ X)
