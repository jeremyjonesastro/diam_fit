import warnings
warnings.simplefilter(action='ignore')

from pprint import pprint
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat
from progressbar import progressbar
import seaborn as sns
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.special import jn,gamma
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import random as rnd
import math
import itertools as it
import argparse

"""
Fits a diameter and limb-darkening coefficient to visibilities and/or closure amplitudes. 
Linear or power law limb darkening laws are both available. 
When fitting to visibilities, fits a scaling factor for each night+combiner. Uncertainties calculated by a block-bootstrap method.

Required Inputs
--data_dir - the relative path to the directory with the reduced+calibrated oifits that you want to use are located
--stars - a star or list of stars to fit
--diam_init - the initial guess for the diameter
--ldcK_init - the initial guess for the K-band limb-darkening coefficient
--ldcH_init - the initial guess for the H-band limb-darkening coefficient
--combiners - a beam combiner (or list of combiners) your data come from
--nights - the night or list of nights your data come from

***IMPORTANT NOTE - Your oifits files must be named in the following format to be recognized by diam_fit:
    STAR.COMBINER.NIGHT*.oifits
    Examples:
    altair.MIRCX.2026Jun16.oifits
    vega.MYSTIC.2025Oct03.01.oifits
    vega.MYSTIC.2025Oct03.02.oifits
    HD_27371.PIONIER.2027Jul01.extra_label.oifits
    
    When you call diam_fit, the star, combiner, and nights must match the values in the filenames or else the files won't be read

Optional Inputs
--scenarios - accepted values are ['by_file','by_night','by_combiner','all'], 'all' is default. Multiple options allowed
--law - accepted values are ['power','linear'], 'power' is default
--boot_N - number of iterations for the bootstrapping. Default is 1000
--fix_ldcK - fix ldcK to ldcK_init (default = false)
--fix_ldcH - fix ldcH to ldcH_init (default = false)
--camp - fit closure amplitudes (default = false)

"""

#Example runs
# Multistar
#  python3 diam_fit.py --data_dir final_data_2023Dec --stars HD_27371 HD_27697 HD_28305 HD_28307 --diam_init 1.5 1.5 1.5 1.5 --ldcK_init 0.3 0.3 0.3 0.3 --ldcH_init 0.3 0.3 0.3 0.3 --combiners MIRCX MYSTIC --nights 2023Dec12 2023Dec13
# Single star
#  python3 diam_fit.py --data_dir final_data_2023Dec --stars HD_27371 --diam_init 1.5 --ldcK_init 0.3 --ldcH_init 0.3 --combiners MIRCX MYSTIC --nights 2023Dec12 2023Dec13
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=ascii)
    parser.add_argument('--stars',nargs='+')
    parser.add_argument('--diam_init',nargs='+',type=float)
    parser.add_argument('--ldcK_init',nargs='+',type=float)
    parser.add_argument('--ldcH_init',nargs='+',type=float)
    parser.add_argument('--combiners',nargs='+')
    parser.add_argument('--nights',nargs='+')
    parser.add_argument('--scenarios',nargs='+',default=['all'])
    parser.add_argument('--law',type=ascii, default='power')
    parser.add_argument('--boot_N',type=int,default=1000)
    parser.add_argument('--fix_ldcK',action='store_true')
    parser.add_argument('--fix_ldcH',action='store_true')
    parser.add_argument('--camp',action='store_true')   #Closure amplitudes
    
    args = parser.parse_args()
    
    args.data_dir = args.data_dir.replace("'","")   #clean up the incoming strings
    args.law = args.law.replace("'","")
    
    path = Path().resolve()    #Set path so the code knows where we are
    save_dir = path / args.data_dir / 'results'  #Where the results are to be saved
    if not save_dir.exists():   #If results directory doesn't exist, make it
        save_dir.mkdir(parents=True)
    
    data_dir = path / args.data_dir  #Where the data are stored
    l1_data_dir = path / args.data_dir / 'l1_data'
    for i,star in enumerate(args.stars): #Loop through each star
        print('====================================================================')
        print(star)
        for s in args.scenarios:
            file_groups,notes = select_files(s,data_dir,star,args.nights,args.combiners)
            l1_file_groups,l1_notes = select_files(s,l1_data_dir,star,args.nights,args.combiners)
            
            for j,fg in enumerate(file_groups):
                note = notes[j]+'.'+args.law
                
                v2df = make_v2df(fg,data_dir)
                plot_sf = range(int(min(v2df['sf'])),int(max(v2df['sf'])),1)
                popt, pcov, groups = fit_v2(v2df,args.law,args.diam_init[i],args.ldcK_init[i],args.ldcH_init[i],args.fix_ldcK,args.fix_ldcH)
                report_fit_results(v2df,args.law,popt,groups,save_dir,s,note)
                
                
                if args.camp:
                    cadf = make_cadf(v2df)
                    
                    try:
                        capopt = fit_ca(cadf,args.law,args.diam_init[i],args.ldcK_init[i],args.ldcH_init[i],args.fix_ldcK,args.fix_ldcH,maxiter=100000)
                        report_ca_fit_results(cadf,args.law,capopt,save_dir,s,note)
                    except:
                        print('CA fit failed for {}'.format(star))
                        
                    l1_v2df = make_v2df(l1_file_groups[j],data_dir)
                    l1_cadf = make_cadf(l1_v2df)
                    try:
                        cal1popt = fit_ca(l1_cadf,args.law,args.diam_init[i],args.ldcK_init[i],args.ldcH_init[i],args.fix_ldcK,args.fix_ldcH,maxiter=100000)
                        report_ca_fit_results(l1_cadf,args.law,cal1popt,save_dir,s,note+'.L1')
                    except:
                        print('L1 CA fit failed for {}'.format(star))
                    
                    do_bootstrap_by_obs(v2df,args.law,save_dir,s,note,args.diam_init[i],args.ldcK_init[i],args.ldcH_init[i],popt,groups,args.boot_N,plot_sf,args.fix_ldcK,args.fix_ldcH,ca=True,capopt=capopt,cal1popt=cal1popt,cadf=cadf,l1_v2df=l1_v2df,l1_cadf=l1_cadf)
                else:
                    do_bootstrap_by_obs(v2df,args.law,save_dir,s,note,args.diam_init[i],args.ldcK_init[i],args.ldcH_init[i],popt,groups,args.boot_N,plot_sf,args.fix_ldcK,args.fix_ldcH)

def make_cadf(v2df):
    #Makes a dataframe of closure amplitudes based on the V^2
    cadf = pd.DataFrame(columns=['CA','CAerr','lnCA','lnCAerr','wl','dwl','ucoords','vcoords','sflist','sfmax','quad','bls','MJD','night','obs','quad_obs'])
    
    uni_obs = v2df['obs'].unique()  #Unique observations
    
    for this_obs in uni_obs:
        obs_v2df = v2df[v2df.obs == this_obs]           #Dataframe of V^2 for this observation
        uni_bls = obs_v2df['bl'].unique()
        uni_wls = obs_v2df['wl'].unique()
        scopes,N_scopes = get_scopes_for_ca(uni_bls)    #Which scopes are in this obs
        if N_scopes >= 4:    #No CAs for <4 scopes
            quads = [list(q) for q in it.combinations(scopes,4)]    #List of the different quadrangles
            for quad in quads:
                quad = sorted(quad)
                quad_bls = [list(bl) for bl in it.combinations(quad,2)]
                quad_str_bls = ['{} {}'.format(bl[0],bl[1]) for bl in quad_bls]
                quad_str_bls_rev = ['{} {}'.format(bl[1],bl[0]) for bl in quad_bls]
                bl_options = quad_str_bls + quad_str_bls_rev
                quad_v2df = obs_v2df[obs_v2df.bl.isin(bl_options)]
                quad_v2df = quad_v2df.sort_values(by=['wl','bl','MJD'])
                cadf = calc_CA(quad,quad_v2df,cadf)
    
    return cadf

def calc_CA(quad,v2df,cadf):
    #Calculate the CAs and add them to the df
    bls = v2df.bl.unique()
    
    #Get the visibilities
    V_12,V_12err,wl_12,dwl_12,uwl_12,vwl_12,MJD_12,night_12,obs_12 = get_Vxy(quad[0],quad[1],v2df)
    V_13,V_13err,wl_13,dwl_13,uwl_13,vwl_13,MJD_13,night_13,obs_13 = get_Vxy(quad[0],quad[2],v2df)
    V_14,V_14err,wl_14,dwl_14,uwl_14,vwl_14,MJD_14,night_14,obs_14 = get_Vxy(quad[0],quad[3],v2df)
    V_23,V_23err,wl_23,dwl_23,uwl_23,vwl_23,MJD_23,night_23,obs_23 = get_Vxy(quad[1],quad[2],v2df)
    V_24,V_24err,wl_24,dwl_24,uwl_24,vwl_24,MJD_24,night_24,obs_24 = get_Vxy(quad[1],quad[3],v2df)
    V_34,V_34err,wl_34,dwl_34,uwl_34,vwl_34,MJD_34,night_34,obs_34 = get_Vxy(quad[2],quad[3],v2df)
    
    data_amt_a = [len(V_12),len(V_34),len(V_13),len(V_24)]
    data_amt_b = [len(V_13),len(V_24),len(V_14),len(V_23)]
    
    a_good = all([l == data_amt_a[0] for l in data_amt_a])
    b_good = all([l == data_amt_b[0] for l in data_amt_b])
    
    if data_amt_a[0] == 0:
        a_good = False
    if data_amt_b[0] == 0:
        b_good = False
    
    #Calculate spatial frequency (in M-lambda)
    sf_12 = np.sqrt(uwl_12**2+vwl_12**2)/1e6
    sf_13 = np.sqrt(uwl_13**2+vwl_13**2)/1e6
    sf_14 = np.sqrt(uwl_14**2+vwl_14**2)/1e6
    sf_23 = np.sqrt(uwl_23**2+vwl_23**2)/1e6
    sf_24 = np.sqrt(uwl_24**2+vwl_24**2)/1e6
    sf_34 = np.sqrt(uwl_34**2+vwl_34**2)/1e6
    
    #Closure Amplitudes
    if a_good:
        CA_a = (V_12*V_34)/(V_13*V_24)
    if b_good:
        CA_b = (V_13*V_24)/(V_14*V_23)
    #CA_c = (V_14*V_23)/(V_12*V_34)  #Redundant with a and b
    
    #partial derivatives
    if a_good:
        dCAadV_12 = (V_34)/(V_13*V_24)
        dCAadV_34 = (V_12)/(V_13*V_24)
        dCAadV_13 = (V_12*V_34)/(V_13**2*V_34)
        dCAadV_24 = (V_12*V_34)/(V_13*V_34**2)
    if b_good:
        dCAbdV_13 = (V_24)/(V_14*V_23)
        dCAbdV_24 = (V_13)/(V_14*V_23)
        dCAbdV_14 = (V_13*V_24)/(V_14**2*V_23)
        dCAbdV_23 = (V_13*V_24)/(V_14*V_23**2)
    #dCAcdV_14 = (V_23)/(V_12*V_34)
    #dCAcdV_23 = (V_14)/(V_12*V_34)
    #dCAcdV_12 = (V_14*V_23)/(V_12**2*V_34)
    #dCAcdV_34 = (V_14*V_23)/(V_12*V_34**2)
    
    #Closure Amplitude Errors
    if a_good:
        CA_a_err = np.sqrt((dCAadV_12*V_12err)**2+(dCAadV_34*V_34err)**2+(dCAadV_13*V_13err)**2+(dCAadV_24*V_24err)**2)
    if b_good:
        CA_b_err = np.sqrt((dCAbdV_13*V_13err)**2+(dCAbdV_24*V_24err)**2+(dCAbdV_14*V_14err)**2+(dCAbdV_23*V_23err)**2)
    #CA_c_err = np.sqrt((dCAcdV_14*V_14err)**2+(dCAcdV_23*V_23err)**2+(dCAcdV_12*V_12err)**2+(dCAcdV_34*V_34err)**2)
    
    #Create df for the first closure amplitude
    if a_good:
        adf = pd.DataFrame(columns=['CA','CAerr','lnCA','lnCAerr','wl','dwl','ucoords','vcoords','sflist','sfmax','quad','bls','MJD','night','obs','quad_obs'])
        adf['CA'] = CA_a
        adf['CAerr'] = CA_a_err
        adf['lnCA'] = np.log(np.absolute(CA_a))
        adf['lnCAerr'] = CA_a_err/CA_a
        adf['wl'] = wl_12
        adf['dwl'] = dwl_12
        adf['ucoords'] = list(map(list,zip(uwl_12,uwl_34,uwl_13,uwl_24)))
        adf['vcoords'] = list(map(list,zip(vwl_12,vwl_34,vwl_13,vwl_24)))
        adf['sflist'] = list(map(list,zip(sf_12,sf_34,sf_13,sf_24)))
        adf['sfmax'] = get_max_sf(sf_12,sf_34,sf_13,sf_24)
        adf['quad'] = ' '.join(quad)
        adf['bls'] = [['{} {}'.format(quad[0],quad[1]),'{} {}'.format(quad[2],quad[3]),'{} {}'.format(quad[0],quad[2]),'{} {}'.format(quad[1],quad[3])]] * len(adf)
        adf['MJD'] = MJD_12
        adf['night'] = night_12
        adf['obs'] = obs_12
        adf['quad_obs'] = ' '.join(quad)+' {}'.format(obs_12[0])
    
    #Create df for the second closure amplitude
    if b_good: #Make sure all the data has the same length
        bdf = pd.DataFrame(columns=['CA','CAerr','lnCA','lnCAerr','wl','dwl','ucoords','vcoords','sflist','sfmax','quad','bls','MJD','night','obs','quad_obs'])
        bdf['CA'] = CA_b
        bdf['CAerr'] = CA_b_err
        bdf['lnCA'] = np.log(np.absolute(CA_b))
        bdf['lnCAerr'] = CA_b_err/CA_b
        bdf['wl'] = wl_13
        bdf['dwl'] = dwl_13
        bdf['ucoords'] = list(map(list,zip(uwl_13,uwl_24,uwl_14,uwl_23)))
        bdf['vcoords'] = list(map(list,zip(vwl_13,vwl_24,vwl_14,vwl_23)))
        bdf['sflist'] = list(map(list,zip(sf_13,sf_24,sf_14,sf_23)))
        bdf['sfmax'] = get_max_sf(sf_13,sf_24,sf_14,sf_23)
        bdf['quad'] = ' '.join(quad)
        bdf['bls'] = [['{} {}'.format(quad[0],quad[2]),'{} {}'.format(quad[1],quad[3]),'{} {}'.format(quad[0],quad[3]),'{} {}'.format(quad[1],quad[2])]] * len(bdf)
        bdf['MJD'] = MJD_13
        bdf['night'] = night_13
        bdf['obs'] = obs_13
        bdf['quad_obs'] = ' '.join(quad)+' {}'.format(obs_13[0])
    
    if a_good:
        cadf = pd.concat([cadf,adf])
    if b_good:
        cadf = pd.concat([cadf,bdf])
    return cadf

def get_max_sf(a,b,c,d):
    #a,b,c, and d are 1-d arrays. This generates a 1-d array of the same length that is the maximum of a[i],b[i],c[i],d[i]
    sfmax = [max([a[i],b[i],c[i],d[i]]) for i,v in enumerate(a)]
    return sfmax

def get_Vxy(s1,s2,v2df):
    V = []
    Verr = []
    wl = []
    dwl = []
    u_wl = []
    v_wl = []
    MJD = []
    night = []
    obs = []
    
    bl = '{} {}'.format(s1,s2)
    bl_v2df = v2df[v2df.bl == bl]
    uni_wl = bl_v2df.wl.unique()
    
    for this_wl in uni_wl:
        wl_v2df = bl_v2df[bl_v2df.wl == this_wl]
        V2_arr = np.array(wl_v2df.V2)
        V2err_arr = np.array(wl_v2df.V2err)
        weights = 1/V2err_arr**2
        V2 = np.sum(V2_arr * weights) / np.sum(weights)
        V2err = np.sqrt(1 / np.sum(weights))
        #this_V = np.sqrt(np.absolute(V2))
        this_V = np.sqrt(0.5*(V2 + np.sqrt(V2**2 + 2 * V2err**2)))   #Based on Eq 3.98a of Sivia's Data Analysis: A Bayesian Tutorial, Second Edition
        #this_Verr = V2err/(2*this_V)
        this_Verr = 1/np.sqrt((1/this_V**2) + (2*(3*this_V**2 - V2))/(V2err**2))   #Based on Eq 3.98b of Sivia's Data Analysis: A Bayesian Tutorial, Second Edition
        V.append(this_V)
        Verr.append(this_Verr)
        wl.append(this_wl)
        dwl.append(np.average(wl_v2df.dwl))
        u_wl.append(np.average(wl_v2df.u_wl))
        v_wl.append(np.average(wl_v2df.v_wl))
        MJD.append(np.average(wl_v2df.MJD))
        night.append(np.average(wl_v2df.night))
        obs.append(wl_v2df.obs.iloc[0])
        
    V = np.array(V)
    Verr = np.array(Verr)
    wl = np.array(wl)
    dwl = np.array(dwl)
    u_wl = np.array(u_wl)
    v_wl = np.array(v_wl)
    MJD = np.array(MJD)
    night = np.array(night)
    obs = np.array(obs)
    
    orig_V2 = np.array(bl_v2df.V2)
    orig_V2err = np.array(bl_v2df.V2err)
    orig_V = np.sqrt(np.absolute(orig_V2))
    orig_Verr = orig_V2err/(2*orig_V)
    orig_wl = np.array(bl_v2df.wl)
    
    return V,Verr,wl,dwl,u_wl,v_wl,MJD,night,obs

def get_scopes_for_ca(bls):
    #Get a list of scopes and their number from the list of baselines
    scopes = []
    for bl in bls:
        scopes.extend(bl.split(' '))
    uni_scopes = np.unique(scopes)
    N = len(uni_scopes)
    return uni_scopes,N

def make_df_from_data(data,wl,dwl,scopes,mjd_cutoff):
    df = pd.DataFrame(columns=['V2','V2err','u_m','v_m','wl','dwl','u_wl','v_wl','sf','bl','MJD','night','obs','bl_obs'])
    df['V2'] = data[4]
    df['V2err'] = data[5]
    df['u_m'] = data[6]
    df['v_m'] = data[7]
    df['wl'] = wl
    df['dwl'] = dwl
    df['u_wl'] = [data[6]/this_wl for this_wl in wl]
    df['v_wl'] = [data[7]/this_wl for this_wl in wl]
    df['sf'] = np.sqrt(df['u_wl']**2 + df['v_wl']**2)/1e6   #SF in Mλ
    df['bl'] = get_bl(data[8],scopes)
    df['MJD'] = data[2]
    df['night'] = math.floor(data[2])
    df['obs'] = get_obs(df['MJD'],mjd_cutoff)
    df = pd.DataFrame.from_records(df.to_dict("records"))
    df = df[df['obs'] != 'X']
    if len(df) == 0:
        return df
    df['bl_obs'] = df[['bl','obs']].agg(' '.join, axis=1)
    
    return df

def get_obs(mjd_slice,mjd_cutoff):
    #Makes a slice that gives the observation number based on the MJD and cutoffs
    obs_slice = [get_ind_obs(mjd,mjd_cutoff) for mjd in mjd_slice]
    
    return obs_slice

def get_ind_obs(mjd,mjd_cutoff):
    #Determines what obs an individual mjd would fall under
    for i,obs in enumerate(mjd_cutoff.obs):
        if mjd > mjd_cutoff.mjd_start.iloc[i] and mjd < mjd_cutoff.mjd_end.iloc[i]:
            return str(obs)
    
    return 'X'

def get_scopes(data):
    #scope[sta_index] = scope_name
    scopes = dict()
    for d in data:
        scopes[d[2]] = d[0]
    return scopes

def get_bl(data,scopes):
    these_scopes = sorted([scopes[data[0]],scopes[data[1]]])
    bl_str = '{} {}'.format(these_scopes[0],these_scopes[1])
    return bl_str

def make_group_index(v2df):
    """
    Build a sorted list of unique (combiner, obs) groups present in v2df and
    return an integer index array mapping each row to its group.

    Returns
    -------
    groups : list of (combiner, obs) tuples, sorted
    group_index : np.ndarray of int, shape (len(v2df),)
    """
    groups = sorted(v2df.groupby(['combiner', 'night']).groups.keys())
    group_index = np.array([
        groups.index((row['combiner'], row['night']))
        for _, row in v2df.iterrows()
    ])
    return groups, group_index

def make_v0_arr(v0_params, group_index):
    """Expand per-group v0 scalars into a per-row array."""
    arr = np.empty(len(group_index))
    for g, v0g in enumerate(v0_params):
        arr[group_index == g] = v0g
    return arr

def linear_ldd_func(sf_wl_v0, tht, aK, aH):
    """
    Limb-darkened disk model.

    Parameters
    ----------
    sf_wl_v0 : tuple of (sf, wl, v0_arr)
        sf     : spatial frequency array (Mλ)
        wl     : wavelength array (m)
        v0_arr : per-row visibility scaling array
    tht : angular diameter (mas)
    aK  : K-band limb-darkening coefficient
    aH  : H-band limb-darkening coefficient
    """
    sf, wl, v0_arr = sf_wl_v0
    wl    = np.array(wl)
    v0_arr = np.array(v0_arr)
    tht *= np.pi/(180*3600000)    #convert tht from mas to rad
    sf = np.array(sf)*1e6         #convert SF from Mλ to λ
    x = np.pi*tht*sf
    
    mask_K = (wl > 1.85e-6)
    mask_H = (wl < 1.85e-6)
    
    xK = x[mask_K]
    xH = x[mask_H]
    
    y = np.empty_like(x)
    
    y[mask_K] = v0_arr[mask_K]*(1./((1-aK)/2.+aK/3.)*((1-aK)*jn(1,xK)/xK+aK*np.sqrt(np.pi/2.)*jn(1.5,xK)/xK**1.5))**2
    y[mask_H] = v0_arr[mask_H]*(1./((1-aH)/2.+aH/3.)*((1-aH)*jn(1,xH)/xH+aH*np.sqrt(np.pi/2.)*jn(1.5,xH)/xH**1.5))**2
    
    return y

def linear_ldd_func_single_v0(sf_wl, tht, aK, aH, v0):
    """
    Convenience wrapper for plotting with a single scalar v0.
    sf_wl is a (sf, wl) two-tuple (no group index needed).
    """
    sf, wl = sf_wl
    v0_arr = np.full(len(np.array(sf)), v0)
    return linear_ldd_func((sf, wl, v0_arr), tht, aK, aH)

def linear_ldd_ca_func(sflist, wl, tht, aK, aH):
    tht *= np.pi/(180*3600000)    #convert tht from mas to rad
    sflist = np.vstack(sflist)*1e6   #convert SF from Mλ to λ
    
    wl = np.vstack(wl)
    wl = np.repeat(wl, 4,axis=1)
    
    x=np.pi*tht*sflist
    
    #Vlist = 1./((1-a)/2.+a/3.)*((1-a)*jn(1,x)/x+a*np.sqrt(np.pi/2.)*jn(1.5,x)/x**1.5)
    
    mask_K = (wl > 1.85e-6)
    mask_H = (wl < 1.85e-6)
    
    xK = x[mask_K]
    xH = x[mask_H]
    
    Vlist = np.empty_like(x)
    
    Vlist[mask_K] = 1./((1-aK)/2.+aK/3.)*((1-aK)*jn(1,xK)/xK+aK*np.sqrt(np.pi/2.)*jn(1.5,xK)/xK**1.5)
    Vlist[mask_H] = 1./((1-aH)/2.+aH/3.)*((1-aH)*jn(1,xH)/xH+aH*np.sqrt(np.pi/2.)*jn(1.5,xH)/xH**1.5)
    
    if len(Vlist) == 4:
        CA = (Vlist[0]*Vlist[1])/(Vlist[2]*Vlist[3])
        CA = np.log(np.absolute(CA[0]))
    else:
        CA = np.log(np.absolute((Vlist[:,0]*Vlist[:,1])/(Vlist[:,2]*Vlist[:,3])))
    
    return CA
def power_ldd_func(sf_wl_v0, tht, aK, aH):
    """
    Limb-darkened disk model using power law.
    
    I = mu^a

    Parameters
    ----------
    sf_wl_v0 : tuple of (sf, wl, v0_arr)
        sf     : spatial frequency array (Mλ)
        wl     : wavelength array (m)
        v0_arr : per-row visibility scaling array
    tht : angular diameter (mas)
    aK  : K-band limb-darkening coefficient
    aH  : H-band limb-darkening coefficient
    """
    sf, wl, v0_arr = sf_wl_v0
    wl    = np.array(wl)
    v0_arr = np.array(v0_arr)
    tht *= np.pi/(180*3600000)    #convert tht from mas to rad
    sf = np.array(sf)*1e6         #convert SF from Mλ to λ
    x = np.pi*tht*sf
    
    mask_K = (wl > 1.85e-6)
    mask_H = (wl < 1.85e-6)
    
    xK = x[mask_K]
    xH = x[mask_H]
    
    y = np.empty_like(x)
    
    nuK = aK/2 + 1
    nuH = aH/2 + 1
    
    y[mask_K] = v0_arr[mask_K]*(gamma(nuK+1)*jn(nuK, xK)/((xK/2)**nuK))**2
    y[mask_H] = v0_arr[mask_H]*(gamma(nuH+1)*jn(nuH, xH)/((xH/2)**nuH))**2
    
    return y

def power_ldd_func_single_v0(sf_wl, tht, aK, aH, v0):
    """
    Convenience wrapper for plotting with a single scalar v0.
    sf_wl is a (sf, wl) two-tuple (no group index needed).
    """
    sf, wl = sf_wl
    v0_arr = np.full(len(np.array(sf)), v0)
    return power_ldd_func((sf, wl, v0_arr), tht, aK, aH)

def power_ldd_ca_func(sflist, wl, tht, aK, aH):
    tht *= np.pi/(180*3600000)    #convert tht from mas to rad
    sflist = np.vstack(sflist)*1e6   #convert SF from Mλ to λ
    
    wl = np.vstack(wl)
    wl = np.repeat(wl, 4,axis=1)
    
    x=np.pi*tht*sflist
    
    #Vlist = 1./((1-a)/2.+a/3.)*((1-a)*jn(1,x)/x+a*np.sqrt(np.pi/2.)*jn(1.5,x)/x**1.5)
    
    mask_K = (wl > 1.85e-6)
    mask_H = (wl < 1.85e-6)
    
    xK = x[mask_K]
    xH = x[mask_H]
    
    Vlist = np.empty_like(x)
    
    nuK = aK/2 + 1
    nuH = aH/2 + 1
    
    Vlist[mask_K] = gamma(nuK+1)*jn(nuK, xK)/((xK/2)**nuK)
    Vlist[mask_H] = gamma(nuH+1)*jn(nuH, xH)/((xH/2)**nuH)
    
    if len(Vlist) == 4:
        CA = (Vlist[0]*Vlist[1])/(Vlist[2]*Vlist[3])
        CA = np.log(np.absolute(CA[0]))
    else:
        CA = np.log(np.absolute((Vlist[:,0]*Vlist[:,1])/(Vlist[:,2]*Vlist[:,3])))
    
    return CA

def select_files(scenario,data_dir,star,nights,combiners):
    #Determines what files to use based on the scenario
    file_groups = []
    notes = []
    if scenario == 'by_file':
        all_files = list(map(str, data_dir.glob("{}*.oifits".format(star))))
        for f in all_files:
            file_groups.append([f])
            fn = f.split('/')[-1]
            fn = fn.replace('.oifits','')
            notes.append(fn)
    if scenario == 'by_night':
        for n in nights:
            files = list(map(str, data_dir.glob("{}*{}.oifits".format(star,n))))
            file_groups.append(files)
            notes.append('{}.{}'.format(star,n))
    if scenario == 'by_combiner':
        for bc in combiners:
            files = list(map(str, data_dir.glob("{}*{}*.oifits".format(star,bc))))
            file_groups.append(files)
            notes.append('{}.{}'.format(star,bc))
    if scenario == 'all':
        all_files = list(map(str, data_dir.glob("{}*.oifits".format(star))))
        file_groups.append(all_files)
        notes.append('{}.all'.format(star))
    
    
    return file_groups,notes

def make_v2df(files,data_dir):
    #makes the pandas dataframe with all the V2 data based on the given files
    v2df = pd.DataFrame(columns=['V2','V2err','u_m','v_m','wl','dwl','u_wl','v_wl','sf','bl','MJD','night','obs','bl_obs','combiner'])
    mjd_cutoffs = pd.read_csv(data_dir / 'MJD_cutoffs.csv')
    for this_file in files:
        star = this_file.split('/')[-1].split('.')[0]
        combiner = this_file.split('/')[-1].split('.')[1]
        this_mjd_cutoff = mjd_cutoffs[mjd_cutoffs.starname == star]
        with fits.open(this_file) as data:
            wl = []
            dwl = []
            for wl_dwl in data['OI_WAVELENGTH'].data:
                wl.append(wl_dwl[0])
                dwl.append(wl_dwl[1])
            scopes = get_scopes(data['OI_ARRAY'].data)
            for this_data in data['OI_VIS2'].data:
                this_df = make_df_from_data(this_data,wl,dwl,scopes,this_mjd_cutoff)
                if len(this_df) == 0:
                    continue
                this_df['combiner'] = combiner   # tag each row with its instrument
                if len(v2df) == 0:
                    v2df = this_df
                else:
                    v2df = pd.concat([v2df,this_df])
    v2df = v2df.dropna(subset=['V2'])
    return v2df

def fit_v2(v2df, law, diam_init, ldcK_init, ldcH_init, fix_ldcK, fix_ldcH,
           x=None, y=None, yerr=None, ref_v2df=None):
    """
    Fit a limb-darkened disk model to V^2 data.

    One independent v0 scaling factor is fitted per (combiner, obs) group so
    that MIRCX and MYSTIC, and different nights, each get their own scaling.

    Parameters
    ----------
    v2df      : DataFrame with all V^2 data (used for xdata when x is None)
    law       : fitting law string, currently only 'linear' and 'power' are supported
    diam_init : initial angular diameter estimate (mas)
    ldcK_init : initial K-band limb-darkening coefficient
    ldcH_init : initial H-band limb-darkening coefficient
    fix_ldcK  : if True, hold aK fixed at ldcK_init
    fix_ldcH  : if True, hold aH fixed at ldcH_init
    x, y, yerr: optional pre-selected x/y/yerr arrays for bootstrap resampling
    ref_v2df  : when x/y/yerr are supplied, pass the original (un-resampled)
                v2df here so the group structure stays consistent across
                bootstrap iterations

    Returns
    -------
    popt   : np.ndarray  [diam, ldcK, ldcH, v0_g0, v0_g1, …]
    pcov   : covariance matrix from curve_fit
    groups : list of (combiner, obs) tuples in the same order as the v0 values
    """
    diam_lim = [diam_init/2, diam_init*2]
    ldcK_lim = [ldcK_init/5, ldcK_init*5]
    ldcH_lim = [ldcH_init/5, ldcH_init*5]
    v0_lim   = [0.5, 2.0]

    # Build group structure from v2df
    groups, group_index = make_group_index(v2df)
    N_groups = len(groups)

    # Initial guesses and bounds: [diam, ldcK, ldcH, v0_g0, …, v0_gN]
    initial_guess = [diam_init, ldcK_init, ldcH_init] + [1.0] * N_groups
    lower_bound   = [diam_lim[0], ldcK_lim[0], ldcH_lim[0]] + [v0_lim[0]] * N_groups
    upper_bound   = [diam_lim[1], ldcK_lim[1], ldcH_lim[1]] + [v0_lim[1]] * N_groups

    # xdata tuple includes the group index so the wrapper can build v0_arr
    if x is not None:
        xdata  = (list(x), list(ref_df['wl']), group_index)
        ydata  = y
        yerrdata = yerr
    else:
        xdata  = (list(v2df['sf']), list(v2df['wl']), group_index)
        ydata  = v2df['V2']
        yerrdata = v2df['V2err']
        
    # Build a wrapper function for each fix_ldc combination.
    # The wrapper expands per-group v0 scalars into a per-row array
    # before calling linear_ldd_func.
    if fix_ldcK and fix_ldcH:
        def fit_func(sf_wl_idx, diam, *v0_params):
            sf, wl, idx = sf_wl_idx
            v0_arr = make_v0_arr(v0_params, idx)
            if law == 'linear':
                return linear_ldd_func((sf, wl, v0_arr), diam, ldcK_init, ldcH_init)
            elif law == 'power':
                return linear_ldd_func((sf, wl, v0_arr), diam, ldcK_init, ldcH_init)
        p0     = [initial_guess[0]] + [1.0] * N_groups
        bounds = (
            [lower_bound[0]] + [v0_lim[0]] * N_groups,
            [upper_bound[0]] + [v0_lim[1]] * N_groups
        )
    elif fix_ldcK:
        def fit_func(sf_wl_idx, diam, ldcH, *v0_params):
            sf, wl, idx = sf_wl_idx
            v0_arr = make_v0_arr(v0_params, idx)
            if law == 'linear':
                return linear_ldd_func((sf, wl, v0_arr), diam, ldcK_init, ldcH)
            elif law == 'power':
                return power_ldd_func((sf, wl, v0_arr), diam, ldcK_init, ldcH)
        p0     = [initial_guess[0], initial_guess[2]] + [1.0] * N_groups
        bounds = (
            [lower_bound[0], lower_bound[2]] + [v0_lim[0]] * N_groups,
            [upper_bound[0], upper_bound[2]] + [v0_lim[1]] * N_groups
        )
    elif fix_ldcH:
        def fit_func(sf_wl_idx, diam, ldcK, *v0_params):
            sf, wl, idx = sf_wl_idx
            v0_arr = make_v0_arr(v0_params, idx)
            if law == 'linear':
                return linear_ldd_func((sf, wl, v0_arr), diam, ldcK, ldcH_init)
            elif law == 'power':
                return power_ldd_func((sf, wl, v0_arr), diam, ldcK, ldcH_init)
        p0     = [initial_guess[0], initial_guess[1]] + [1.0] * N_groups
        bounds = (
            [lower_bound[0], lower_bound[1]] + [v0_lim[0]] * N_groups,
            [upper_bound[0], upper_bound[1]] + [v0_lim[1]] * N_groups
        )
    else:
        def fit_func(sf_wl_idx, diam, ldcK, ldcH, *v0_params):
            sf, wl, idx = sf_wl_idx
            v0_arr = make_v0_arr(v0_params, idx)
            if law == 'linear':
                return linear_ldd_func((sf, wl, v0_arr), diam, ldcK, ldcH)
            elif law == 'power':
                return power_ldd_func((sf, wl, v0_arr), diam, ldcK, ldcH)
        p0     = initial_guess
        bounds = (lower_bound, upper_bound)

    popt_free, pcov = curve_fit(
        fit_func, xdata, ydata,
        p0=p0, bounds=bounds, sigma=yerrdata
    )

    # Reconstruct full popt in canonical order: [diam, ldcK, ldcH, v0_g0, …]
    if fix_ldcK and fix_ldcH:
        popt = np.array([popt_free[0], ldcK_init, ldcH_init] + list(popt_free[1:]))
    elif fix_ldcK:
        popt = np.array([popt_free[0], ldcK_init] + list(popt_free[1:]))
    elif fix_ldcH:
        popt = np.array([popt_free[0], popt_free[1], ldcH_init] + list(popt_free[2:]))
    else:
        popt = popt_free

    return popt, pcov, groups

def fit_ca(cadf,law,diam_init,ldcK_init,ldcH_init,fix_ldcK,fix_ldcH,x=None,y=None,yerr=None,maxiter=5000):
    diam_lim = [diam_init/2,diam_init*2]
    ldcK_lim = [ldcK_init/5,ldcK_init*5]
    ldcH_lim = [ldcH_init/5,ldcH_init*5]
    
    if fix_ldcK:
        ldcK_lim = [ldcK_init,ldcK_init]
    if fix_ldcH:
        ldcH_lim = [ldcH_init,ldcH_init]
    
    bounds=[(diam_lim[0],diam_lim[1]),(ldcK_lim[0],ldcK_lim[1]),(ldcH_lim[0],ldcH_lim[1])]
    
    #Fits a limb-darkened disk to the data
    result = differential_evolution(ca_objective,bounds,args=(cadf['lnCA'],cadf['sflist'],cadf['wl'],law))
    bf = result.x
    
    return bf

def ca_objective(params,lnCA,sflist,wl,law):
    if law == 'linear':
        return np.sum((lnCA-linear_ldd_ca_func(sflist, wl, *params))**2)
    elif law == 'power':
        return np.sum((lnCA-power_ldd_ca_func(sflist, wl, *params))**2)

def report_fit_results(v2df, law, popt, groups, save_dir, scenario, note):
    """Print and save report of the fit results, and plot the fits."""
    report_file    = save_dir / '{}.txt'.format(note)
    best_fit_plot  = save_dir / '{}.fit.png'.format(note)
    print(note)

    N_groups = len(groups)
    dof = 3 + N_groups   # diam, ldcK, ldcH, + one v0 per group

    diam  = popt[0]
    ldcK  = popt[1]
    ldcH  = popt[2]
    v0s   = popt[3:]   # one per group

    # Build per-row v0 array for residual / model evaluation
    _, group_index = make_group_index(v2df)
    v0_arr = make_v0_arr(v0s, group_index)
    
    if law == 'linear':
        r = v2df['V2'] - linear_ldd_func((v2df['sf'], v2df['wl'], v0_arr), diam, ldcK, ldcH)
    elif law == 'power':
        r = v2df['V2'] - power_ldd_func((v2df['sf'], v2df['wl'], v0_arr), diam, ldcK, ldcH)
    chisq     = sum((r / v2df['V2err'])**2)
    red_chisq = chisq / (len(v2df['V2']) - dof)

    # Write report
    report  = 'Initial V2 fit results\n'
    report += '---------------------------\n'
    report += 'Diam (mas) | LDC_K | LDC_H\n'
    report += '{:.3f}      | {:.3f} | {:.3f}\n'.format(diam, ldcK, ldcH)
    report += 'Per-group V2(0) scaling factors:\n'
    for g, (combiner, night) in enumerate(groups):
        report += '  {}/MJD {}: {:.3f}\n'.format(combiner, night, v0s[g])
    report += 'chi^2 = {:.3f}\n'.format(chisq)
    report += 'reduced chi^2 = {:.3f}\n'.format(red_chisq)
    report += '---------------------------\n'

    print(report)
    with open(report_file, 'w') as f:
        f.write(report)

    # Plot the data after it has been scaled by the scaling factor for the relevant group
    for g, (combiner, night) in enumerate(groups):
        mask = (group_index == g)
        sf_g  = np.array(v2df['sf'])[mask]
        v2_g  = np.array(v2df['V2'])[mask]
        v2e_g = np.array(v2df['V2err'])[mask]
        plt.errorbar(sf_g, v2_g/v0s[g], yerr=v2e_g, fmt='.', color='k')

    # Plot a visibility curve with the best fitting diameter and LDCs using a scaling factor of 1
    #   because the data are scaled
    sf_plot = np.linspace(float(v2df['sf'].min()), float(v2df['sf'].max()), 300)
    plot_wl = 2.2e-6
    wl_plot = np.full_like(sf_plot, plot_wl)
    if law == 'linear':
        y_plot  = linear_ldd_func_single_v0((sf_plot, wl_plot), diam, ldcK, ldcH, 1.0)
    elif law == 'power':
        y_plot  = power_ldd_func_single_v0((sf_plot, wl_plot), diam, ldcK, ldcH, 1.0)
    plt.plot(sf_plot, y_plot, '-', color='g', alpha=0.8,zorder=3)

    plt.yscale('log')
    plt.ylim([1e-5, 3])
    plt.xlabel('Spatial Frequency (Mλ)')
    plt.ylabel('Scaled Squared Visibility')
    plt.legend(fontsize=6)
    plt.savefig(best_fit_plot)
    plt.close()

    return

def report_ca_fit_results(cadf,law,popt,save_dir,scenario,note):
    #print and save report of the fit results, and plot the fits
    report_file = save_dir / '{}.CA.txt'.format(note)
    best_fit_plot = save_dir / '{}.CA.fit.png'.format(note)
    residual_plot = save_dir / '{}.CA.fit.res.png'.format(note)
    print(note)
    
    dof = 3
    #Calculate chi^2
    if law == 'linear':
        r = cadf['lnCA']-linear_ldd_ca_func(cadf['sflist'],cadf['wl'], *popt)
    elif law == 'power':
        r = cadf['lnCA']-power_ldd_ca_func(cadf['sflist'],cadf['wl'], *popt)
    chisq = sum((r / cadf['lnCAerr']) **2 )
    red_chisq = chisq/(len(cadf['lnCA'])-dof)
    
    [diam,ldcK,ldcH] = popt
    
    #Write report
    report = ''
    report += 'Initial CA fit results\n'
    report += '---------------------------\n'
    report += 'Diam (mas) | LDC_K | LDC_H    \n'
    report += '{:.3f}      | {:.3f} | {:.3f} \n'.format(diam,ldcK,ldcH)
    report += 'chi^2 = {:.3f}\n'.format(chisq)
    report += 'reduced chi^2 = {:.3f}\n'.format(red_chisq)
    report += '---------------------------\n'
    
    print(report)
    with open(report_file,'w') as f:
        f.write(report)
    
    if law == 'linear':
        plt.plot(cadf['sfmax'],linear_ldd_ca_func(cadf['sflist'],cadf['wl'], *popt), 'g.', zorder=3)
    elif law == 'power':
        plt.plot(cadf['sfmax'],power_ldd_ca_func(cadf['sflist'],cadf['wl'], *popt), 'g.', zorder=3)
    plt.errorbar(cadf['sfmax'],cadf['lnCA'],yerr=cadf['lnCAerr'],fmt='k.')
    #plt.yscale('log')
    plt.ylim([-15,15])
    plt.xlabel('Max Spatial Frequency in Quadrangle (Mλ)')
    plt.ylabel('log(Closure Amplitude)')
    plt.savefig(best_fit_plot)
    plt.close()
    
    plt.errorbar(cadf['sfmax'],r,yerr=cadf['lnCAerr'],fmt='k.')
    plt.xlabel('Max Spatial Frequency in Quadrangle (Mλ)')
    plt.ylabel('log(CA_obs) - log(CA_calc)')
    plt.ylim([-15,15])
    plt.savefig(residual_plot)
    plt.close()
    
    for quad in cadf.quad.unique():
        nows_quad = quad.replace(' ','')
        quad_plot = save_dir / '{}.CA.{}.fit.png'.format(note,nows_quad)
        quad_cadf = cadf[cadf.quad == quad]
        if law == 'linear':
            plt.plot(quad_cadf['sfmax'],linear_ldd_ca_func(quad_cadf['sflist'], quad_cadf['wl'], *popt), 'g.', zorder=3)
        elif law == 'linear':
            plt.plot(quad_cadf['sfmax'],power_ldd_ca_func(quad_cadf['sflist'], quad_cadf['wl'], *popt), 'g.', zorder=3)
        plt.errorbar(quad_cadf['sfmax'],quad_cadf['lnCA'],yerr=quad_cadf['lnCAerr'],fmt='k.')
        plt.ylim([-15,15])
        plt.title(quad)
        plt.xlabel('Max Spatial Frequency in Quadrangle (Mλ)')
        plt.ylabel('log(Closure Amplitude)')
        plt.savefig(quad_plot)
        plt.close()
            
    
    return

def do_bootstrap_by_obs(v2df, law, save_dir, scenario, note, diam_init, ldcK_init, ldcH_init,
                        bf_popt, groups, boot_N, plot_sf, fix_ldcK, fix_ldcH,
                        ca=False, capopt=[], cal1popt=[], cadf=[], l1_v2df=[], l1_cadf=[]):
    """
    Bootstrap resampling by observation.

    bf_popt and groups come from the initial fit_v2() call and define the
    best-fit parameters and (combiner, obs) group ordering used throughout.
    """
    star = note.split('.')[0]
    N_groups = len(groups)
    v2_results = []
    v2_report_file  = save_dir / '{}.bootstrap.txt'.format(note)
    v2_results_file = save_dir / '{}.bootstrap.res'.format(note)
    v2_results_plot = save_dir / '{}.bootstrap.pdf'.format(note)
    ca_results   = []
    ca_report_file  = save_dir / '{}.CA.bootstrap.txt'.format(note)
    ca_results_file = save_dir / '{}.CA.bootstrap.res'.format(note)
    ca_results_plot = save_dir / '{}.CA.bootstrap.pdf'.format(note)
    cal1_results = []
    cal1_report_file  = save_dir / '{}.CAL1.bootstrap.txt'.format(note)
    cal1_results_file = save_dir / '{}.CAL1.bootstrap.res'.format(note)
    cal1_results_plot = save_dir / '{}.CAL1.bootstrap.pdf'.format(note)
    
    quad_results_plots = dict()
    
    unique_obs = v2df['bl_obs'].unique()
    obs_n = len(unique_obs)    #Number of unique baseline+observation pairs
    
    
    plots = dict()
    plot_list = ['V2']
    if ca:
        plot_list.append('CA')
        plot_list.append('CAL1')
        for quad in cadf.quad.unique():
            plot_list.append(quad)
            plot_list.append(quad+'L1')
            nows_quad = quad.replace(' ','')
            quad_results_plots[quad]      = save_dir / '{}.CA.{}.bootstrap.pdf'.format(note,nows_quad)
            quad_results_plots[quad+'L1'] = save_dir / '{}.CAL1.{}.bootstrap.pdf'.format(note,nows_quad)
    
    for plot_name in plot_list:
        fig, ax = plt.subplots()
        plots[plot_name] = (fig,ax)
    
    for n in progressbar(range(boot_N)):
        #Make bootstrap choice
        choice_set = rnd.choices(range(obs_n), k=obs_n)
        #Define v2df for this bootstrap choice
        this_v2df = get_v2df_from_choice(v2df, unique_obs[choice_set])
        #Do V^2 fit, passing the original v2df as ref so group structure is stable
        popt, pcov, _ = fit_v2(this_v2df, law, diam_init, ldcK_init, ldcH_init,
                                fix_ldcK, fix_ldcH, ref_v2df=v2df)
        v2_results.append(popt)

        # Plot visibility curves for each bootstrap with the best fitting diameter and LDCs for that iteration 
        #   using a scaling factor of 1 because the data are scaled
        v0s_boot = popt[3:]
        plot_wl = 2.2e-6 #H-band representative wavelength
        wl_plot = np.empty_like(np.array(plot_sf)) + plot_wl
        if law == 'linear':
            y_plot  = linear_ldd_func_single_v0((plot_sf, wl_plot), popt[0], popt[1], popt[2], 1.0)
        elif law == 'power':
            y_plot  = power_ldd_func_single_v0((plot_sf, wl_plot), popt[0], popt[1], popt[2], 1.0)
        plots['V2'][1].plot(plot_sf, y_plot, 'c-', alpha=0.1, zorder=2)
        
        if ca:
            try:
                #Calculate cadf for this bootstrap choice
                this_cadf = make_cadf(this_v2df)
                #Do CA fit
                capopt = fit_ca(this_cadf,law,diam_init,ldcK_init,ldcH_init,fix_ldcK,fix_ldcH,maxiter=100000)
                ca_results.append(capopt)
                #Plot this CA fit for all quadrangles
                if law == 'linear':
                    plots['CA'][1].plot(this_cadf['sfmax'],linear_ldd_ca_func(this_cadf['sflist'],this_cadf['wl'], *capopt), 'c.', alpha=0.1, zorder=2)
                elif law == 'power':
                    plots['CA'][1].plot(this_cadf['sfmax'],power_ldd_ca_func(this_cadf['sflist'],this_cadf['wl'], *capopt), 'c.', alpha=0.1, zorder=2)
                #Plot this CA fit for each quadrangle
                for quad in this_cadf.quad.unique():
                    quad_cadf = this_cadf[this_cadf.quad == quad]
                    if law == 'linear':
                        plots[quad][1].plot(quad_cadf['sfmax'],linear_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *capopt), 'c.', alpha=0.1, zorder=2)
                    elif law == 'power':
                        plots[quad][1].plot(quad_cadf['sfmax'],power_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *capopt), 'c.', alpha=0.1, zorder=2)
                
                #Do stuff with L1 data
                #Define l1_v2df for this bootstrap choice
                this_l1_v2df = get_v2df_from_choice(l1_v2df,unique_obs[choice_set])
                #Calculate l1_cadf for this bootstrap choice
                this_l1_cadf = make_cadf(this_l1_v2df)
                #Do CA L1 fit
                cal1popt = fit_ca(this_l1_cadf,law,diam_init,ldcK_init,ldcH_init,fix_ldcK,fix_ldcH,maxiter=100000)
                cal1_results.append(cal1popt)
                #Plot this CA L1 fit for all quadrangles
                if law == 'linear':
                    plots['CAL1'][1].plot(this_l1_cadf['sfmax'],linear_ldd_ca_func(this_l1_cadf['sflist'],this_l1_cadf['wl'], *cal1popt), 'c.', alpha=0.1, zorder=2)
                elif law == 'power':
                    plots['CAL1'][1].plot(this_l1_cadf['sfmax'],power_ldd_ca_func(this_l1_cadf['sflist'],this_l1_cadf['wl'], *cal1popt), 'c.', alpha=0.1, zorder=2)
                #Plot this CA L1 fit for each quadrangle
                for quad in this_l1_cadf.quad.unique():
                    quad_cadf = this_l1_cadf[this_l1_cadf.quad == quad]
                    if law == 'linear':
                        plots[quad+'L1'][1].plot(quad_cadf['sfmax'],linear_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *cal1popt), 'c.', alpha=0.1, zorder=2)
                    elif law == 'power':
                        plots[quad+'L1'][1].plot(quad_cadf['sfmax'],power_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *cal1popt), 'c.', alpha=0.1, zorder=2)
            except:
                #print('CA fit failed for this iteration')
                pass
                
    #Finalize V^2 plot
    bf_v0s = bf_popt[3:]
    #Get the group index
    _, group_index = make_group_index(v2df)
    #Plot the data after it has been scaled by the best-fit scaling factor for the relevant group
    for g, (combiner, night) in enumerate(groups):
        mask = (group_index == g)
        sf_g  = np.array(v2df['sf'])[mask]
        v2_g  = np.array(v2df['V2'])[mask]
        v2e_g = np.array(v2df['V2err'])[mask]
        if combiner == 'MIRCX':
            bc_col = 'b'
        elif combiner == 'MYSTIC':
            bc_col = 'r'
        else:
            bc_col = 'k'
        plots['V2'][1].errorbar(sf_g,v2_g/bf_v0s[g],yerr=v2e_g,fmt='.',color=bc_col, zorder=1)
    # Plot a visibility curve with the best fitting diameter and LDCs using a scaling factor of 1
    #   because the data are scaled
    plot_wl = 2.2e-6
    wl_plot = np.empty_like(np.array(plot_sf)) + plot_wl
    if law == 'linear':
        y_plot  = linear_ldd_func_single_v0((plot_sf, wl_plot), bf_popt[0], bf_popt[1], bf_popt[2], 1.0)
    elif law == 'power':
        y_plot  = power_ldd_func_single_v0((plot_sf, wl_plot), bf_popt[0], bf_popt[1], bf_popt[2], 1.0)
    plots['V2'][1].plot(plot_sf, y_plot, '-',color='purple', zorder=3)
    plots['V2'][1].set_yscale('log')
    plots['V2'][1].set_ylim([1e-5,3])
    plots['V2'][1].set_xlabel('Spatial Frequency (Mλ)')
    plots['V2'][1].set_ylabel('Scaled Squared Visibility')
    plots['V2'][1].legend(fontsize=6)
    
    if ca:
        #Finalize CA plot
        mircx_mask = (cadf['wl'] < 1.85e-6)
        mystic_mask = (cadf['wl'] > 1.85e-6)
        mircx_sfmax = cadf['sfmax'][mircx_mask]
        mircx_lnCA = cadf['lnCA'][mircx_mask]
        mircx_lnCAerr = cadf['lnCAerr'][mircx_mask]
        mystic_sfmax = cadf['sfmax'][mystic_mask]
        mystic_lnCA = cadf['lnCA'][mystic_mask]
        mystic_lnCAerr = cadf['lnCAerr'][mystic_mask]
        #plots['CA'][1].errorbar(cadf['sfmax'],cadf['lnCA'],yerr=cadf['lnCAerr'],fmt='k.', zorder = 1)
        plots['CA'][1].errorbar(mircx_sfmax,mircx_lnCA,yerr=mircx_lnCAerr,fmt='b.', zorder = 1)
        plots['CA'][1].errorbar(mystic_sfmax,mystic_lnCA,yerr=mystic_lnCAerr,fmt='r.', zorder = 1)
        if law == 'linear':
            y_plot = linear_ldd_ca_func(cadf['sflist'],cadf['wl'], *capopt)
        elif law == 'power':
            y_plot = power_ldd_ca_func(cadf['sflist'],cadf['wl'], *capopt)
        plots['CA'][1].plot(cadf['sfmax'],y_plot, '.',color='purple',zorder=3)
        plots['CA'][1].set_ylim([-15,15])
        plots['CA'][1].set_xlabel('Max Spatial Frequency in Quadrangle (Mλ)')
        plots['CA'][1].set_ylabel('log(Closure Amplitude)')
        plots['CA'][0].savefig(ca_results_plot)
        plt.close(plots['CA'][0])
        
        #Finalize CAL1 plot
        mircx_mask = (l1_cadf['wl'] < 1.85e-6)
        mystic_mask = (l1_cadf['wl'] > 1.85e-6)
        mircx_sfmax = l1_cadf['sfmax'][mircx_mask]
        mircx_lnCA = l1_cadf['lnCA'][mircx_mask]
        mircx_lnCAerr = l1_cadf['lnCAerr'][mircx_mask]
        mystic_sfmax = l1_cadf['sfmax'][mystic_mask]
        mystic_lnCA = l1_cadf['lnCA'][mystic_mask]
        mystic_lnCAerr = l1_cadf['lnCAerr'][mystic_mask]
        #plots['CAL1'][1].errorbar(l1_cadf['sfmax'],l1_cadf['lnCA'],yerr=l1_cadf['lnCAerr'],fmt='k.', zorder = 1)
        plots['CAL1'][1].errorbar(mircx_sfmax,mircx_lnCA,yerr=mircx_lnCAerr,fmt='b.', zorder = 1)
        plots['CAL1'][1].errorbar(mystic_sfmax,mystic_lnCA,yerr=mystic_lnCAerr,fmt='r.', zorder = 1)
        if law == 'linear':
            y_plot = linear_ldd_ca_func(l1_cadf['sflist'],l1_cadf['wl'], *cal1popt)
        elif law == 'power':
            y_plot = power_ldd_ca_func(l1_cadf['sflist'],l1_cadf['wl'], *cal1popt)
        plots['CAL1'][1].plot(l1_cadf['sfmax'],y_plot, '.',color='purple',zorder=3)
        plots['CAL1'][1].set_ylim([-15,15])
        plots['CAL1'][1].set_xlabel('Max Spatial Frequency in Quadrangle (Mλ)')
        plots['CAL1'][1].set_ylabel('log(Closure Amplitude)')
        plots['CAL1'][0].savefig(cal1_results_plot)
        plt.close(plots['CAL1'][0])
        
        #Finalize quad plots
        for quad in cadf.quad.unique():
            try:
                quad_cadf = cadf[cadf.quad == quad]
                plots[quad][1].errorbar(quad_cadf['sfmax'],quad_cadf['lnCA'],yerr=quad_cadf['lnCAerr'],fmt='k.',zorder = 1)
                if law == 'linear':
                    y_plot = linear_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *capopt)
                elif law == 'power':
                    y_plot = power_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *capopt)
                plots[quad][1].plot(quad_cadf['sfmax'],y_plot, '.',color='purple',zorder=3)
                plots[quad][1].set_ylim([-15,15])
                plots[quad][1].set_xlabel('Max Spatial Frequency in Quadrangle (Mλ)')
                plots[quad][1].set_ylabel('log(Closure Amplitude)')
                plots[quad][0].savefig(quad_results_plots[quad])
                plt.close(plots[quad][0])
            except:
                plt.close(plots[quad][0])
            try:
                plots[quad+'L1'][1].errorbar(quad_cadf['sfmax'],quad_cadf['lnCA'],yerr=quad_cadf['lnCAerr'],fmt='k.',zorder = 1)  # Note: using quad_l1_cadf would be ideal but preserving original logic
                if law == 'linear':
                    y_plot = linear_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *cal1popt)
                elif law == 'power':
                    y_plot = power_ldd_ca_func(quad_cadf['sflist'],quad_cadf['wl'], *cal1popt)
                plots[quad+'L1'][1].plot(quad_cadf['sfmax'],y_plot, '.',color='purple',zorder=3)
                plots[quad+'L1'][1].set_ylim([-15,15])
                plots[quad+'L1'][1].set_xlabel('Max Spatial Frequency in Quadrangle (Mλ)')
                plots[quad+'L1'][1].set_ylabel('log(Closure Amplitude)')
                plots[quad+'L1'][0].savefig(quad_results_plots[quad+'L1'])
                plt.close(plots[quad+'L1'][0])
            except:
                plt.close(plots[quad+'L1'][0])

    try:
        #Make V^2 report
        v2_arr = np.array(v2_results)   # shape: (boot_N, 3 + N_groups)
        diams  = v2_arr[:, 0]
        ldcKs  = v2_arr[:, 1]
        ldcHs  = v2_arr[:, 2]
        v0s_all = v2_arr[:, 3:]         # shape: (boot_N, N_groups)

        # Build results dataframe dynamically — one v0 column per group
        v0_col_names = ['v0_{}_{}'.format(combiner, night) for combiner, night in groups]
        v2_results_df = pd.DataFrame()
        v2_results_df['diam']  = diams
        v2_results_df['ldcK']  = ldcKs
        v2_results_df['ldcH']  = ldcHs
        for g, col in enumerate(v0_col_names):
            v2_results_df[col] = v0s_all[:, g]
        v2_results_df.to_csv(v2_results_file, index=False)

        # Calculate statistics
        diam_median  = stat.median(diams)
        diam_stdev   = stat.stdev(diams)
        ldcK_median  = stat.median(ldcKs)
        ldcK_stdev   = stat.stdev(ldcKs)
        ldcH_median  = stat.median(ldcHs)
        ldcH_stdev   = stat.stdev(ldcHs)

        # Write report
        v2_report  = 'V^2 Bootstrap results\n'
        v2_report += '------------------------------------------\n'
        v2_report += 'Diam (mas)      | LDC_K           | LDC_H\n'
        v2_report += '{:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f}\n'.format(
            diam_median, diam_stdev, ldcK_median, ldcK_stdev, ldcH_median, ldcH_stdev)
        v2_report += 'Per-group V2(0) scaling factors:\n'
        for g, (combiner, night) in enumerate(groups):
            v0_med = stat.median(v0s_all[:, g])
            v0_std = stat.stdev(v0s_all[:, g])
            v2_report += '  {}/MJD {}: {:.3f} +/- {:.3f}\n'.format(combiner, night, v0_med, v0_std)
        v2_report += '------------------------------------------\n'

        print(v2_report)
        with open(v2_report_file, 'w') as f:
            f.write(v2_report)

        # Build title string for plot
        v0_title_parts_mircx = []
        v0_title_parts_mystic = []
        for g, (combiner, obs) in enumerate(groups):
            v0_med = stat.median(v0s_all[:, g])
            v0_std = stat.stdev(v0s_all[:, g])
            if combiner == 'MIRCX':
                v0_title_parts_mircx.append('V0({}/MJD {}): {:.3f}±{:.3f}'.format(combiner, obs, v0_med, v0_std))
            if combiner == 'MYSTIC':
                v0_title_parts_mystic.append('V0({}/MJD {}): {:.3f}±{:.3f}'.format(combiner, obs, v0_med, v0_std))
            
        title_str = '{}: Diam={:.3f}±{:.3f} mas, LDC_K={:.3f}±{:.3f}, LDC_H={:.3f}±{:.3f}'.format(
            star, diam_median, diam_stdev, ldcK_median, ldcK_stdev, ldcH_median, ldcH_stdev)
        
        title_str += '\n' + ', '.join(v0_title_parts_mircx)
        title_str += '\n' + ', '.join(v0_title_parts_mystic)
        plots['V2'][1].set_title(title_str, fontsize=6)
        plots['V2'][0].savefig(v2_results_plot)
        plt.close(plots['V2'][0])

    except:
        print('V^2 bootstrap report failed')

    try:        
        #Make CA report
        ca_diams = np.array(ca_results).T[0]
        ca_ldcKs = np.array(ca_results).T[1]
        ca_ldcHs = np.array(ca_results).T[2]
        ca_results_df = pd.DataFrame(columns=['diam','ldcK','ldcH'])
        ca_results_df['diam'] = ca_diams
        ca_results_df['ldcK'] = ca_ldcKs
        ca_results_df['ldcH'] = ca_ldcHs
        ca_results_df.to_csv(ca_results_file,index=False)
        ca_diam_median = stat.median(ca_diams)
        ca_diam_stdev  = stat.stdev(ca_diams)
        ca_ldcK_median = stat.median(ca_ldcKs)
        ca_ldcK_stdev  = stat.stdev(ca_ldcKs)
        ca_ldcH_median = stat.median(ca_ldcHs)
        ca_ldcH_stdev  = stat.stdev(ca_ldcKs)
        ca_report  = 'CA Bootstrap results\n'
        ca_report += '----------------------------------\n'
        ca_report += 'Diam (mas)      | LDC_K | LDC_H \n'
        ca_report += '{:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} \n'.format(ca_diam_median,ca_diam_stdev,ca_ldcK_median,ca_ldcK_stdev,ca_ldcH_median,ca_ldcH_stdev)
        ca_report += '----------------------------------\n'
        print(ca_report)
        with open(ca_report_file,'w') as f:
            f.write(ca_report)
    except:
        print('CA bootstrap report failed')

    try:
        #Make CA L1 report
        cal1_diams = np.array(cal1_results).T[0]
        cal1_ldcKs = np.array(cal1_results).T[1]
        cal1_ldcHs = np.array(cal1_results).T[2]
        cal1_results_df = pd.DataFrame(columns=['diam','ldcK','ldcH'])
        cal1_results_df['diam'] = cal1_diams
        cal1_results_df['ldcK'] = cal1_ldcKs
        cal1_results_df['ldcH'] = cal1_ldcHs
        cal1_results_df.to_csv(cal1_results_file,index=False)
        cal1_diam_median = stat.median(cal1_diams)
        cal1_diam_stdev  = stat.stdev(cal1_diams)
        cal1_ldcK_median = stat.median(cal1_ldcKs)
        cal1_ldcK_stdev  = stat.stdev(cal1_ldcKs)
        cal1_ldcH_median = stat.median(cal1_ldcHs)
        cal1_ldcH_stdev  = stat.stdev(cal1_ldcHs)
        cal1_report  = 'CA L1 Bootstrap results\n'
        cal1_report += '----------------------------------\n'
        cal1_report += 'Diam (mas)      | LDC_K | LDC_H \n'
        cal1_report += '{:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} | {:.3f} +/- {:.3f} \n'.format(cal1_diam_median,cal1_diam_stdev,cal1_ldcK_median,cal1_ldcK_stdev,cal1_ldcH_median,cal1_ldcH_stdev)
        cal1_report += '----------------------------------\n'
        print(cal1_report)
        with open(cal1_report_file,'w') as f:
            f.write(cal1_report)
    except:
        print('CA L1 bootstrap report failed')
        
    return
    

  
def get_v2df_from_choice(v2df,chosen_obs):
    #Based on the randomly selected observations, makes a new v2df 
    #V^2 points for each obs are randomly moved within the average uncertainties for that obs 
    new_v2df = pd.DataFrame({col: pd.Series(dtype=dt) for col,dt in v2df.dtypes.items()})
    for obs in chosen_obs:
        this_v2df = v2df[v2df.bl_obs == obs]
        
        orig_v2 = np.array(this_v2df['V2'])
        this_v2err = list(this_v2df['V2err'])
        this_sf = list(this_v2df['sf'])
        
        avg_v2err = np.mean(this_v2err)
        new_v2 = orig_v2 * rnd.gauss(1,avg_v2err)
        
        
        this_v2df.loc[:,'V2'] = new_v2
        
        new_v2df = pd.concat([new_v2df,this_v2df],ignore_index=True)
        
    return new_v2df



if __name__ == '__main__':
    main()