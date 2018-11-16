#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from gPhoton import gFind
from gPhoton import gAperture
from gPhoton import gMap
from gPhoton.gphoton_utils import read_lc
import datetime

from astropy.time import Time
from astropy import units as u
# from astropy.analytic_functions import blackbody_lambda #OLD!
from astropy.modeling.blackbody import blackbody_lambda
from gatspy.periodic import LombScargleFast

#import extinction

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})

ra = 359.1879
dec = -0.82008

exp_data = gFind(band='NUV', skypos=[ra, dec], exponly=True)

exp_data

(exp_data['NUV']['t0'] - exp_data['NUV']['t0'][0]) / (60. * 60. * 24. * 365.)

step_size = 5

target = 'J235645.0-004912'

phot_rad = 0.0045 # in deg
ap_in = 0.0050  #in deg
ap_out = 0.0060  #in deg

print(datetime.datetime.now())
for k in range(len(exp_data['NUV']['t0'])):
    photon_events = gAperture(band='NUV', skypos=[ra, dec], stepsz=step_size, radius=phot_rad,
                              annulus=[ap_in, ap_out], verbose=3, csvfile=target+ '_' +str(k)+"_lc.csv",                             
                              trange=[int(exp_data['NUV']['t0'][k]), int(exp_data['NUV']['t1'][k])+1], 
                               overwrite=True)
    
    print(datetime.datetime.now(), k)

med_flux = np.array(np.zeros(4), dtype='float')
med_flux_err = np.array(np.zeros(4), dtype='float')

time_big = np.array([], dtype='float')
mag_big = np.array([], dtype='float')
flux_big = np.array([], dtype='float')

for k in range(4):
    data = read_lc(target+ '_' +str(k)+"_lc.csv")
    med_flux[k] = np.nanmedian(data['flux_bgsub'])
    med_flux_err[k] = np.std(data['flux_bgsub'])

    time_big = np.append(time_big, data['t_mean'])
    flux_big = np.append(flux_big, data['flux_bgsub'])
    mag_big = np.append(mag_big, data['mag'])
    
#     t0k = Time(int(data['t_mean'][0]) + 315964800, format='unix').mjd
    flg0 = np.where((data['flags'] == 0))[0]
    
    # for Referee: convert GALEX time -> MJD
    t_unix = Time(data['t_mean'] + 315964800, format='unix')
    mjd_time = t_unix.mjd
    t0k = (mjd_time[0])
    
    plt.figure()
    plt.errorbar((mjd_time - t0k)*24.*60.*60., data['flux_bgsub']/(1e-15), yerr=data['flux_bgsub_err']/(1e-15), 
             marker='.', linestyle='none', c='k', alpha=0.75, lw=0.5, markersize=2)
    
    plt.errorbar((mjd_time[flg0] - t0k)*24.*60.*60., data['flux_bgsub'][flg0]/(1e-15), 
                 yerr=data['flux_bgsub_err'][flg0]/(1e-15), 
             marker='.', linestyle='none')
#     plt.xlabel('GALEX time (sec - '+str(t0k)+')')
    plt.xlabel('MJD - '+ format(t0k, '9.3f') +' (seconds)')
    plt.ylabel('NUV Flux \n'
               r'(x10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ ${\rm\AA}^{-1}$)')
    #plt.savefig(target+ '_' +str(k)+"_lc.pdf", dpi=150, bbox_inches='tight', pad_inches=0.25)

    
    flagcol = np.zeros_like(mjd_time)
    flagcol[flg0] = 1
    dfout = pd.DataFrame(data={'MJD':mjd_time, 
                               'flux':data['flux_bgsub']/(1e-15), 
                               'fluxerr':data['flux_bgsub_err']/(1e-15),
                               'flag':flagcol})
    dfout.to_csv(target+ '_' +str(k)+'data.csv', index=False, columns=('MJD', 'flux','fluxerr', 'flag'))

k=2
data = read_lc(target+ '_' +str(k)+"_lc.csv")

t0k = int(data['t_mean'][0])
plt.figure(figsize=(14,5))
plt.errorbar(data['t_mean'] - t0k, data['flux_bgsub'], yerr=data['flux_bgsub_err'], marker='.', linestyle='none')
plt.xlabel('GALEX time (sec - '+str(t0k)+')')
plt.ylabel('NUV Flux')
 
flg0 = np.where((data['flags'] == 0))[0]
plt.figure(figsize=(14,5))
plt.errorbar(data['t_mean'][flg0] - t0k, data['flux_bgsub'][flg0]/(1e-15), yerr=data['flux_bgsub_err'][flg0]/(1e-15), 
             marker='.', linestyle='none')
plt.xlabel('GALEX time (sec - '+str(t0k)+')')
# plt.ylabel('NUV Flux')
plt.ylabel('NUV Flux \n' 
           r'(x10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ ${\rm\AA}^{-1}$)')
plt.title('Flags = 0')

minper = 10 # my windowing
maxper = 200000
nper = 1000
pgram = LombScargleFast(fit_offset=False)
pgram.optimizer.set(period_range=(minper,maxper))

pgram = pgram.fit(time_big - min(time_big), flux_big - np.nanmedian(flux_big))

df = (1./minper - 1./maxper) / nper
f0 = 1./maxper

pwr = pgram.score_frequency_grid(f0, df, nper)

freq = f0 + df * np.arange(nper)
per = 1./freq

##
plt.figure()
plt.plot(per, pwr, lw=0.75)
plt.xlabel('Period (seconds)')
plt.ylabel('L-S Power')
plt.xscale('log')
plt.xlim(10,500)
plt.savefig('periodogram.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

t_unix = Time(exp_data['NUV']['t0'] + 315964800, format='unix')
mjd_time_med = t_unix.mjd
t0k = (mjd_time[0])

plt.figure(figsize=(9,5))
plt.errorbar(mjd_time_med - mjd_time_med[0], med_flux/1e-15, yerr=med_flux_err/1e-15, linestyle='none', marker='o')
plt.xlabel('MJD - '+format(mjd_time[0], '9.3f')+' (days)')
# plt.ylabel('NUV Flux')
plt.ylabel('NUV Flux \n' 
           r'(x10$^{-15}$ erg s$^{-1}$ cm$^{-2}$ ${\rm\AA}^{-1}$)')
# plt.title(target)
plt.savefig(target+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

# average time of the gPhoton data
print(np.mean(exp_data['NUV']['t0']))
t_unix = Time(np.mean(exp_data['NUV']['t0']) + 315964800, format='unix')
t_date = t_unix.yday
print(t_date)

mjd_date = t_unix.mjd
print(mjd_date)

plt.errorbar([10, 14], [16.46, 16.499], yerr=[0.01, 0.006], linestyle='none', marker='o')
plt.xlabel('Quarter (approx)')
plt.ylabel(r'$m_{NUV}$ (mag)')
plt.ylim(16.52,16.44)

gck_time = Time(1029843320.995 + 315964800, format='unix')
gck_time.mjd

# and to push the comparison to absurd places...
#http://astro.uchicago.edu/~bmontet/kic8462852/reduced_lc.txt

#df = pd.read_table('reduced_lc.txt', delim_whitespace=True, skiprows=1, 
#                   names=('time','raw_flux', 'norm_flux', 'model_flux'))

#time = BJD-2454833
#MJD = JD - 2400000.5

#plt.figure()
#plt.plot(df['time'] + 2454833 - 2400000.5, df['model_flux'], c='grey', lw=0.2)

#gtime = [mjd_date, gck_time.mjd]
#gmag = np.array([16.46, 16.499])
#gflux = np.array([1, 10**((gmag[1] - gmag[0]) / (-2.5))])
#gerr = np.abs(np.array([0.01, 0.006]) * np.log(10) / (-2.5) * gflux)

#plt.errorbar(gtime, gflux, yerr=gerr, 
#             linestyle='none', marker='o')
#plt.ylim(0.956,1.012)
#plt.xlabel('MJD (days)')
#plt.ylabel('Relative Flux')

# plt.savefig(target+'_compare.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

####################
# add in WISE
#plt.figure()
#plt.plot(df['time'] + 2454833 - 2400000.5, df['model_flux'], c='grey', lw=0.2)
#plt.errorbar(gtime, gflux, yerr=gerr, 
#             linestyle='none', marker='o')
# the WISE W1-band results from another notebook
#wise_time = np.array([55330.86838, 55509.906929000004])
#wise_flux = np.array([ 1.,0.98627949])
#wise_err = np.array([ 0.02011393,  0.02000256])
#plt.errorbar(wise_time, wise_flux, yerr=wise_err,
#             linestyle='none', marker='o')
#plt.ylim(0.956,1.025)
#plt.xlabel('MJD (days)')
#plt.ylabel('Relative Flux')

#plt.ylim(0.956,1.012)
# plt.ylim(0.9,1.1)


#plt.savefig(target+'_compare.pdf', dpi=150, bbox_inches='tight', pad_inches=0.25)

#print('gflux: ', gflux, gerr)



