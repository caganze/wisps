from astropy.table import Table, join
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import sys
import matplotlib
from astropy.coordinates import SkyCoord
from astropy import units as u
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('text', usetex=True)
plt.rc('legend', fontsize=8)
plt.rc('axes', labelsize=12)



d2as  = 3600.

T0 = Table.read('../Gaia/Gaia_matches_DR2.tsv', format='ascii', delimiter='|', comment='#')
print T0.colnames
print len(T0)
print len(set(T0['Source']))
#T = T0[np.where( (T0['Plx'].filled(-9999) != -9999) & (T0['Plx'].filled(-9999) > 1) )]
T = T0[np.where( (T0['Plx'].filled(-9999) != -9999) & (T0['Plx'].filled(-9999)/T0['e_Plx'].filled(-9999) > 3) )]


#import collections
#print [item for item, count in collections.Counter(T['Source']).items() if count > 1]
print ''

#T2 = Table.read('Catalogs5/LaTeMoVeRS_v0_9_2.hdf5') # open an HDF5 file

T02 = Table.read('../Catalogs5/LaTeMoVeRS_v0_9_2.hdf5') # open an HDF5 file
#T22 = Table.read('../Catalogs5/GOOD_OBJIDS.csv')
T22 = Table.read('../Catalogs5/GOOD_OBJIDS_WiseParallax.csv')
T2 = join(T22, T02, keys='SDSS_OBJID')
print T2.colnames
print 'LaTE-MoVERS', len(T02), len(T22), len(T2)
print len(T2[np.where( np.sqrt(T2['PMRA']**2 + T2['PMDEC']**2) > 2.*np.sqrt(T2['PMRA_TOTERR']**2 + T2['PMDEC_TOTERR']**2) )])
print ''

j = np.where( (T2['W2MPRO'] > 14) & (T2['IMAG'] - T2['ZMAG'] > 2.5) ) # very late things with Gaia DR2 discrepant parallaxes (giants?)
j = np.where( (T2['W2MPRO'] > 17) ) # very earlish faint things
j = np.where( (T2['IMAG'] - T2['ZMAG'] > 2.3) ) # very earlish faint things
print 'Object:', T2['SDSS_OBJID'][j].data
print 'PMRA:', T2['PMRA'][j].data
print 'PMRAERR:', T2['PMRA_TOTERR'][j].data
print 'PMDEC:', T2['PMDEC'][j].data
print 'PMDECERR:', T2['PMDEC_TOTERR'][j].data
print 'DIST:', T2['DIST_AVE'][j].data
print 'DISTERR:', T2['DISTERR_AVE'][j].data
print 'i-z:', T2['IMAG'][j].data - T2['ZMAG'][j].data
print 'i:', T2['IMAG'][j].data
print 'W2:', T2['W2MPRO'][j].data 


#sys.exit()


#gaia = SkyCoord(ra=T['RA_ICRS']*u.degree, dec=T['DE_ICRS']*u.degree)  
gaia = SkyCoord(ra=T['RA_ICRS']*u.degree, dec=T['DE_ICRS']*u.degree)  
#gaia = SkyCoord(ra=c1*u.degree, dec=c2*u.degree)  
late = SkyCoord(ra=T2['SDSS_RA']*u.degree, dec=T2['SDSS_DEC']*u.degree) 
idx, d2d, d3d = gaia.match_to_catalog_sky(late) 

"""
bins = int(np.sqrt(len(T2)))
#bins = np.arange(1, 3.5, 0.05)
plt.hist(T2['W1MPRO'], histtype='step', bins=bins, label=r'LaTE-MoVeRS', log=True, color='0.5', alpha=0.5)
plt.hist(T2['W1MPRO'][idx], histtype='step', bins=bins, label=r'Matches to $Gaia$', log=True, color='b', alpha=0.5)
plt.legend()
plt.show()
#sys.exit()
"""


#fig = plt.figure(1, figsize=(3.4, 3.4*3/4.))
fig = plt.figure(1, figsize=(6, 5*3/4.))
ax = fig.add_subplot(111)

print fig.get_size_inches()


step1, step2 = 0.2, 0.5
COLOR = np.arange(1, 3+step1, step1)
MAG   = np.arange(9, 18+step2, step2)
#VALUE = []
#VALUE = np.zeros((len(COLOR), len(MAG)))
VALUE = np.zeros((len(MAG), len(COLOR)))
#VALUE3 = np.zeros((len(MAG), len(COLOR))) - 9999
Mask1 = np.zeros((len(MAG), len(COLOR)))

#Colors     = []
#Magnitudes = []
#HexVals    = []
Percent1 = []
Percent2 = []
NumberT  = []

for i in range(len(COLOR)):
	for j in range(len(MAG)):
		len1 = len(T2[np.where( (T2['IMAG'] - T2['ZMAG'] > COLOR[i]) & 
			                    (T2['IMAG'] - T2['ZMAG'] <= COLOR[i]+step1) &
			                    (T2['W2MPRO'] > MAG[j]) & (T2['W2MPRO'] <= MAG[j]+step2) ) ] )
		len2 = len(T2[idx][np.where( (T2['IMAG'][idx] - T2['ZMAG'][idx] > COLOR[i]) & 
			                         (T2['IMAG'][idx] - T2['ZMAG'][idx] <= COLOR[i]+step1) &
			                         (T2['W2MPRO'][idx] > MAG[j]) & (T2['W2MPRO'][idx] <= MAG[j]+step2) ) ] )

		if len1 == 0:
			#VALUE.append(np.nan)
			VALUE[j][i] = np.nan
			Mask1[j][i] = 1
			#Colors.append(COLOR[i]+step1)
			#Magnitudes.append(MAG[j]+step2)
			#HexVals.append(np.nan)
		else:
			#VALUE.append( float(len2) / float(len1) )
			VALUE[j][i] = float(len2) / float(len1)
			#VALUE3[j][i] = float(len2) / float(len1)
			#Colors.append(COLOR[i]+step1)
			#Magnitudes.append(MAG[j]+step2)
			#HexVals.append(float(len2) / float(len1))
			if MAG[j] <= 20: 
				Percent1.append(float(len2) / float(len1))
				NumberT.append(len1)
			if MAG[j] > 20: 
				Percent2.append(float(len2) / float(len1))

			print COLOR[i], MAG[j], len1, float(len2) / float(len1)

			#if COLOR[i] == 2.5: print i, j, COLOR[i], MAG[j], len1, len2, float(len2) / float(len1)
			if float(len2) / float(len1) < 0.5: plt.text(COLOR[i]+step1/2., MAG[j]+step2/2., '%s'%len1, ha='center', va='center', fontsize=7, color='w')
			else: plt.text(COLOR[i]+step1/2., MAG[j]+step2/2., '%s'%len1, ha='center', va='center', fontsize=7, color='k')
			#ax.text(COLOR[i]+step1/2., MAG[j]+step2/2., '%s'%len1, ha='center', va='center', fontsize=7, color='w')
			#plt.text(COLOR[i]+step1/2., MAG[j]+step2/2., '%0.3f'%(float(len2) / float(len1)), ha='center', va='center', fontsize=7, color='w')
"""
Percent1 = np.array(Percent1)
NumberT  = np.array(NumberT)
print np.mean(Percent1), np.median(Percent1), np.std(Percent1)
Percent2 = np.array(Percent2)
print np.mean(Percent2), np.median(Percent2), np.std(Percent2)
Percent3 = 0
for n,m in zip(Percent1, NumberT):
	Percent3 += n * m
print Percent3 / float(np.sum(NumberT))
fig3 = plt.figure(4)
plt.hist(Percent1, bins=int(np.sqrt(len(Percent1))), histtype='step', label='i less 20')
plt.hist(Percent2, bins=int(np.sqrt(len(Percent2))), histtype='step', label='i greater 20')
plt.legend()
plt.show()
sys.exit()
"""
VALUE2 = np.ma.array(VALUE, mask=Mask1)
#plt.hexbin(Colors, Magnitudes, C=HexVals)
#cax = ax.pcolormesh(COLOR, MAG, VALUE2, vmin=0, vmax=1, cmap='copper')
cax = ax.pcolormesh(COLOR, MAG, VALUE2, vmin=0, vmax=1, cmap='cubehelix')
plt.axis('tight')
#axis = plt.gca()
#axis.set_aspect('equal')
#plt.pcolormesh(COLOR, MAG, VALUE2, vmin=0, vmax=1, cmap='magma')
#plt.pcolor(COLOR, MAG, VALUE3, vmin=0, vmax=1, cmap='copper')

#fig.tight_layout()

ymin, ymax = plt.ylim()

xp = np.linspace(1, 2.9)
m10 = 7.13 + 4.88*xp
m100 = m10 + 5*np.log10(100) - 5
m200 = m10 + 5*np.log10(200) - 5

#plt.plot(xp, m10, 'r--')
#plt.plot(xp, m100, 'r:')
#plt.plot(xp, m200, 'r-.')

###### Distance polynomial
rz1 = np.linspace(1, 3)
Mi = 7.13 + 4.88*rz1
imags  = 5*np.log10(20)  - 5 + Mi
imags1 = 5*np.log10(100) - 5 + Mi
imags2 = 5*np.log10(400) - 5 + Mi
imagsT = 5*np.log10(50) - 5 + Mi

#ax.plot(rz1, imags,  'r--', lw=1)
#ax.plot(rz1, imags1, 'r--', lw=1)
#ax.plot(rz1, imags2, 'r--', lw=1)
#ax.plot(rz1, imagsT, 'r--', lw=1)

#ax.text(2.7, 21.75,'20 pc', fontsize=8, weight='extra bold', color='r')
#ax.text(2.3, 21.75,'50 pc', fontsize=8, weight='extra bold', color='r')
#ax.text(2.0, 21.75,'100 pc', fontsize=8, weight='extra bold', color='r')
#ax.text(1.4, 21.75,'400 pc', fontsize=8, weight='extra bold', color='r')

#plt.plot([1,3], [20,20], 'r:', lw=1)
#plt.text(2.6, 19.9,'Gaia limit', fontsize=6, weight='extra bold', color='r')
######

ax.minorticks_on()

ax.set_ylim(ymax, ymin)
cbar = plt.colorbar(cax)
cbar.set_label('Fraction of LaTE-MoVeRS Sources' + '\n' + 'with Parallaxes in $Gaia$ DR2')
cbar.set_clim(0, 1.2)

ax.set_xlabel(r'$i-z$')
ax.set_ylabel(r'$W2$')


######### Add in the Gaia magnitude limits
#small bins
G17 = np.poly1d(np.array([-4.19392834, 18.94985407]))
G20 = np.poly1d(np.array([-2.74899347, 19.00442341]))

#big bins
G17 = np.poly1d(np.array([-1.71726951, 14.00196331]))
G19 = np.poly1d(np.array([-1.71726951, 14.00196331+2]))
G20 = np.poly1d(np.array([-1.71726951, 14.00196331+3]))
G21 = np.poly1d(np.array([-1.71726951, 14.00196331+4]))

Xs = np.linspace(1,3)
#ax.plot(Xs, G17(Xs), 'b--', lw=1)
#ax.text(2.42, 9.88, r'$G \approx 17$', color='b', fontsize=8, rotation=14, zorder=100)
ax.plot(Xs, G19(Xs), 'b--', lw=1.2)
#ax.text(2.41, 11.88, r'$G \approx 19$', color='b', fontsize=8, rotation=14, zorder=100)
ax.text(2.6, 11.1, r'$G \approx 19$', color='b', fontsize=8, rotation=14, zorder=100)
#ax.plot(Xs, G20(Xs), 'b-.', lw=1)
#ax.text(2.41, 12.88, r'$G \approx 20$', color='b', fontsize=8, rotation=14, zorder=100)
ax.plot(Xs, G21(Xs), 'b:', lw=1.2)
ax.text(2.41, 13.88, r'$G \approx 21$', color='b', fontsize=8, rotation=14, zorder=100)
######### Add in the Gaia magnitude limits

######### Add in incompleteness region
#ax.axvline(2, c='r', ls='--')
#ax.axhline(11.25, c='r', ls='--')
#ax.plot([2, 2], [11, 18], c='r', ls=':', lw=1)
#ax.plot([2, 3], [11, 11], c='r', ls=':', lw=1)
ax.plot([2, 2], [G19(2), 18], c='r', ls=':', lw=1)
ax.plot([2, 3], [G19(2), G19(3)], c='r', ls=':', lw=1)
ax.text(2.5, 17, r'$Gaia$'+'\n'+'Incompletness'+'\n'+'Region', color='r', fontsize=9, ha='center', va='center')
######### Add in incompleteness region

######### Add in distance limits
w2conv    = np.poly1d([0.48737076, 3.08898142])
rz1       = np.linspace(1, 3)
Mi        = 7.13 + 4.88*rz1
imags5    = 5*np.log10(5) - 5 + Mi
imags10   = 5*np.log10(10) - 5 + Mi
imags20   = 5*np.log10(20) - 5 + Mi
imags100  = 5*np.log10(100) - 5 + Mi
imags200  = 5*np.log10(200) - 5 + Mi
W2mags5   = w2conv(imags5)
W2mags10  = w2conv(imags10)
W2mags20  = w2conv(imags20)
W2mags100 = w2conv(imags100)
W2mags200 = w2conv(imags200)

ax.plot(rz1, W2mags10, 'm--', lw=1, zorder=10)
ax.text(2.73, 13.4, r'$d \approx 10~pc$', color='m', fontsize=8, rotation=-24, zorder=100)
ax.plot(rz1, W2mags20, 'm:', lw=1, zorder=10)
ax.text(2.73, 14.2, r'$d \approx 20~pc$', color='m', fontsize=8, rotation=-24, zorder=100)
ax.plot(rz1, W2mags5, 'm:', lw=1, zorder=10)
ax.plot(rz1, W2mags100, 'm:', lw=1, zorder=10)
ax.plot(rz1, W2mags200, 'm:', lw=1, zorder=10)
######### Add in distance limits

#plt.tight_layout()

ax2 = ax.twiny() # now, ax3 is responsible for "top" axis and "right" axis
colors = [1.17, 1.48, 1.66, 1.86, 2.16, 2.43, 2.81]
ax2.set_xticks( colors )
ax2.set_xticklabels(["M7", "M8", "M9", "L3",
	                 "L5", "L6", "L8"])
#ax2.set_yticklabels([])
ax2.set_xlim(1, 3)

print fig.get_size_inches()

#fig.tight_layout()

#plt.savefig('Gaia_W2_iz_2.pdf')
plt.savefig('Gaia_W2_iz_DR2plx.png', dpi=600, bbox_inches='tight')
plt.savefig('Gaia_W2_iz_DR2plx.pdf', dpi=600, bbox_inches='tight')

plt.show()
