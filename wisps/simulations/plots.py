def plot_luminosity_function():
	spts=np.arange(15, 40)
	j_mags=[spe.typeToMag(splat.typeToNum(x),'2MASS J',unc=0.5) for x in spts]
	j_mags=np.array(j_mags)
	fig, (ax, ax1)=plt.subplots(1, 2, figsize=(10, 6), sharex=False)
	ax.errorbar( j_mags[:,0], spts, xerr=j_mags[:, 1], fmt='o', ms=14)

	ax.set_yticks([17.0, 20.0, 25.0, 30.0, 35.0, 39.0])
	ax.set_yticklabels(['M.7','L0.0', 'L5.0', 'T0.0', 'T5.0', 'T9.0'])
	ax.set_ylim([14, 40])


	ax.set_ylabel('SpT')
	ax1.set_yscale("log", nonposy='clip')

	m=Galaxy()
	c=ax1.scatter( j_mags[:,0], [m.luminosity_function(j) for j in j_mags[:,0]] ,
				c=spts, cmap='viridis', marker='D', s=100 )
	synthj=np.arange(8, 40, 1)
	ax1.plot( synthj, [Galaxy.luminosity_function(j) for j in synthj], c='#111111')
	ax1.set_ylim([-1, 10])
	ax1.set_xlabel('J mag')

	#ax1.set_xlim([12, 30])
	ax.set_xlim([8, 16])

	ax.set_xlabel('J mag')

	#ax1.set_xlim([12, 30])
	ax1.set_xlim([8, 16])



	ax1.set_ylabel('$\Phi(J)$ $[10^{-3} pc^{−3} mag^{−1}]$')

	plt.tight_layout()

	cbar=fig.colorbar(c)
	cbar.set_label('Spectral Type', size=19)
	cbar.ax.set_yticklabels(['L0', 'L5', 'T0', 'T5'], fontsize=15) 

	fig.savefig(wisps.OUTPUT_FIGURES+'/luminosity_function.pdf')#[m.luminosity_function(j) for j in j_mags[:,0]]