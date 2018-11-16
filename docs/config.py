"""
My own config file to build documentation, sphinx failed me : (
"""

import pdoc
import wisps


submodules=['wisps.simulations', 'wisps.simulations.model','wisps.data_analysis.image',
	 'wisps.data_analysis.indices', 'wisps.data_analysis.photometry', 
	 'wisps.data_analysis.spex_indices', 'wisps.data_analysis.spectrum_tools', 
	 'wisps.data_analysis.selection_criteria','wisps.data_analysis.plot_spectrum']

#build index.hmtl
main_html=pdoc.html('wisps.data_analysis')
main_file=open('index.html', 'w')
main_file.write(main_html)
main_file.close()

def build_pages():
	for mdl in submodules :
		print ('building {}'.format(mdl))
		html_input=pdoc.html(mdl)
		fname=mdl.split('.')[-1]+'.m.html'
		fil=open(fname, 'w')
		fil.write(html_input)
		fil.close() #this works? idk
	return 

if __name__== '__main__':
	build_pages()