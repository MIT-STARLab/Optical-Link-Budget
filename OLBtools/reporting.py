import datetime
import sys
import os
import numpy as np
from .. import OLBtools as olb
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import io
import base64

import dataclasses

class Report:
    def __init__(self,title:str='Link Budget'):
        self.title = title
        self.body = ''

    def AddHtml(self, html):
        self.body += '<p>' + html + '</p>'

    def AddFigure(self, fig:Figure, alt_text:str=''): 
        ''' Save the current figure as a base64 embeded in html'''
        image_string = io.BytesIO()
        fig.savefig(image_string, format='jpg')
        image_string.seek(0)
        image_base64 = base64.b64encode(image_string.read()).decode()

        self.AddHtml('<img src="data:image/png;base64,%s", alt="%s"/>' % (image_base64,alt_text))

    def HtmlString(self):

        report_header = f'''
            <h1>{self.title}</h1>
        '''      
        script_file = sys.argv[0]
        scrict_name = os.path.basename(script_file)
        script_path = os.path.dirname(script_file)

        try:
            import git
            repo = git.Repo(script_path)
        except:
            git_string = ''
        else:
            commit_hash = repo.head.commit.hexsha
            changed = [item.a_path for item in repo.index.diff(None)]
            if scrict_name   in repo.untracked_files: file_status = 'file is not tracked'
            elif scrict_name in changed: file_status = 'file changed'
            else: file_status = ''
            if repo.is_dirty(untracked_files=True): file_status += ', repos is dirty'
            git_string = '<br>Git: %s, %s' % (commit_hash,file_status)
                
        report_footer = f'''
            <p>
            File: {scrict_name} {git_string}
            <br>
            Generated on {str(datetime.datetime.today())}</p>
        '''

        return REPORT_STYLE + report_header + self.body + report_footer

    def AsPDF(self, pdf_file_name='link_budget.pdf'):
        html_rep = self.HtmlString()

        import pdfkit
        # Todo: find a more flexible option
        config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
        pdfkit.from_string(html_rep, pdf_file_name, configuration=config)

    def AsHtmlPage(self, html_file_name='link_budget.html'):
        html_rep = self.HtmlString()
        with open(html_file_name,'w') as html_file: html_file.write(html_rep)

    def FormatSI(self, value:float):
        value_range =  [1e0,1e3,1e6,1e9,1e12]
        prefix = ['m', '', 'K', 'M', 'G', 'T']

        index = 0
        while value > value_range[index]:
            index+=1
            if index == len(value_range): break
        value = value / value_range[index-1]
        prefix[index]

        return f'{value:.4g} {prefix[index]}'



class Table:
    def __init__(self, content:list, colapse_identical_row:bool=False):
        self.content = content
        self.colapse_identical_row = colapse_identical_row

    def ColapseRow(self,row:list[str]):
        if self.colapse_identical_row:

            last = row[0]
            row_html = ''   
            span_counter = 1

            def comit_last(row_html):
                if span_counter > 1: #More than one span, use the html attribute
                    row_html += '<td colspan="%d" class="fused">%s</td>' % (span_counter, last)
                else: row_html += '<td>%s</td>' % (last)
                return row_html

            for item in row[1:]:
                if item == last: # New item is identical to previous one:
                    span_counter += 1 # We need the span to be longer

                else: # new item is different, comit the span using last
                    row_html = comit_last(row_html)               
                    span_counter = 1 # back to single span
                    last = item # current item is saved for later, next span comit

            #   Last item need to be added
            row_html = comit_last(row_html)

        else: 
            row_html = ''.join(['<td>%s</td>' % col for col in row])

        return row_html


    def Html(self,html_class='time') -> str:
        self.html_rep = ''
        for row in self.content:
            self.html_rep += '<tr>' + self.ColapseRow(row) + '</tr>'
        self.html_rep = '<table class="%s">' % html_class + self.html_rep + '</table>'
        return self.html_rep

class CapacityVsRange(Report):

    @dataclasses.dataclass
    class InputsCase:
        range_start       : float
        range_end         : float
        optical_power     : float
        wavelength        : float
        beam_divergence   : float
        receive_diameter  : float
        APD_gain          : float          
        APD_responsivity  : float
        APD_bandwidth     : float
        APD_excess_noise  : float
        APD_dark_current  : float
        APD_current_noise : float       = 0.0
        target_BER        : float       = 1e-5
        M2_factor         : float       = 1.0
        pointing_error    : float       = 0.0
        datarate_to_BW    : float       = 1.0
        db_losses         : list[float] = dataclasses.field(default_factory=list)
        nametag           : str         = ''
        plot_legend       : str         = ''

        def textList(self) -> list[str]:
            return [
                self.nametag,
                f'{self.range_start/1000:,.0f} km to {self.range_end/1000:,.0f} km',
                f'{self.optical_power*1000:,.0f} mW',
                f'{self.wavelength*1e9:.02f} nm',
                f'{self.beam_divergence*1e6:.2f} urad 1/e<sup>2</sup>',
                f'{self.receive_diameter*1e3:.2f} mm',
                f'{self.APD_gain:.1f}',
                f'{self.APD_responsivity:.2f} A/W',
                f'{self.APD_bandwidth/1e6:,.0f} MHz',
                f'{self.APD_excess_noise:.2f}',
                f'{self.APD_dark_current*1e9:.2f} nA',
                f'{self.APD_current_noise*1e12:.2f} pA/rtHz',
                f'10E{np.log10(self.target_BER):.1f}',
                f'{self.pointing_error*1e6:.2f} urad',
                f'{self.M2_factor:.2f}',
                f'{self.datarate_to_BW:.2f} b/Hz',
            ]
        
        def copy(self) -> 'CapacityVsRange.InputsCase': return dataclasses.replace(self)

    def __init__(self,cases:list[InputsCase] = [], n_points=1000, **karg):
        super().__init__(**karg)
        self.cases = cases
        self.n_points = n_points

    def AddCase(self, case:InputsCase):
        self.cases.append(case)
        if case.nametag == '':  case.nametag = f'Case {len(self.cases)}'

    def MakeCasesTable(self) -> 'Table':
        NAME_LIST = [
            'Case Name',
            'Range',
            'Optical power',
            'Wavelength',
            'Beam divergence',
            'Receiver diameter',
            'APD gain',
            'APD responsivity',
            'APD bandwidth',
            'APD excess noise ratio',
            'APD dark current',
            'APD amplifer noise density',
            'Target uncoded BER',
            'Static pointing error',
            'M2 quality factor',
            'Throughput to Bandwidth ratio',
        ]

        param_list = list(zip(NAME_LIST, *[case.textList() for case in self.cases]))

        losses = {}

        for case_index in range(len(self.cases)):

            case = self.cases[case_index]

            for loss_name, loss_value in case.db_losses:

                if not loss_name in losses: losses[loss_name] = [0]*len(self.cases)
                
                losses[loss_name][case_index] = loss_value

        param_list += [[key]+[f'{v:.1f} dB' for v in vals] for key, vals in losses.items()]
            

        return Table(param_list, colapse_identical_row=1)
    
    def ConcatenateCases(self):
        '''Copy over the fileds from the case list, and concatenate them as numpy arrays using the second dimention as case index'''
        
        for fd in dataclasses.fields(self.InputsCase):

            thing_list = [getattr(case, fd.name) for case in self.cases]

            if fd.name in ('nametag','plot_legend'): # The names are just concatenated ins normal python lists
                self.__setattr__(fd.name, thing_list)
            elif fd.name == 'db_losses': # Losses needs to be added up first
                db_losses = [sum(tuple(zip(*loss_list))[1]) for loss_list in thing_list]
                self.db_losses =  np.array(db_losses)
            else: # it's a value that should be a numpy array, wioth second axis as case index
                self.__setattr__(fd.name, np.array(thing_list))

        #self.range_start = np.array([case.range_start for case in self.cases])[np.newaxis,:]

    def AddParametersTable(self): self.AddHtml('<h2>Parameters</h2>' + self.MakeCasesTable().Html())

    def Run(self,ranges=None):

        # Generate the inut arrays
        self.ConcatenateCases()
        
        # Log scale link ranges vector
        if not ranges: link_range = np.logspace(np.log10(self.range_start),np.log10(self.range_end),self.n_points)
        else: link_range = np.array(ranges)[:,np.newaxis]

        # Position error at receiver
        r = np.tan(self.pointing_error)*link_range

        # Beam waist
        W_0 = olb.divergence_to_radius(self.beam_divergence,self.wavelength)

        # Angular wave number, = 2*pi/lambda
        k = olb.angular_wave_number(self.wavelength)

        # Range loss for a gaussian beam
        range_loss = olb.path_loss_gaussian(W_0, self.wavelength, link_range, self.receive_diameter, r, self.M2_factor)

        # Addingup all losses
        all_losses = range_loss-self.db_losses
        
        # Received power
        P_rx_avg = self.optical_power*10**(all_losses/10)

        # Declare photodetector
        pd = olb.Photodiode(
            gain=self.APD_gain  ,
            responsivity=self.APD_responsivity,
            bandwidth=self.APD_bandwidth,
            excess_noise_factor=self.APD_excess_noise,
            dark_current=self.APD_dark_current,
            amp_noise_density=self.APD_current_noise)

        # Estimate required bandwidth for givien BER
        detector_bw = olb.suported_bandwidth_OOK(pd,P_rx_avg,self.target_BER)

        # Supported bandwidth is at most hardware bandwidth
        detector_bw = olb.filter_maximum(detector_bw, self.APD_bandwidth, 1)

        # Data is up to 2 time faster than bandwidth
        datarate = self.datarate_to_BW*detector_bw

        return link_range, datarate

    def GetThroughputFigure(self) -> tuple[Figure, Axes]:

        link_range, datarate = self.Run()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        index = 0
        for case_name in self.plot_legend:
            ax.loglog(link_range[:,index]/1e3, datarate[:,index]/1e6,label=case_name)
            index +=1
        if index > 1: ax.legend()
        ax.set_xlabel('Link range, km')
        ax.set_ylabel('Throughput, Mbps')

        return fig, ax

    
    def AddSampleTable(self, link_ranges:[int]):

        _, datarate = self.Run(link_ranges)

        sample_list = []

        sample_list += [['Range'] + [case.nametag for case in self.cases]]

        range_index = 0
        for range in link_ranges:
            sample_list += [[f'{range*1e-3:,.2f} km'] + [f'{self.FormatSI(dt)}bps' for dt in datarate[range_index,:]]]
            range_index += 1


        self.AddHtml(Table(sample_list).Html())


    def GetReport(self):
        return self.rep

REPORT_STYLE = '''
<style>

html * {
    font-family: arial, sans-serif;
}

h1 {
    text-align: center;
}

table.time {
  border-collapse: collapse;
  width: 100%;
  table-layout:fixed;
}

table.time td, table.time th {
  border: 1px solid #000000;
  text-align: left;
  padding: 8px;
}

table.time td.fused {
    text-align: center;
    vertical-align: middle;
}

table.time tr:nth-child(even) {
  background-color: #dddddd;
}

footer {
    position: fixed;
    bottom: 0;
}

</style>
'''