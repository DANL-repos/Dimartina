"""

-find nifti files
-make QC folder
-nifti to nrrd (save to QC folder)
-DTIPrep - output QC documents
-FSL Motion Outliers - output QC images

"""

import numpy as np
import scipy.signal as signal
import os.path as path
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

class DtiData:

	def __init__(self):
		self.qc_output = None
		self.fd = np.nan
		self.project_directory = ''
		self.subid = ''
		self.corrected = np.nan
		self.rejected = np.nan
		self.total_count = None
		self.rejected_count = None
		self.rejected_nums = []
		self.corrected_nums = []

class DtiObject:

	def __init__(self):
		self.subid = ''
		self.protocol_template = ''

	def load_data(self, infile, bids_dir):
		
		data = DtiData()
		data.subid = self.subid	
		
		error_folder = infile.split('/')
		error_folder = '/'.join(error_folder[:-1])
		
		source_dir = self.project_dir

		sub_folder = os.path.join(source_dir, self.subid)
		dicom_folder = os.path.join(sub_folder, 'originals')
		qc_folder = os.path.join(sub_folder, 'QC')
		if not os.path.exists(qc_folder):
			os.makedirs(qc_folder)

		bids_file = os.path.join(bids_dir, self.subid, 'dwi', '{}_dir-AP_dwi.nii.gz'.format(self.subid))
		if not os.path.exists(bids_file) and self.subid[-1].isalpha():
			bids_file = os.path.join(bids_dir, self.subid[:-1], 'dwi', '{}_dir-AP_dwi.nii.gz'.format(self.subid[:-1]))
			

		df = pd.read_csv(infile)
		df = df.set_index('Scan_Subject_ID')

		try:
			dicom = df['DIFF_137_AP'][self.subid.lstrip('sub-')]
			dti_dicom = os.path.join(dicom_folder, dicom)
			dti_nrrd_path = os.path.join(qc_folder, 'dti')
			if not os.path.exists(dti_nrrd_path):
				os.makedirs(dti_nrrd_path)
			dti_nrrd = os.path.join(dti_nrrd_path, 'dwi.nrrd')
			if dicom == '0':
				dicom = dicom / 2

			try:
				convert_to_nrrd(dti_dicom, dti_nrrd)
			except:
				print('{}: conversion error'.format(self.subid))
				f = open(error_folder + '/error_log.txt', 'a')
				f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-dwi-coins', self.subid, 'conversion to nrrd failure'))
				f.close()

			try:
				run_dtiprep(dti_nrrd, self.protocol_template, dti_nrrd_path)
				pass
			except:
				print('{}: DTIPrep run error'.format(self.subid))
				f = open(error_folder + '/error_log.txt', 'a')
				f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-dwi-coins', self.subid, 'DTIPrep run error'))
				f.close()

			try:
				data.fd = run_fsl_qc(bids_file, dti_nrrd_path)
				rejected_nums, corrected_nums = run_meanfd_qc(data, 3)
				data.rejected_nums = rejected_nums
				data.corrected_nums = corrected_nums
				self.plot_fsl_graphs(data, dti_nrrd_path)
			except IOError:
				print('{}: FD calc error'.format(self.subid))
				f = open(error_folder + '/error_log.txt', 'a')
				f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-dwi-coins', self.subid, 'Mean FD Calculation Error'))
				f.close()

			try:
				qc_results = os.path.join(dti_nrrd_path, 'dwi_XMLQCResult.xml')
				data.rejected_count, data.total_count, rejected_nums, corrected_nums, data.rejected, data.corrected = parse_dti_qc(qc_results, data)
				data.rejected_nums.extend(rejected_nums)
				data.corrected_nums.extend(corrected_nums)
				if len(data.rejected_nums) > 0:
					data.rejected_nums = list(set(data.rejected_nums))
				if len(data.corrected_nums) > 0:
					data.corrected_nums = list(set(data.corrected_nums))
				self.plot_dtiprep_graphs(data, dti_nrrd_path)
			except IOError:
				print('{}: Parse QC error'.format(self.subid))
				f = open(error_folder + '/error_log.txt', 'a')
				f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-dwi-coins', self.subid, 'DTIPrep QC Parse Error'))
				f.close()

			try:
				qc_csv = os.path.join(error_folder, 'dti_qc.csv')
				self.save_qc_csv(data, qc_csv)
			except IOError:
				print('{}: Save CSV error'.format(self.subid))
				f = open(error_folder + '/error_log.txt', 'a')
				f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-dwi-coins', self.subid, 'Save CSV error'))
				f.close()

		except IOError:
			print('No diffusion data for any subjects listed')
			f = open(error_folder + '/error_log.txt', 'a')
			f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-dwi-coins', self.subid, 'no diffusion data'))
			f.close()

	def plot_dtiprep_graphs(self, dti_data, output_dir):
		rejected = dti_data.rejected_count
		total = dti_data.total_count
		rejected_vectors = dti_data.rejected
		corrected_vectors = dti_data.corrected

		pie_path = os.path.join(output_dir, dti_data.subid + '_dtidonut.png')

		plt.pie([rejected, total-rejected], labels=['rejected', 'remaining'], colors=['orange', 'lightgreen'], autopct='%1.1f%%', pctdistance=.85)
		my_circle = plt.Circle((0,0), .7, color = 'white')
		p = plt.gcf()
		p.gca().add_artist(my_circle)
		plt.title('Gradients Post-Processing: {}'.format(dti_data.subid))
		plt.savefig(pie_path)
		plt.close()

		vectors_path = os.path.join(output_dir, dti_data.subid + 'dtigradients.png')

		fig = plt.figure()
		ax = Axes3D(fig)

		for i in range(len(rejected_vectors[0])):
			xline = np.linspace(0, rejected_vectors[0][i], 2)
			yline = np.linspace(0, rejected_vectors[1][i], 2)
			zline = np.linspace(0, rejected_vectors[2][i], 2)
			line1 = ax.plot3D(xline, yline, zline, 'orange')

		for i in range(len(corrected_vectors[0])):
			xline = np.linspace(0, corrected_vectors[0][i], 2)
			yline = np.linspace(0, corrected_vectors[1][i], 2)
			zline = np.linspace(0, corrected_vectors[2][i], 2)

			line2 = ax.plot3D(xline, yline, zline, 'green', alpha = .6)

		green_line = mlines.Line2D([],[], color='green', label='remaining')
		orange_line = mlines.Line2D([],[], color='orange', label='rejected')
		plt.legend(handles=[green_line, orange_line])
		plt.title('Gradients Post-Processing: {}'.format(dti_data.subid))
		plt.savefig(vectors_path)
		plt.close()

	def plot_fsl_graphs(self, dti_data, output_dir):
		pngpath = os.path.join(output_dir, dti_data.subid + '_dtimeanfd.png')

		data = dti_data.fd

		maximum = data.max()
		minimum = data.min()
		ran = int(maximum - minimum)
		
		plt.hist(data, bins=ran)
		plt.title('Mean FrameWise Displacement')
		plt.savefig(pngpath, dpi=300)
		plt.close()

	def save_qc_csv(self, dti_data, output_file):
		fd = dti_data.fd

		mean_fd = fd.mean()
		rejected = dti_data.rejected_count
		total = dti_data.total_count
		percent_fail = round((float(rejected)/float(total)) * 100, 2)
		percent_pass = 100 - percent_fail

		line = '{}, {}, {}, {}, {}\n'.format(self.subid, mean_fd, rejected, total, percent_pass)
		header = 'subid, mean_fd, rejected_gradients, total_gradients, percent_pass\n'

		if not os.path.exists(output_file):
			f = open(output_file, 'w')
			f.write(header)
			f.write(line)
			f.close()
		else:
			f = open(output_file, 'a')
			f.write(line)
			f.close()

def run_meanfd_qc(data, thr):
	fd = data.fd

	rejected_nums = []
	corrected_nums = []

	for x in range(len(fd)):
		if fd[x] > thr:
			rejected_nums.append(x)
		else:
			corrected_nums.append(x)

	return rejected_nums, corrected_nums

		

def convert_to_nrrd(input_file, output_file):
	cmd = 'DWIConvert -i {} -o {}'.format(input_file, output_file)
	os.system(cmd)

def run_dtiprep(input_file, protocol_file, output_dir):
	cmd = 'DTIPrep --DWINrrdFile {} --xmlProtocol {} --check --outputFolder {}'.format(input_file, protocol_file, output_dir)
	os.system(cmd)

def parse_dti_qc(input_file, data):
	f = open(input_file)
	lines = f.readlines()

	rejected = 0
	total = 0

	rejected_gradients_x = []
	rejected_gradients_y = []
	rejected_gradients_z = []

	corrected_gradients_x = []
	corrected_gradients_y = []
	corrected_gradients_z = []

	rejected_gradients = []
	corrected_gradients = []

	for index, line in enumerate(lines):
		if '<entry parameter="gradient_' in line:
			gradient_num = line.split('_')
			gradient_num = gradient_num[-1]
			gradient_num = gradient_num[:4]
			gradient_num = int(gradient_num)
			check_line = lines[index+1]
			gradient_line = lines[index+9]
			gradient = gradient_line.strip()
			gradient = gradient.strip('<value>/')
			gradient = gradient.split(' ')
			if 'EXCLUDE' in check_line or gradient_num in data.rejected_nums:
				rejected += 1
				total += 1
				rejected_gradients_x.append(float(gradient[0]))
				rejected_gradients_y.append(float(gradient[1]))
				rejected_gradients_z.append(float(gradient[2]))
				rejected_gradients.append(gradient_num)
			else:
				total += 1
				corrected_gradients_x.append(float(gradient[0]))
				corrected_gradients_y.append(float(gradient[1]))
				corrected_gradients_z.append(float(gradient[2]))
				corrected_gradients.append(gradient_num)
		
	return rejected, total, rejected_gradients, corrected_gradients, [rejected_gradients_x, rejected_gradients_y, rejected_gradients_z], [corrected_gradients_x, corrected_gradients_y, corrected_gradients_z]


def run_fsl_qc(input_file, output_dir):
	output_txt = os.path.join(output_dir, 'tmp.txt')
	output_txt_2 = os.path.join(output_dir, 'mean_fd.txt')
	cmd = 'fsl_motion_outliers -i {} -o {} -s {} --fd'.format(input_file, output_txt, output_txt_2)
	os.system(cmd)

	data = np.genfromtxt(output_txt_2)
	return data

	
	
	
