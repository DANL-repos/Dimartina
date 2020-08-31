import os, argparse, sys
import dwi_libs as dl

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s/n' % message)
        self.print_help()
        sys.exit(2)
        
parser = MyParser(prog='eyetracking processing')

parser.add_argument('-project', '--source_dir', help='The project folder containing sourceData, etc. (Required)', required=True, dest='project_directory')
parser.add_argument('-sublist', '--subject_list', help='File name listing subjects to process; defaults to "subject_list.txt" in the working directory.', default='subject_list.txt')
parser.add_argument('-template', '--protocol_template', help='Directory for tracking templates; defaults to "tracking_templates" in project dir', dest='protocol_template')
parser.add_argument('-coins', '--COINS_BIDS', help='Path to selected_scans.csv', default='selected_scans.csv', required=True, dest='COINS_BIDS')
parser.add_argument('--bids_dir', help='BIDs directory', required=True)

args = parser.parse_args()

# In[Set Up Directories]:

project_dir = os.path.abspath(os.path.expanduser(args.project_directory))

# In[Generate SubList]:

sublist = [fil.strip() for fil in open(args.subject_list)]

# In[Process]:

for sub in sublist:
	print(sub.upper().strip())
	dti = dl.DtiObject()
	dti.project_directory = project_dir
	dti.subid = sub.strip()
	dti.protocol_template = args.protocol_template
	dti.load_data(args.COINS_BIDS, args.bids_dir)
