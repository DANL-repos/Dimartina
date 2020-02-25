# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:19:00 2019

@author: allan
"""

import pandas as pd
import numpy as np
import os, datetime, json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate, signal
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import butter, lfilter

# In[Helper Functions]:

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def decimate(array, end_ms_sample, cur_samp_rate):
    cur_ms_samp = 1000/cur_samp_rate
    sample_step = int(end_ms_sample / cur_ms_samp)
    return array[::int(sample_step)]

def transparent_cmap(cmap, N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, .8, N+4)
    return mycmap

def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]
    
# In[Track Data Class]:

class TrackData:
    
    def __init__(self):
        self.gazex = np.nan #
        self.gazey = np.nan #
        self.pupil = np.nan #
        self.time = np.nan #
        self.sampling_rate = np.nan
        self.org_start = np.nan
        self.org_end = np.nan
        self.org_first = np.nan
        self.org_last = np.nan
        self.startsize = np.nan
        self.endsize = np.nan
        self.blink_count = np.nan
        self.blink_rate = np.nan
        self.res = np.nan
        self.qc_metrics = []
        self.graphs = ()
        self.name = ''
        self.messages = np.nan
        self.header = np.nan
        self.saccs = np.nan
        self.fixs = np.nan
        self.blinks = np.nan
        self.std_dev = np.nan

# In[Track Object Class]:
        
class TrackObject:
    
    def __init__(self):
        self.subid = ''
        self.project_directory = ''
        self.task_list = ['face1', 'face2']
        self.target_sampling_rate = 1.25
        self.native_sampling_rate = np.nan
        self.hasloaded = False
        self.run = {}
        for task in self.task_list:
            self.run[task] = TrackData()
        
    def load_from_template(self, infile, bids_dir):
        infile = os.path.abspath(os.path.expanduser(infile))
        error_folder = infile.split('/')
        error_folder = '/'.join(error_folder[:-2])
        
        try:
            with open(infile) as json_data:
                d = json.load(json_data)
            for task in self.task_list:
                taskfile = os.path.abspath(os.path.expanduser(d[task]))
                if os.path.isfile(taskfile) and d[task] != '':
                    try:
                        print('loading: {}'.format(taskfile))
                        ascfile = create_asc(taskfile)
                        self.run[task] = parse_asc(ascfile)
                    except:
                        print('Error loading {}, {}'.format(self.subid, task))
                        f = open(error_folder + '/error_log.txt', 'a')
                        f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-tracking-coins', self.subid, 'load error'))
                        f.close()
                        return
                    print('Processing: {}, {}'.format(self.subid, task))
                    self.native_sample_rate = self.run[task].sampling_rate
                    self.run[task] = downsample(self.run[task], 4)
                    try:
                        self.run[task] = remove_blinks(self.run[task])
                        self.run[task] = trendline_filter(self.run[task])
                    except:
                        print('Error processing - data too noisy, no data remaining after filtering {}, {}'.format(self.subid, task))
                        f = open(error_folder + '/error_log.txt', 'a')
                        f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-tracking-coins', self.subid, 'processing error'))
                        f.close()
                        return
                    self.run[task] = normalize(self.run[task])
                    self.run[task] = flipy(self.run[task])
                    self.run[task] = downsample(self.run[task], 800)
                    generate_qc(self.run[task])
                elif not os.path.isfile(taskfile) and d[task] != '':
                        print('Error loading {}, {}'.format(self.subid, task))
                        f = open(error_folder + '/error_log.txt', 'a')
                        f.write('{} : {} : {} : {}\n'.format(datetime.datetime.now(), 'proc-tracking-coins', self.subid, 'no json file'))
                        f.close()
                        return
            self.hasloaded=True
            
        except IOError:
                print('No file: {}'.format(infile))
                
    def save_track_csv(self, track_data, output_dir):
        output_csv = os.path.join(output_dir, 'tracking_values.csv')
        odat = {}
        odat['subject_id'] = track_data.subid
        for task in track_data.task_list:
            odat[task+'_gazex'] = track_data.run[task].gazex
            odat[task+'_gazey'] = track_data.run[task].gazey
            odat[task+'_pupil'] = track_data.run[task].pupil
        df = pd.DataFrame([odat])
        if os.path.isfile(output_csv) is True:
            indf = pd.read_csv(output_csv)
            df = df.append(indf)
        print('Writing x/y/pup to: {}'.format(output_csv))
        df.to_csv(output_csv, index=False)
        
    def save_track_tsv(self, track_data, output_dir):
        tasklist = track_data.task_list
        subid = track_data.subid
        subid2 = track_data.subid
        tsv_name = {}
        tsv_name['face1'] = subid+'_task-faces_run-01_tracking'
        tsv_name['face2'] = subid+'_task-faces_run-02_tracking'
        track_json = """{
            "SamplingFrequency": 1.25,
            "StartTime": 0.0,
            "Columns": ["gazex", "gazey", "pupil"]
            }"""
        for task in tasklist:
            gazex = track_data.run[task].gazex
            gazey = track_data.run[task].gazey
            pupil = track_data.run[task].pupil
            if gazex is not np.nan and gazey is not np.nan and pupil is not np.nan:
                outfile = os.path.join(output_dir, subid2, 'func', tsv_name[task]+'.tsv')
                print('Writing: {}'.format(outfile))
                k = []
                k.append(track_data.run[task].gazex[:])
                k.append(track_data.run[task].gazey[:])
                k.append(track_data.run[task].pupil[:])
                p = np.array(k)
                np.savetxt(outfile, p.transpose(), delimiter='\t')
                jsonpath = os.path.join(output_dir, subid2, 'func', tsv_name[task]+'.json')
                jsonfile = open(jsonpath, 'w')
                jsonfile.write(track_json)
                jsonfile.close()

                
    def plot_qc_graphs(self, track_data, output_dir):
        tasklist = track_data.task_list
        subid = track_data.subid
        png_name = {}
        png_name['face1'] = subid+'_task-faces_run-01_tracking'
        png_name['face2'] = subid+'_task-faces_run-02_tracking'
        
        for task in tasklist:
            qc_metrics = track_data.run[task].qc_metrics
            graphs = track_data.run[task].graphs
            
            if qc_metrics != [] and graphs:
                
                gaze_plot = graphs[0]
                pupil_plot = graphs[1]
                
                pngpath = os.path.join(output_dir, png_name[task]+'_gaze.png')
                gaze_plot.savefig(pngpath, dpi=300)
                
                pngpath = os.path.join(output_dir, png_name[task]+'_pupil.png')
                pupil_plot.savefig(pngpath, dpi=300)

    
    def save_qc_csv(self, track_data, output_dir):
        tasklist = track_data.task_list
        subid = track_data.subid
        csv_name = 'tracking_qc.csv'
        csv_path = os.path.join(output_dir, csv_name)
        
        for task in tasklist:
            
            qc_metrics = track_data.run[task].qc_metrics
            
            if qc_metrics != []:
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    df = pd.DataFrame({
                           'pct_any_roi' : [],
                           'pct_valid' : [],
                           'blink_count' : [],
                           'blink_rate' : []})
                    
                qc_metrics['task'] = task
                qc_metrics['subid'] = subid
                print(qc_metrics)
                new_df = pd.DataFrame(qc_metrics)
                
                final_df = df.merge(new_df, how = 'outer')
                final_df.to_csv(csv_path, index=False)
                    
        
# In[Read Data]:
        
def create_asc(edffile):
    cmd = 'edf2asc -y {}'.format(edffile)
    os.system(cmd)
    
    taskfile = edffile.rstrip('.edf')
    taskfile += '.asc'
    
    return taskfile

def parse_asc(ascfile):
    
    data = TrackData()

    f = open(ascfile)
    asc = f.read()
    asc = asc.split('\n')
    
    asc_list = asc[:]
    
    blinks = []
    fixs = []
    saccs = []
    samples = []
    messages = []
    header = []
    event = None
    blink_count = 0

    for line in asc_list:
        l_split = line.split(' ')
        if '**' in l_split[0]:
            header.append(line)
        elif 'MSG' in l_split[0]:
            messages.append(line)
            if 'GAZE_COORDS' in l_split[1]:
                x = l_split[4].split('.')
                y = l_split[5].split('.')
                data.resolution = (int(x[0]),int(y[0]))
        elif 'SFIX' in l_split[0]:
            event = 'fix'
        elif 'SSACC' in l_split[0]:
            event = 'sacc'
        elif 'SBLINK' in l_split[0]:
            event = 'blink'
            blink_count += 1
        elif 'EFIX' in l_split[0] or 'ESACC' in l_split[0] or 'EBLINK' in l_split[0]:
            event = None
        elif is_number(l_split[0].strip('/t')):
            if event == 'fix':
                samples.append(line)
                fixs.append(True)
                blinks.append(False)
                saccs.append(False)
            elif event == 'sacc':
                samples.append(line)
                fixs.append(False)
                blinks.append(False)
                saccs.append(True)
            elif event == 'blink':
                samples.append(line)
                fixs.append(False)
                blinks.append(True)
                saccs.append(False)
            else:
                samples.append(line)
                fixs.append(False)
                blinks.append(False)
                saccs.append(False)
                
    time = []
    gazex = []
    gazey = []
    pupil = []
    
    for line in samples:
        l_split = line.split('\t')
        if is_number(l_split[0]):
            for item in range(len(l_split)):
                l_split[item] = l_split[item].strip()
                if l_split[item] == '.':
                    l_split[item] = '0'
            time.append(float(l_split[0]))
            gazex.append(float(l_split[1]))
            gazey.append(float(l_split[2]))
            pupil.append(float(l_split[3]))
        
    dif = time[1] - time[0]
    time = np.arange(0, len(time)*dif, dif)
    
    data.sampling_rate = 1000/dif
    data.org_start = time[0]
    data.org_end = time[-1]
    data.org_first = (gazex[0], gazey[0], pupil[0])
    data.org_last = (gazex[-1], gazey[-1], pupil[-1])
    
    data.blink_count = blink_count
    data.blink_rate = blink_count / ((time[-1] / 1000) / 60)
        
    d = {}
    d['time'] = time
    d['gazex'] = gazex
    d['gazey'] = gazey
    d['pupil'] = pupil
    
    df = pd.DataFrame(d)
    
    data.time = np.array(df['time'])
    data.gazex = np.array(df['gazex'])
    data.gazey = np.array(df['gazey'])
    data.pupil = np.array(df['pupil'])
    data.blinks = blinks
    data.saccs = saccs
    data.fixs = fixs
    
    return data

# In[Process Data]:
    
def flipy(data):
    gazey = data.gazey[:]
    
    c = data.resolution[1] / 2
    
    for i in range(gazey.size):
        x = c - gazey[i]
        gazey[i] = c + x
        
    data.gazey = gazey
        
    return data

def flipx(data):
    gazex = data.gazex[:]
    
    c = data.resolution[0] / 2
    
    for i in range(gazex.size):
        x = c - gazex[i]
        gazex[i] = c + x
        
    data.gazex = gazex
    
    return data

def remove_blinks(data):
    
    gazex = data.gazex[:]
    gazey = data.gazey[:]
    pupil = data.pupil[:]
    time = data.time[:]
    data.startsize = len(time)
    
    blinks = data.blinks[:]
    blinks = np.invert(blinks)
    
    data.gazex = gazex[blinks]
    data.gazey = gazey[blinks]
    data.pupil = pupil[blinks]
    data.time = time[blinks]
    
    data.endsize = len(data.time)
    data = interpolate_data(data)
    
    return data

def interpolate_data(data):
    
    gazex = data.gazex[:]
    gazey = data.gazey[:]
    pupil = data.pupil[:]
    time = data.time[:]
    
    if time[0] != data.org_start:
        
        gx = [data.org_first[0]]
        gy = [data.org_first[1]]
        p = [data.org_first[2]]
        t = [data.org_start]
        
        gx.extend(list(gazex))
        gy.extend(list(gazey))
        p.extend(list(pupil))
        t.extend(list(time))
        
        gazex = gx
        gazey = gy
        pupil = p
        time = t
        
    else:
        
        gazex = list(gazex)
        gazey = list(gazey)
        pupil = list(pupil)
        time = list(time)
        
    if time[-1] != data.org_end:
        gazex.append(data.org_last[0])
        gazey.append(data.org_last[1])
        pupil.append(data.org_last[2])
        time.append(data.org_end)
        
    gazex = np.array(gazex)
    gazey = np.array(gazey)
    pupil = np.array(pupil)
    time = np.array(time)
    
    interfunc = interpolate.interp1d(time, pupil)
    interfunc2 = interpolate.interp1d(time, gazex)
    interfunc3 = interpolate.interp1d(time, gazey)
    
    time = np.arange(data.org_start, data.org_end + (1000./data.sampling_rate), (1000./data.sampling_rate))
    
    pupil = interfunc(time)
    gazex = interfunc2(time)
    gazey = interfunc3(time)
    
    data.time = time[:]
    data.pupil = pupil[:]
    data.gazex = gazex[:]
    data.gazey = gazey[:]
    
    return data

def trendline_filter(data):
    
    gazex = data.gazex[:]
    gazey = data.gazey[:]
    pupil = data.pupil[:]
    time = data.time[:]

    #data.startsize = len(time)
    
    inter_pupil = pupil[:]
    inter_time = time[:]
    
    kernel_size = int(data.sampling_rate)
    if kernel_size % 2 == 0:
        kernel_size += 1
    trend_pupil = signal.medfilt(inter_pupil, kernel_size = kernel_size)
    
    med = np.median(trend_pupil)
    
    mad_list = []
    for i in range(trend_pupil.size):
        mad_list.append(abs(trend_pupil[i] - med))
        
    mad = np.median(mad_list)
    
    pos_limit = trend_pupil + mad
    neg_limit = trend_pupil - mad
    
    cuta = pupil > neg_limit
    cutb = pupil < pos_limit
    cutc = np.logical_and(cuta, cutb)
    
    inter_time = inter_time[cutc]
    inter_pupil = inter_pupil[cutc]
    
    if time[-1] not in inter_time:
        inter_time =  np.append(inter_time, np.array(time[-1]))
        inter_pupil =  np.append(inter_pupil, np.array(0))
    if time[0] not in inter_time:
        inter_time =  np.append(inter_time, np.array(time[0]))
        inter_pupil =  np.append(inter_pupil, np.array(0))
        
    inter_func = interpolate.interp1d(inter_time, inter_pupil)
    time = np.arange(data.org_start, data.org_end + (1000/data.sampling_rate), (1000/data.sampling_rate))
    inter_pupil = inter_func(inter_time)
    
    
    trend_pupil = signal.medfilt(inter_pupil, kernel_size=501)
    
    med = np.median(trend_pupil)
    
    mad_list = []
    for i in range(trend_pupil.size):
        mad_list.append(abs(trend_pupil[i] - med))
        
    mad = np.median(mad_list)
    
    inter_func = interpolate.interp1d(inter_time, inter_pupil)
    inter_func2 = interpolate.interp1d(inter_time, trend_pupil)
    inter_time = time
    inter_pupil = inter_func(inter_time)
    trend_pupil = inter_func2(inter_time)
    
    pos_limit = trend_pupil + (.1 * mad)
    neg_limit = trend_pupil - (.1 * mad)
    
    cutd = inter_pupil > neg_limit
    cute = inter_pupil < pos_limit
    cutf = np.logical_and(cutd, cute)
    
    gazex = gazex[cutf]
    gazey = gazey[cutf]
    pupil = pupil[cutf]
    time = time[cutf]
    
    t = 0
    for i in range(pupil.size):
        if time[i] - t >= 100 and time[i] - t <= 200:
            data.blink_count += 1
            t = time[i]
        else:
            t = time[i]
    data.blink_rate = float(data.blink_count) / float(float(time[-1]) / float(60000))
    
    data.gazex = gazex
    data.gazey = gazey
    data.pupil = pupil
    data.time = time
    #data.endsize = len(time)
    
    data = interpolate_data(data)
    
    return data

def downsample(data, new_ms_sample):
    
    gazex = data.gazex[:]
    gazey = data.gazey[:]
    pupil = data.pupil[:]
    time = data.time[:]
    fixs = data.fixs[:]
    saccs = data.saccs[:]
    blinks = data.blinks[:]
    
    sample_step = int(new_ms_sample / (1000. / data.sampling_rate))
    chunkgazex = list(chunks(gazex, sample_step))
    chunkgazey = list(chunks(gazey, sample_step))
    chunkgazex = chunkgazex[:-1]
    chunkgazey = chunkgazey[:-1]
    
    stdx = np.std(chunkgazex, axis=1)
    stdy = np.std(chunkgazey, axis=1)
    
    stdx = np.append(stdx, np.mean(stdx))
    stdy = np.append(stdy, np.mean(stdy))
    
    data.save = stdx
    
    data.std_dev = np.maximum(stdx, stdy)
    
    gazex = decimate(gazex, new_ms_sample, data.sampling_rate)
    gazey = decimate(gazey, new_ms_sample, data.sampling_rate)
    pupil = decimate(pupil, new_ms_sample, data.sampling_rate)
    time = decimate(time, new_ms_sample, data.sampling_rate)
    fixs = decimate(fixs, new_ms_sample, data.sampling_rate)
    saccs = decimate(saccs, new_ms_sample, data.sampling_rate)
    blinks = decimate(blinks, new_ms_sample, data.sampling_rate)
    
    data.sampling_rate = 1000. / new_ms_sample
    
    data.gazex = gazex
    data.gazey = gazey
    data.time = time
    data.pupil = pupil
    data.org_start = time[0]
    data.org_end = time[-1]
    data.fixs = fixs
    data.saccs = saccs
    data.blinks = blinks
    
    return data

def normalize(data):
    
    gazex = data.gazex[:]
    gazey = data.gazey[:]
    pupil = data.pupil[:]
    time = data.time[:]
    
    mean = pupil.mean()
    stddev = pupil.std()
    pupil = (pupil - mean) / stddev
    
    cut = np.logical_and(pupil < 3, pupil > -3)
    gazex = gazex[cut]
    gazey = gazey[cut]
    pupil = pupil[cut]
    time = time[cut]
    
    pupil = signal.detrend(pupil)
    
    data.gazex = gazex
    data.gazey = gazey
    data.pupil = pupil
    data.time = time
    
    data = interpolate_data(data)
    
    return data

# In[View Data]:
    
def generate_qc(data):
    
    res = data.resolution

    l_c = [(res[0] * .3), (res[1] * .65)]
    r_c = [(res[0] * .7), (res[1] * .64)]
    c_c = [(res[0] * .5), (res[1] * .33)]
    
    c = res[1]/2
    
    l_c[1] = c - l_c[1]
    l_c[1] = c + l_c[1]
    
    r_c[1] = c - r_c[1]
    r_c[1] = c + r_c[1]
    
    c_c[1] = c - c_c[1]
    c_c[1] = c + c_c[1]
    
    x = (195 * 1.875) /2
    y = (240 * 1.406) /2
    
    #ROI Locations
    l_x = [l_c[0]+x, l_c[0]+x, l_c[0]-x, l_c[0]-x, l_c[0]+x]
    l_y = [l_c[1]-y, l_c[1]+y, l_c[1]+y, l_c[1]-y, l_c[1]-y]
    
    r_x = [r_c[0]+x, r_c[0]+x, r_c[0]-x, r_c[0]-x, r_c[0]+x]
    r_y = [r_c[1]-y, r_c[1]+y, r_c[1]+y, r_c[1]-y, r_c[1]-y]
    
    c_x = [c_c[0]+x, c_c[0]+x, c_c[0]-x, c_c[0]-x, c_c[0]+x]
    c_y = [c_c[1]-y, c_c[1]+y, c_c[1]+y, c_c[1]-y, c_c[1]-y]
    
    f_x = [(res[0] / 2)+30, (res[0] / 2)+30, (res[0] / 2)-30, (res[0] / 2)-30, (res[0] / 2)+30]
    f_y = [(res[1] / 2)-50, (res[1] / 2)+20, (res[1] / 2)+20, (res[1] / 2)-50, (res[1] / 2)-50]
    
    gazex = data.gazex[:]
    gazex = gazex[data.fixs]
    gazey = data.gazey[:]
    gazey = gazey[data.fixs]
    pupil = data.pupil[:]
    pupil = pupil[data.fixs]
    weights = data.std_dev[:]
    weights = weights[data.fixs]
    time = data.time[:]
    time = time[data.fixs]
    
    total = float(gazex.size)
 
    fig, ax = plt.subplots()
    
    center_x = gazex
    center_y = gazey
    
    max_ys = [np.max(l_y), np.max(r_y), np.max(c_y), np.max(f_y)]
    max_y = np.max(max_ys)

    min_ys = [np.min(l_y), np.min(r_y), np.min(c_y), np.min(f_y)]
    min_y = np.min(min_ys)

    max_xs = [np.max(l_x), np.max(r_x), np.max(c_x), np.max(f_x)]
    max_x = np.max(max_xs)

    min_xs = [np.min(l_x), np.min(r_x), np.min(c_x), np.min(f_x)]
    min_x = np.min(min_xs)

    cutx = np.logical_and(center_x > min_x, center_x < max_x)
    cuty = np.logical_and(center_y > min_y, center_y < max_y)
    cut = np.logical_and(cutx, cuty)    

    all_roi = float(center_x[cut].size)

    per_roi = round(all_roi/total, 3) * 100

    heatmap, xedges, yedges = np.histogram2d(center_x, center_y, weights = weights, bins=1000, range=[[0, res[0]],[0,res[1]]])
    heatmap = gaussian_filter(heatmap, sigma=48)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    mycmap = transparent_cmap(cm.jet)
    g = plt.imshow(heatmap.T, extent=extent, origin = 'lower', cmap = mycmap)
    
    per_valid = round(float(data.endsize)/float(data.startsize), 2) * 100
    blink_count = data.blink_count
    blink_rate = round(data.blink_rate, 2)
    
    ax.plot(l_x, l_y, c = 'r')
    ax.plot(r_x, r_y, c = 'r')
    ax.plot(c_x, c_y, c = 'r')
    ax.plot(f_x, f_y, c = 'r')
    ax.axis([0, res[0], 0, res[1]])
    fig.suptitle('% in ROI: {}'.format(per_roi))
    cb = fig.colorbar(g, ax=ax)
    cb.set_label('% Fixations in Region')
    fig = plt.gcf()
        
    fig2, ax2 = plt.subplots()
    ax2.plot(time/1000, pupil)
    fig2.suptitle ('% valid: {}, blink count: {}, blink rate: {}'.format(per_valid, blink_count, blink_rate))
    fig2 = plt.gcf()
    
#    fig3, ax3 = plt.subplots()
#    ax3.scatter(gazex, gazey, s = weights, c=weightsb)
#    ax3.axis([0, res[0], 0, res[1]])

    
    data.qc_metrics = {'pct_any_roi' : [per_roi],
                       'pct_valid' : [per_valid],
                       'blink_count' : [blink_count],
                       'blink_rate' : [blink_rate]}
    
    data.graphs = (fig, fig2)
    
    
