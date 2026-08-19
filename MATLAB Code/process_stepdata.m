function [DATA_LST,DATAsrc,actualtime,maxval]= process_stepdata(stepdata,resampledxsens)


%% Processing
% expdata_row=52;
% stepdata = experimental_data(expdata_row).data;
% time = table2array(experimental_data(expdata_row).data(:,2));
time =table2array(stepdata(:,2));

% for i = (length(time)+1):height(resampledxsens)
%     resampledxsens(height(resampledxsens)) = []; %same number of datapoints for time and muscle data
% end


for i = (length(resampledxsens)+1):height(time)
    time(height(time)) = []; %same number of datapoints for time and muscle data
end

%let strides be from min peak to min peak - want the middle 30 strides
% = 31 middle peaks
% [trough,troughloc] = findpeaks(-resampledxsens,MinPeakHeight=25); %troughloc holds the cell values
% 
% if mod(length(troughloc),2) == 0
%     
%     troughloc(length(troughloc)+1)=0;
% end
% 
%  %find(troughloc==median(troughloc+makeodd));  %the middle peak
% 
% firststrideval =  trough((find(troughloc==median(troughloc)) - 15));  %data value where the first stride begins in xsens
% laststrideval = trough((find(troughloc==median(troughloc)) + 15));  %data value where the last stride ends in xsens
% 
% firststrideloc = find(resampledxsens == (-firststrideval)); %cell value for first stride
% laststrideloc = find(resampledxsens == (-laststrideval)); %cell value for last stride


    actualxsens = resampledxsens;   %xsens for the middle 30 strides
    actualtime = time;    %time for the middle 30 strides
    actualstepdata = stepdata;  %walking data for the middle 30 strifes





% Create a structure for step data which contains:
% DATA_LST: the list of muscle names (nmus x ncharacters)
% DATAsrc: the data matrix you want to extract synergies from(muscle data) (nmus x ncond)
% Walking data files also should include "time" for plotting purposes.
%%


muscles = stepdata.Properties.VariableNames; %column names are the muscles names (frame time muscle1 muscle2 ...)
muscles(:,1)=[]; %first cell contained 'frame' - remove
muscles(:,1) = []; % second cell (now first cell) contained 'time' - remove
DATA_LST = muscles.'; %a 16x1 column array containing names of muscles


DATA_src = actualstepdata;
DATA_src(:,1)=[];
DATA_src(:,1)=[];
DATASRC=rows2vars(DATA_src);
DATASRC(:,1)= [];
raw_signal = table2array(DATASRC); %Data matrix with muscle synergy data (16x30369)



% the data were high-pass filtered at 40Hz using a 4th order BW,
% de-meaned, recitifed, low-pass filtered at 4Hz using a 4th order BW, and
% resampled to 1000Hz


Fs = 1000; %sampling frequency in Hz
Fcut_low = 4; %lower cut off frequency
Fcut_high = 40; %higher cut off frequency
n_order = 4;
nmus=13;

%% HIGHPASS FILTERING
[b,a] = butter(n_order,Fcut_high/(Fs/2)); %get filter coeff of butter filter

for i=1:nmus
    filt_high_signal(i,:) = filtfilt(b,a,raw_signal(i,:));
    i = i+1 ;
end
%%  DE-MEAN SIGNAL (detrend)
demeaned_signal = detrend(filt_high_signal);


%% RECTIFY the SIGNAL
rect_signal = abs(demeaned_signal);


%% LOWPASS FILTERING

[b,a] = butter(n_order,Fcut_low/(Fs/2)); %get filter coeff of butter filter

for i=1:nmus
    DATAsrc(i,:) = filtfilt(b,a,rect_signal(i,:));
    i = i+1 ;
end

%% Normalizing
% using the peak value in the muscle signal

maxval = 0; %initialize maxval
tempDATAsrc = DATAsrc; %temporary holder of DATAsrc table

for i = 1:height(tempDATAsrc) %for each muscle
    maxval(i) = max(tempDATAsrc(i,:)); %finds the peak contraction for each muscle
end
for n = 1:height(tempDATAsrc)
    DATAsrc(n,:) = tempDATAsrc(n,:)./maxval(1,n);
end





