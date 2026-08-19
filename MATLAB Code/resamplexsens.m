function resampledxsens= resamplexsens(rawxsens)
% sampling frequency of EMG sensor  = 1000 Hz
% sampling frequency of XSENS MTW IMU sensor = 100Hz


%isolate pitch data   
pitch = table2array(rawxsens(:,3));
%resample it
resampledxsens = resample(pitch,1000,100);

end