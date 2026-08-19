%% Main file


%% 1. obtain structure containing all the data needed for the project

experimental_data = create_struct;

 
%% 2. process the data
expdata_row = 102 % row within the experimental data structure you are dealing with

%resample xsens data
resampledxsens = resamplexsens(experimental_data(expdata_row).xsensdata); % xsens 'pitch' data, oversampled to BTS emg frequency


[experimental_data(expdata_row).musclenames,experimental_data(expdata_row).muscledata, timedata]= process_data(experimental_data(expdata_row).data,resampledxsens); %function to process the mucle signal
%[experimental_data(expdata_row).musclenames,experimental_data(expdata_row).muscledata, timedata]= process_stepdata(experimental_data(expdata_row).data,resampledxsens); %function to process the mucle signal

DATA_LST = experimental_data(expdata_row).musclenames;  %needed for MS extraction code 'PosturalData_NMFvsPCA'
DATAsrc = experimental_data(expdata_row).muscledata;    %needed for MS extraction code
time = timedata;  %needed for MS extration code
save("muscledata.mat","time","DATAsrc","DATA_LST")   %save variables that will be loaded into MS extraction code 'PosturalData_NMFvsPCA....'
disp('muscle data saved');

%At this point run the 'PosturalData_NMFvsPCA' interface and extract the
%muscle synergies. The values will be prined to the command window and the
%variables 'cutoff_VAF','onesynergy_VAF and 'num_of_synergy_at_threshold'
%will be saved into a file called 'posturaldatavalues.mat'

%% 3. Obtain VAF (95% cutoff value and single synergy value)
%VAF array will hold VAFs of all the patients

load posturaldatavalues.mat  %Synergy values from posturaldata_NMFvsPCS code

%put values into structure in the revelant participant row
experimental_data(expdata_row).cutoff_VAF = synergy_nmf(num_of_synergy_at_threshold).VAF; 
experimental_data(expdata_row).onesynergy_VAF = synergy_nmf(1).VAF;
numofMS(expdata_row) = num_of_synergy_at_threshold;


if experimental_data(expdata_row).patient_group == 1
    young_cutoff_VAF(expdata_row) = synergy_nmf(num_of_synergy_at_threshold).VAF;
    young_VAF_single_synergy(expdata_row) = synergy_nmf(1).VAF;
    
else
    old_cutoff_VAF(expdata_row) = synergy_nmf(num_of_synergy_at_threshold).VAF;
    old_VAF_single_synergy(expdata_row) = synergy_nmf(1).VAF;
end

disp('Vaf data saved')
%% 4. calculate the dmci for the required exercise

experimental_data(expdata_row).dmci = calc_dmci(young_VAF_single_synergy, experimental_data(1).onesynergy_VAF);

%% 5. produce figures of results - see test file












