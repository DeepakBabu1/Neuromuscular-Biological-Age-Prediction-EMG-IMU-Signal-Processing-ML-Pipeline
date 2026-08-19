function experimental_data = create_struct()
%function to create and fill structure with necessary manual inputs
%(patient identifier,age,exercise etc.)

experimental_data = struct('patient_identifier',[],'age',[], 'patient_group',[],'exercise',[],'xsensdata',[],'data',[],'musclenames',[],'muscledata',[],'onesynergy_VAF',[],'cutoff_VAF',[],'dmci',[]);

%% participant 1 expdata(1-4)

experimental_data(1).patient_identifier = 1;
experimental_data(1).age = 31;
experimental_data(1).patient_group = 1; %young
experimental_data(1).exercise = 'walking';
experimental_data(1).data = readtable('part1walking.xlsx'); %read in data


experimental_data(2).patient_identifier = 1;
experimental_data(2).age = 31;
experimental_data(2).patient_group = 1; %young
experimental_data(2).exercise = 'walkingincline';
experimental_data(2).data = readtable('part1walkingincline.xlsx'); %read in data

experimental_data(3).patient_identifier = 1;
experimental_data(3).age = 31;
experimental_data(3).patient_group = 1; %young
experimental_data(3).exercise = 'stepforward';
experimental_data(3).data = readtable('part1stepforward.xlsx'); %read in data

experimental_data(4).patient_identifier = 1;
experimental_data(4).age = 31;
experimental_data(4).patient_group = 1; %young
experimental_data(4).exercise = 'steplateral';
experimental_data(4).data = readtable('part1steplateral.xlsx'); %read in data

%% participant 2 expdata(5-8)

experimental_data(5).patient_identifier = 2;
experimental_data(5).age = 22;
experimental_data(5).patient_group = 1; %young
experimental_data(5).exercise = 'walking';
experimental_data(5).data = readtable('part2walking.xlsx'); %read in data
experimental_data(5).data = removevars(experimental_data(5).data, 'RightTensorFasciaeLatae');
experimental_data(5).xsensdata = readtable('part2_walking_foot.txt'); %read in xsens data

experimental_data(6).patient_identifier = 2;
experimental_data(6).age = 22;
experimental_data(6).patient_group = 1; %young
experimental_data(6).exercise = 'walkingincline';
experimental_data(6).data = readtable('part2walkingincline.xlsx'); %read in data
experimental_data(6).data = removevars(experimental_data(6).data, 'RightTensorFasciaeLatae');
experimental_data(6).xsensdata = readtable('part2_walkingincline_foot.txt'); %read in xsens data

experimental_data(7).patient_identifier = 2;
experimental_data(7).age = 22;
experimental_data(7).patient_group = 1; %young
experimental_data(7).exercise = 'stepforward';
experimental_data(7).data = readtable('part2stepforward.xlsx'); %read in data
experimental_data(7).data = removevars(experimental_data(7).data, 'RightTensorFasciaeLatae');
experimental_data(7).xsensdata = readtable('part2_stepforward_foot.txt'); %read in xsens data

experimental_data(8).patient_identifier = 2;
experimental_data(8).age = 22;
experimental_data(8).patient_group = 1; %young
experimental_data(8).exercise = 'steplateral';
experimental_data(8).data = readtable('part2steplateral.xlsx'); %read in data
experimental_data(8).data = removevars(experimental_data(8).data, 'RightTensorFasciaeLatae');
experimental_data(8).xsensdata = readtable('part2_steplateral_foot.txt'); %read in xsens data


%% participant 3 expdata(9-12)

experimental_data(9).patient_identifier = 3;
experimental_data(9).age = 24;
experimental_data(9).patient_group = 1; %young
experimental_data(9).exercise = 'walking';
experimental_data(9).data = readtable('part3walking.xlsx'); %read in data
experimental_data(9).data = removevars(experimental_data(9).data, 'RightTensorFasciaeLatae');
experimental_data(9).xsensdata = readtable('part3_walking_foot.txt'); %read in xsens data

experimental_data(10).patient_identifier = 3;
experimental_data(10).age = 24;
experimental_data(10).patient_group = 1; %young
experimental_data(10).exercise = 'walkingincline';
experimental_data(10).data = readtable('part3walkingincline.xlsx'); %read in data
experimental_data(10).data = removevars(experimental_data(10).data, 'RightTensorFasciaeLatae');
experimental_data(10).xsensdata = readtable('part3_walkingincline_foot.txt'); %read in xsens data

experimental_data(11).patient_identifier = 3;
experimental_data(11).age = 24;
experimental_data(11).patient_group = 1; %young
experimental_data(11).exercise = 'stepforward';
experimental_data(11).data = readtable('part3stepforward.xlsx'); %read in data
experimental_data(11).data = removevars(experimental_data(11).data, 'RightTensorFasciaeLatae');
experimental_data(11).xsensdata = readtable('part3_stepforward_foot.txt'); %read in xsens data

experimental_data(12).patient_identifier = 3;
experimental_data(12).age = 24;
experimental_data(12).patient_group = 1; %young
experimental_data(12).exercise = 'steplateral';
experimental_data(12).data = readtable('part3steplateral.xlsx'); %read in data
experimental_data(12).data = removevars(experimental_data(12).data, 'RightTensorFasciaeLatae');
experimental_data(12).xsensdata = readtable('part3_steplateral_foot.txt'); %read in xsens data

%% participant 5 expdata(13-16)

experimental_data(13).patient_identifier = 5;
experimental_data(13).age = 34;
experimental_data(13).patient_group = 1; %young
experimental_data(13).exercise = 'walking';
experimental_data(13).data = readtable('part5walking.xlsx'); %read in data
experimental_data(13).data = removevars(experimental_data(13).data, 'RightTensorFasciaeLatae');
experimental_data(13).xsensdata = readtable('part5_walkingflat_foot.txt'); %read in xsens data

experimental_data(14).patient_identifier = 5;
experimental_data(14).age = 34;
experimental_data(14).patient_group = 1; %young
experimental_data(14).exercise = 'walkingincline';
experimental_data(14).data = readtable('part5walkingincline.xlsx'); %read in data
experimental_data(14).data = removevars(experimental_data(14).data, 'RightTensorFasciaeLatae');
experimental_data(14).xsensdata = readtable('part5_walkingincline_foot.txt'); %read in xsens data

experimental_data(15).patient_identifier = 5;
experimental_data(15).age = 34;
experimental_data(15).patient_group = 1; %young
experimental_data(15).exercise = 'stepforward';
experimental_data(15).data = readtable('part5stepforward.xlsx'); %read in data
experimental_data(15).data = removevars(experimental_data(15).data, 'RightTensorFasciaeLatae');
experimental_data(15).xsensdata = readtable('part5_stepforward_foot.txt'); %read in xsens data

experimental_data(16).patient_identifier = 5;
experimental_data(16).age = 34;
experimental_data(16).patient_group = 1; %young
experimental_data(16).exercise = 'steplateral';
experimental_data(16).data = readtable('part5steplateral.xlsx'); %read in data
experimental_data(16).data = removevars(experimental_data(16).data, 'RightTensorFasciaeLatae');
experimental_data(16).xsensdata = readtable('part5_steplateral_foot.txt'); %read in xsens data

%% participant 6 expdata(17-20)

experimental_data(17).patient_identifier = 6;
experimental_data(17).age = 31;
experimental_data(17).patient_group = 1; %young
experimental_data(17).exercise = 'walking';
experimental_data(17).data = readtable('part6walking.xlsx'); %read in data
experimental_data(17).data = removevars(experimental_data(17).data, 'RightTensorFasciaeLatae');
experimental_data(17).xsensdata = readtable('part6_walkingflat_foot.txt'); %read in xsensdata

experimental_data(18).patient_identifier = 6;
experimental_data(18).age = 31;
experimental_data(18).patient_group = 1; %young
experimental_data(18).exercise = 'walkingincline';
experimental_data(18).data = readtable('part6walkingincline.xlsx'); %read in data
experimental_data(18).data = removevars(experimental_data(18).data, 'RightTensorFasciaeLatae');
experimental_data(18).xsensdata = readtable('part6_walkingincline_foot.txt'); %read in data

experimental_data(19).patient_identifier = 6;
experimental_data(19).age = 31;
experimental_data(19).patient_group = 1; %young
experimental_data(19).exercise = 'stepforward';
experimental_data(19).data = readtable('part6stepforward.xlsx'); %read in data
experimental_data(19).data = removevars(experimental_data(19).data, 'RightTensorFasciaeLatae');
experimental_data(19).xsensdata = readtable('part6_stepforward_foot.txt'); %read in data

experimental_data(20).patient_identifier = 6;
experimental_data(20).age = 31;
experimental_data(20).patient_group = 1; %young
experimental_data(20).exercise = 'steplateral';
experimental_data(20).data = readtable('part6steplateral.xlsx'); %read in data
experimental_data(20).data = removevars(experimental_data(20).data, 'RightTensorFasciaeLatae');
experimental_data(20).xsensdata = readtable('part6_steplateral_foot.txt'); %read in data

%% participant 7 expdata(21-24)

experimental_data(21).patient_identifier = 7;
experimental_data(21).age = 22;
experimental_data(21).patient_group = 1; %young
experimental_data(21).exercise = 'walking';
experimental_data(21).data = readtable('part7walking.xlsx'); %read in data
experimental_data(21).data = removevars(experimental_data(21).data, 'RightTensorFasciaeLatae');
experimental_data(21).xsensdata = readtable('part7_walkingflat_foot.txt'); %read in xsens data

experimental_data(22).patient_identifier = 7;
experimental_data(22).age = 22;
experimental_data(22).patient_group = 1; %young
experimental_data(22).exercise = 'walkingincline';
experimental_data(22).data = readtable('part7walkingincline.xlsx'); %read in data
experimental_data(22).data = removevars(experimental_data(22).data, 'RightTensorFasciaeLatae');
experimental_data(22).xsensdata = readtable('part7_walkingincline_foot.txt'); %read in xsens data

experimental_data(23).patient_identifier = 7;
experimental_data(23).age = 22;
experimental_data(23).patient_group = 1; %young
experimental_data(23).exercise = 'stepforward';
experimental_data(23).data = readtable('part7stepforward.xlsx'); %read in data
experimental_data(23).data = removevars(experimental_data(23).data, 'RightTensorFasciaeLatae');
experimental_data(23).xsensdata = readtable('part7_stepforward_foot.txt'); %read in xsens data

experimental_data(24).patient_identifier = 7;
experimental_data(24).age = 22;
experimental_data(24).patient_group = 1; %young
experimental_data(24).exercise = 'steplateral';
experimental_data(24).data = readtable('part7steplateral.xlsx'); %read in data
experimental_data(24).data = removevars(experimental_data(24).data, 'RightTensorFasciaeLatae');
experimental_data(24).xsensdata = readtable('part7_steplateral_foot.txt'); %read in xsens data

%% participant 8 expdata(25-28)

experimental_data(25).patient_identifier = 8;
experimental_data(25).age = 33;
experimental_data(25).patient_group = 1; %young
experimental_data(25).exercise = 'walking';
experimental_data(25).data = readtable('part8walking.xlsx'); %read in data
experimental_data(25).data = removevars(experimental_data(25).data, 'RightTensorFasciaeLatae');
experimental_data(25).xsensdata = readtable('part8_walking_foot.txt'); %read in xsens data

experimental_data(26).patient_identifier = 8;
experimental_data(26).age = 33;
experimental_data(26).patient_group = 1; %young
experimental_data(26).exercise = 'walkingincline';
experimental_data(26).data = readtable('part8walkingincline.xlsx'); %read in data
experimental_data(26).data = removevars(experimental_data(26).data, 'RightTensorFasciaeLatae');
experimental_data(26).xsensdata = readtable('part8_walkingincline_foot.txt'); %read in xsens data

experimental_data(27).patient_identifier = 8;
experimental_data(27).age = 33;
experimental_data(27).patient_group = 1; %young
experimental_data(27).exercise = 'stepforward';
experimental_data(27).data = readtable('part8stepforward.xlsx'); %read in data
experimental_data(27).data = removevars(experimental_data(27).data, 'RightTensorFasciaeLatae');
experimental_data(27).xsensdata = readtable('part8_stepforward_foot.txt'); %read in xsens data

experimental_data(28).patient_identifier = 8;
experimental_data(28).age = 33;
experimental_data(28).patient_group = 1; %young
experimental_data(28).exercise = 'steplateral';
experimental_data(28).data = readtable('part8steplateral.xlsx'); %read in data
experimental_data(28).data = removevars(experimental_data(28).data, 'RightTensorFasciaeLatae');
experimental_data(28).xsensdata = readtable('part8_steplateral_foot.txt'); %read in xsens data

%% participant 9 expdata(29-32)

experimental_data(29).patient_identifier = 9;
experimental_data(29).age = 51;
experimental_data(29).patient_group = 2; %old
experimental_data(29).exercise = 'walking';
experimental_data(29).data = readtable('part9walking.xlsx'); %read in data
experimental_data(29).data = removevars(experimental_data(29).data, 'RightTensorFasciaeLatae');
experimental_data(29).xsensdata = readtable('part9_walking_foot.txt'); %read in xsens data

experimental_data(30).patient_identifier = 9;
experimental_data(30).age = 51;
experimental_data(30).patient_group = 2; %old
experimental_data(30).exercise = 'walkingincline';
experimental_data(30).data = readtable('part9walkingincline.xlsx'); %read in data
experimental_data(30).data = removevars(experimental_data(30).data, 'RightTensorFasciaeLatae');
experimental_data(30).xsensdata = readtable('part9_walkingincline_foot.txt'); %read in xsens data

experimental_data(31).patient_identifier = 9;
experimental_data(31).age = 51;
experimental_data(31).patient_group = 2; %old
experimental_data(31).exercise = 'stepforward';
experimental_data(31).data = readtable('part9stepforward.xlsx'); %read in data
experimental_data(31).data = removevars(experimental_data(31).data, 'RightTensorFasciaeLatae');
experimental_data(31).xsensdata = readtable('part9_stepforward_foot.txt'); %read in xsens data

experimental_data(32).patient_identifier = 9;
experimental_data(32).age = 51;
experimental_data(32).patient_group = 2; %old
experimental_data(32).exercise = 'steplateral';
experimental_data(32).data = readtable('part9steplateral.xlsx'); %read in data
experimental_data(32).data = removevars(experimental_data(32).data, 'RightTensorFasciaeLatae');
experimental_data(32).xsensdata = readtable('part9_steplateral_foot.txt'); %read in xsens data

%% participant 10 expdata(33-36)

experimental_data(33).patient_identifier = 10;
experimental_data(33).age = 51;
experimental_data(33).patient_group = 2; %old
experimental_data(33).exercise = 'walking';
experimental_data(33).data = readtable('part10walking.xlsx'); %read in data
experimental_data(33).data = removevars(experimental_data(33).data, 'RightTensorFasciaeLatae');
experimental_data(33).xsensdata = readtable('part10_walking_foot.txt'); %read in xsens data

experimental_data(34).patient_identifier = 10;
experimental_data(34).age = 51;
experimental_data(34).patient_group = 2 ; %old
experimental_data(34).exercise = 'walkingincline';
experimental_data(34).data = readtable('part10walkingincline.xlsx'); %read in data
experimental_data(34).data = removevars(experimental_data(34).data, 'RightTensorFasciaeLatae');
experimental_data(34).xsensdata = readtable('part10_walkingincline_foot.txt'); %read in xsens data

experimental_data(35).patient_identifier = 10;
experimental_data(35).age = 51;
experimental_data(35).patient_group = 2; %old
experimental_data(35).exercise = 'stepforward';
experimental_data(35).data = readtable('part10stepforward.xlsx'); %read in data
experimental_data(35).data = removevars(experimental_data(35).data, 'RightTensorFasciaeLatae');
experimental_data(35).xsensdata = readtable('part10_stepforward_foot.txt'); %read in xsens data

experimental_data(36).patient_identifier = 10;
experimental_data(36).age = 51;
experimental_data(36).patient_group = 2; %old
experimental_data(36).exercise = 'steplateral';
experimental_data(36).data = readtable('part10steplateral.xlsx'); %read in data
experimental_data(36).data = removevars(experimental_data(36).data, 'RightTensorFasciaeLatae');
experimental_data(36).xsensdata = readtable('part10_steplateral_foot.txt'); %read in xsens data

%% participant 11 expdata(37-40)

experimental_data(37).patient_identifier = 11;
experimental_data(37).age = 22;
experimental_data(37).patient_group = 1; %young
experimental_data(37).exercise = 'walking';
experimental_data(37).data = readtable('part11walking.xlsx'); %read in data
experimental_data(37).data = removevars(experimental_data(37).data, 'RightTensorFasciaeLatae');
experimental_data(37).xsensdata = readtable('part11_walking_foot.txt'); %read in xsens data

experimental_data(38).patient_identifier = 11;
experimental_data(38).age = 22;
experimental_data(38).patient_group = 1; %young
experimental_data(38).exercise = 'walkingincline';
experimental_data(38).data = readtable('part1walkingincline.xlsx'); %read in data
experimental_data(38).data = removevars(experimental_data(38).data, 'RightTensorFasciaeLatae');
experimental_data(38).xsensdata = readtable('part11_walkingincline_foot.txt'); %read in xsens data

experimental_data(39).patient_identifier = 11;
experimental_data(39).age = 22;
experimental_data(39).patient_group = 1; %young
experimental_data(39).exercise = 'stepforward';
experimental_data(39).data = readtable('part11stepforward.xlsx'); %read in data
experimental_data(39).data = removevars(experimental_data(39).data, 'RightTensorFasciaeLatae');
experimental_data(39).xsensdata = readtable('part11_stepforward_foot.txt'); %read in xsens data

experimental_data(40).patient_identifier = 11;
experimental_data(40).age = 22;
experimental_data(40).patient_group = 1; %young
experimental_data(40).exercise = 'steplateral';
experimental_data(40).data = readtable('part11steplateral.xlsx'); %read in data
experimental_data(40).data = removevars(experimental_data(40).data, 'RightTensorFasciaeLatae');
experimental_data(40).xsensdata = readtable('part11_steplateral_foot.txt'); %read in xsens data

%% participant 12 expdata(41-44)

experimental_data(41).patient_identifier = 12;
experimental_data(41).age = 49;
experimental_data(41).patient_group = 1; %young
experimental_data(41).exercise = 'walking';
experimental_data(41).data = readtable('part12walking.xlsx'); %read in data
experimental_data(41).data = removevars(experimental_data(41).data, 'RightTensorFasciaeLatae');
experimental_data(41).xsensdata = readtable('part12_walking_foot.txt'); %read in xsens data

experimental_data(42).patient_identifier = 12;
experimental_data(42).age = 49;
experimental_data(42).patient_group = 1; %young
experimental_data(42).exercise = 'walkingincline';
experimental_data(42).data = readtable('part12walkingincline.xlsx'); %read in data
experimental_data(42).data = removevars(experimental_data(42).data, 'RightTensorFasciaeLatae');
experimental_data(42).xsensdata = readtable('part12_walkingincline_foot.txt'); %read in xsens data

experimental_data(43).patient_identifier = 12;
experimental_data(43).age = 49;
experimental_data(43).patient_group = 1; %young
experimental_data(43).exercise = 'stepforward';
experimental_data(43).data = readtable('part12stepforward.xlsx'); %read in data
experimental_data(43).data = removevars(experimental_data(43).data, 'RightTensorFasciaeLatae');
experimental_data(43).xsensdata = readtable('part12_stepforward_foot.txt'); %read in xsens data

experimental_data(44).patient_identifier = 12;
experimental_data(44).age = 49;
experimental_data(44).patient_group = 1; %young
experimental_data(44).exercise = 'steplateral';
experimental_data(44).data = readtable('part12steplateral.xlsx'); %read in data
experimental_data(44).data = removevars(experimental_data(44).data, 'RightTensorFasciaeLatae');
experimental_data(44).xsensdata = readtable('part12_steplateral_foot.txt'); %read in xsens data

%% participant 13 expdata(45-48)

experimental_data(45).patient_identifier = 13;
experimental_data(45).age = 63;
experimental_data(45).patient_group = 2; %old
experimental_data(45).exercise = 'walking';
experimental_data(45).data = readtable('part13walking.xlsx'); %read in data
experimental_data(45).data = removevars(experimental_data(45).data, 'RightTensorFasciaeLatae');
experimental_data(45).xsensdata = readtable('part13_walking_foot.txt'); %read in xsens data

experimental_data(46).patient_identifier = 13;
experimental_data(46).age = 63;
experimental_data(46).patient_group = 2 ; %old
experimental_data(46).exercise = 'walkingincline';
experimental_data(46).data = readtable('part13walkingincline.xlsx'); %read in data
experimental_data(46).data = removevars(experimental_data(46).data, 'RightTensorFasciaeLatae');
experimental_data(46).xsensdata = readtable('part13_walking_foot.txt'); %read in xsens data (walkingincline was corrupted)

experimental_data(47).patient_identifier = 13;
experimental_data(47).age = 63;
experimental_data(47).patient_group = 2; %old
experimental_data(47).exercise = 'stepforward';
experimental_data(47).data = readtable('part13stepforward.xlsx'); %read in data
experimental_data(47).data = removevars(experimental_data(47).data, 'RightTensorFasciaeLatae');
experimental_data(47).xsensdata = readtable('part13_stepforward_foot.txt'); %read in xsens data

experimental_data(48).patient_identifier = 13;
experimental_data(48).age = 63;
experimental_data(48).patient_group = 2; %old
experimental_data(48).exercise = 'steplateral';
experimental_data(48).data = readtable('part13steplateral.xlsx'); %read in data
experimental_data(48).data = removevars(experimental_data(48).data, 'RightTensorFasciaeLatae');
experimental_data(48).xsensdata = readtable('part13_steplateral_foot.txt'); %read in xsens data

%% participant 14 expdata(49-52)

experimental_data(49).patient_identifier = 14;
experimental_data(49).age = 21;
experimental_data(49).patient_group = 1; %young
experimental_data(49).exercise = 'walking';
experimental_data(49).data = readtable('part14walking.xlsx'); %read in data
experimental_data(49).data = removevars(experimental_data(49).data, 'RightTensorFasciaeLatae');
experimental_data(49).xsensdata = readtable('part14_walking_foot.txt'); %read in xsens data

experimental_data(50).patient_identifier = 14;
experimental_data(50).age = 21;
experimental_data(50).patient_group = 1; %young
experimental_data(50).exercise = 'walkingincline';
experimental_data(50).data = readtable('part14walkingincline.xlsx'); %read in data
experimental_data(50).data = removevars(experimental_data(50).data, 'RightTensorFasciaeLatae');
experimental_data(50).xsensdata = readtable('part14_walkingincline_foot.txt'); %read in xsens data

experimental_data(51).patient_identifier = 14;
experimental_data(51).age = 21;
experimental_data(51).patient_group = 1; %young
experimental_data(51).exercise = 'stepforward';
experimental_data(51).data = readtable('part14stepforward.xlsx'); %read in data
experimental_data(51).data = removevars(experimental_data(51).data, 'RightTensorFasciaeLatae');
experimental_data(51).xsensdata = readtable('part14_stepforward_foot.txt'); %read in xsens data

experimental_data(52).patient_identifier = 14;
experimental_data(52).age = 21;
experimental_data(52).patient_group = 1; %young
experimental_data(52).exercise = 'steplateral';
experimental_data(52).data = readtable('part14steplateral.xlsx'); %read in data
experimental_data(52).data = removevars(experimental_data(52).data, 'RightTensorFasciaeLatae');
experimental_data(52).xsensdata = readtable('part14_steplateral_foot.txt'); %read in xsens data

%% participant 15 expdata(53-56)

experimental_data(53).patient_identifier = 15;
experimental_data(53).age = 55;
experimental_data(53).patient_group = 2; %old
experimental_data(53).exercise = 'walking';
experimental_data(53).data = readtable('part15walking.xlsx'); %read in data
experimental_data(53).data = removevars(experimental_data(53).data, 'RightTensorFasciaeLatae');
experimental_data(53).xsensdata = readtable('part15_walking_foot.txt'); %read in xsens data

experimental_data(54).patient_identifier = 15;
experimental_data(54).age = 55;
experimental_data(54).patient_group = 2 ; %old
experimental_data(54).exercise = 'walkingincline';
experimental_data(54).data = readtable('part15walkingincline.xlsx'); %read in data
experimental_data(54).data = removevars(experimental_data(54).data, 'RightTensorFasciaeLatae');
experimental_data(54).xsensdata = readtable('part15_walkingincline_foot.txt'); %read in xsens data

experimental_data(55).patient_identifier = 15;
experimental_data(55).age = 55;
experimental_data(55).patient_group = 2; %old
experimental_data(55).exercise = 'stepforward';
experimental_data(55).data = readtable('part15stepforward.xlsx'); %read in data
experimental_data(55).data = removevars(experimental_data(55).data, 'RightTensorFasciaeLatae');
experimental_data(55).xsensdata = readtable('part15_stepforward_foot.txt'); %read in xsens data

experimental_data(56).patient_identifier = 15;
experimental_data(56).age = 55;
experimental_data(56).patient_group = 2; %old
experimental_data(56).exercise = 'steplateral';
experimental_data(56).data = readtable('part15steplateral.xlsx'); %read in data
experimental_data(56).data = removevars(experimental_data(56).data, 'RightTensorFasciaeLatae');
experimental_data(56).xsensdata = readtable('part15_steplateral_foot.txt'); %read in xsens data

%% participant 16 expdata(57-60)

experimental_data(57).patient_identifier = 16;
experimental_data(57).age = 52;
experimental_data(57).patient_group = 2; %old
experimental_data(57).exercise = 'walking';
experimental_data(57).data = readtable('part16walking.xlsx'); %read in data
experimental_data(57).data = removevars(experimental_data(57).data, 'RightTensorFasciaeLatae');
experimental_data(57).xsensdata = readtable('part16_walking_foot.txt'); %read in xsens data

experimental_data(58).patient_identifier = 16;
experimental_data(58).age = 52;
experimental_data(58).patient_group = 2 ; %old
experimental_data(58).exercise = 'walkingincline';
experimental_data(58).data = readtable('part16walkingincline.xlsx'); %read in data
experimental_data(58).data = removevars(experimental_data(58).data, 'RightTensorFasciaeLatae');
experimental_data(58).xsensdata = readtable('part16_walkingincline_foot.txt'); %read in xsens data

experimental_data(59).patient_identifier = 16;
experimental_data(59).age = 52;
experimental_data(59).patient_group = 2; %old
experimental_data(59).exercise = 'stepforward';
experimental_data(59).data = readtable('part16stepforward.xlsx'); %read in data
experimental_data(59).data = removevars(experimental_data(59).data, 'RightTensorFasciaeLatae');
experimental_data(59).xsensdata = readtable('part16_stepforward_foot.txt'); %read in xsens data

experimental_data(60).patient_identifier = 16;
experimental_data(60).age = 52;
experimental_data(60).patient_group = 2; %old
experimental_data(60).exercise = 'steplateral';
experimental_data(60).data = readtable('part16steplateral.xlsx'); %read in data
experimental_data(60).data = removevars(experimental_data(60).data, 'RightTensorFasciaeLatae');
experimental_data(60).xsensdata = readtable('part16_steplateral_foot.txt'); %read in xsens data

%% participant 17 expdata(61-64)

experimental_data(61).patient_identifier = 17;
experimental_data(61).age = 34;
experimental_data(61).patient_group = 1; %young
experimental_data(61).exercise = 'walking';
experimental_data(61).data = readtable('part17walking.xlsx'); %read in data
experimental_data(61).data = removevars(experimental_data(61).data, 'RightTensorFasciaeLatae');
experimental_data(61).xsensdata = readtable('part17_walking_foot.txt'); %read in xsens data

experimental_data(62).patient_identifier = 17;
experimental_data(62).age = 34;
experimental_data(62).patient_group = 1; %young
experimental_data(62).exercise = 'walkingincline';
experimental_data(62).data = readtable('part17walkingincline.xlsx'); %read in data
experimental_data(62).data = removevars(experimental_data(62).data, 'RightTensorFasciaeLatae');
experimental_data(62).xsensdata = readtable('part17_walkingincline_foot.txt'); %read in xsens data

experimental_data(63).patient_identifier = 17;
experimental_data(63).age = 34;
experimental_data(63).patient_group = 1; %young
experimental_data(63).exercise = 'stepforward';
experimental_data(63).data = readtable('part17stepforward.xlsx'); %read in data
experimental_data(63).data = removevars(experimental_data(63).data, 'RightTensorFasciaeLatae');
experimental_data(63).xsensdata = readtable('part17_stepforward_foot.txt'); %read in xsens data

experimental_data(64).patient_identifier = 17;
experimental_data(64).age = 34;
experimental_data(64).patient_group = 1; %young
experimental_data(64).exercise = 'steplateral';
experimental_data(64).data = readtable('part17steplateral.xlsx'); %read in data
experimental_data(64).data = removevars(experimental_data(64).data, 'RightTensorFasciaeLatae');
experimental_data(64).xsensdata = readtable('part17_steplateral_foot.txt'); %read in xsens data

%% participant 18 expdata(65-68)

experimental_data(65).patient_identifier = 18;
experimental_data(65).age = 29;
experimental_data(65).patient_group = 1; %young
experimental_data(65).exercise = 'walking';
experimental_data(65).data = readtable('part18walking.xlsx'); %read in data
experimental_data(65).data = removevars(experimental_data(65).data, 'RightTensorFasciaeLatae');
experimental_data(65).xsensdata = readtable('part18_walking_foot.txt'); %read in xsens data

experimental_data(66).patient_identifier = 18;
experimental_data(66).age = 29;
experimental_data(66).patient_group = 1; %young
experimental_data(66).exercise = 'walkingincline';
experimental_data(66).data = readtable('part18walkingincline.xlsx'); %read in data
experimental_data(66).data = removevars(experimental_data(66).data, 'RightTensorFasciaeLatae');
experimental_data(66).xsensdata = readtable('part18_walkingincline_foot.txt'); %read in xsens data

experimental_data(67).patient_identifier = 18;
experimental_data(67).age = 29;
experimental_data(67).patient_group = 1; %young
experimental_data(67).exercise = 'stepforward';
experimental_data(67).data = readtable('part18stepforward.xlsx'); %read in data
experimental_data(67).data = removevars(experimental_data(67).data, 'RightTensorFasciaeLatae');
experimental_data(67).xsensdata = readtable('part18_stepforward_foot.txt'); %read in xsens data

experimental_data(68).patient_identifier = 18;
experimental_data(68).age = 29;
experimental_data(68).patient_group = 1; %young
experimental_data(68).exercise = 'steplateral';
experimental_data(68).data = readtable('part18steplateral.xlsx'); %read in data
experimental_data(68).data = removevars(experimental_data(68).data, 'RightTensorFasciaeLatae');
experimental_data(68).xsensdata = readtable('part18_steplateral_foot.txt'); %read in xsens data

%% participant 19 expdata(69-72)

experimental_data(69).patient_identifier = 19;
experimental_data(69).age = 25;
experimental_data(69).patient_group = 1; %young
experimental_data(69).exercise = 'walking';
experimental_data(69).data = readtable('part19walking.xlsx'); %read in data
experimental_data(69).data = removevars(experimental_data(69).data, 'RightTensorFasciaeLatae');
experimental_data(69).xsensdata = readtable('part19_walking_foot.txt'); %read in xsens data

experimental_data(70).patient_identifier = 19;
experimental_data(70).age = 25;
experimental_data(70).patient_group = 1; %young
experimental_data(70).exercise = 'walkingincline';
experimental_data(70).data = readtable('part19walkingincline.xlsx'); %read in data
experimental_data(70).data = removevars(experimental_data(70).data, 'RightTensorFasciaeLatae');
experimental_data(70).xsensdata = readtable('part19_walkingincline_foot.txt'); %read in xsens data

experimental_data(71).patient_identifier = 19;
experimental_data(71).age = 25;
experimental_data(71).patient_group = 1; %young
experimental_data(71).exercise = 'stepforward';
experimental_data(71).data = readtable('part19stepforward.xlsx'); %read in data
experimental_data(71).data = removevars(experimental_data(71).data, 'RightTensorFasciaeLatae');
experimental_data(71).xsensdata = readtable('part19_stepforward_foot.txt'); %read in xsens data

experimental_data(72).patient_identifier = 19;
experimental_data(72).age = 25;
experimental_data(72).patient_group= 1; %young
experimental_data(72).exercise = 'steplateral';
experimental_data(72).data = readtable('part19steplateral.xlsx'); %read in data
experimental_data(72).data = removevars(experimental_data(72).data, 'RightTensorFasciaeLatae');
experimental_data(72).xsensdata = readtable('part19_steplateral_foot.txt'); %read in xsens data

%% participant 20
experimental_data(73).patient_identifier = 20;
experimental_data(73).age = 23;
experimental_data(73).patient_group = 1; %young
experimental_data(73).exercise = 'walking';
experimental_data(73).data = readtable('part20walking.xlsx'); %read in data
experimental_data(73).data = removevars(experimental_data(73).data, 'RightTensorFasciaeLatae');
experimental_data(73).xsensdata = readtable('part20_walking_foot.txt'); %read in xsens data

experimental_data(74).patient_identifier = 20;
experimental_data(74).age = 23;
experimental_data(74).patient_group = 1; %young
experimental_data(74).exercise = 'walkingincline';
experimental_data(74).data = readtable('part20walkingincline.xlsx'); %read in data
experimental_data(74).data = removevars(experimental_data(74).data, 'RightTensorFasciaeLatae');
experimental_data(74).xsensdata = readtable('part20_walkingincline_foot.txt'); %read in xsens data

experimental_data(75).patient_identifier = 20;
experimental_data(75).age = 23;
experimental_data(75).patient_group = 1; %young
experimental_data(75).exercise = 'stepfoward';
experimental_data(75).data = readtable('part20stepforward.xlsx'); %read in data
experimental_data(75).data = removevars(experimental_data(75).data, 'RightTensorFasciaeLatae');
experimental_data(75).xsensdata = readtable('part20_stepforward_foot.txt'); %read in xsens data

experimental_data(76).patient_identifier = 20;
experimental_data(76).age = 23;
experimental_data(76).patient_group = 1; %young
experimental_data(76).exercise = 'steplateral';
experimental_data(76).data = readtable('part20steplateral.xlsx'); %read in data
experimental_data(76).data = removevars(experimental_data(76).data, 'RightTensorFasciaeLatae');
experimental_data(76).xsensdata = readtable('part20_steplateral_foot.txt'); %read in xsens data

%% participant 21 

experimental_data(77).patient_identifier = 21;
experimental_data(77).age = 63;
experimental_data(77).patient_group = 2; %old
experimental_data(77).exercise = 'walking';
experimental_data(77).data = readtable('part21walking.xlsx'); %read in data
experimental_data(77).data = removevars(experimental_data(77).data, 'RightTensorFasciaeLatae');
experimental_data(77).xsensdata = readtable('part21_walking_foot.txt'); %read in xsens data

experimental_data(78).patient_identifier = 21;
experimental_data(78).age = 63;
experimental_data(78).patient_group = 2; %old
experimental_data(78).exercise = 'walkingincline';
experimental_data(78).data = readtable('part21walkingincline.xlsx'); %read in data
experimental_data(78).data = removevars(experimental_data(78).data, 'RightTensorFasciaeLatae');
experimental_data(78).xsensdata = readtable('part21_walkingincline_foot.txt'); %read in xsens data

experimental_data(79).patient_identifier = 21;
experimental_data(79).age = 63;
experimental_data(79).patient_group = 2; %old
experimental_data(79).exercise = 'stepforward';
experimental_data(79).data = readtable('part21stepforward.xlsx'); %read in data
experimental_data(79).data = removevars(experimental_data(79).data, 'RightTensorFasciaeLatae');
experimental_data(79).xsensdata = readtable('part21_stepforward_foot.txt'); %read in xsens data

experimental_data(80).patient_identifier = 21;
experimental_data(80).age = 63;
experimental_data(80).patient_group = 2; %old
experimental_data(80).exercise = 'steplateral';
experimental_data(80).data = readtable('part21steplateral.xlsx'); %read in data
experimental_data(80).data = removevars(experimental_data(80).data, 'RightTensorFasciaeLatae');
experimental_data(80).xsensdata = readtable('part21_steplateral_foot.txt'); %read in xsens data

%% participant 22

experimental_data(81).patient_identifier = 22;
experimental_data(81).age = 22;
experimental_data(81).patient_group = 1; %young
experimental_data(81).exercise = 'walking';
experimental_data(81).data = readtable('part22walking.xlsx'); %read in data
experimental_data(81).data = removevars(experimental_data(81).data, 'RightTensorFasciaeLatae');
experimental_data(81).xsensdata = readtable('part22_walking_foot.txt'); %read in xsens data

experimental_data(82).patient_identifier = 22;
experimental_data(82).age = 22;
experimental_data(82).patient_group = 1; %young
experimental_data(82).exercise = 'walkingincline';
experimental_data(82).data = readtable('part22walkingincline.xlsx'); %read in data
experimental_data(82).data = removevars(experimental_data(82).data, 'RightTensorFasciaeLatae');
experimental_data(82).xsensdata = readtable('part22_walkingincline_foot.txt'); %read in xsens data

experimental_data(83).patient_identifier = 22;
experimental_data(83).age = 22;
experimental_data(83).patient_group = 1; %young
experimental_data(83).exercise = 'stepfoward';
experimental_data(83).data = readtable('part22stepforward.xlsx'); %read in data
experimental_data(83).data = removevars(experimental_data(83).data, 'RightTensorFasciaeLatae');
experimental_data(83).xsensdata = readtable('part22_stepforward_foot.txt'); %read in xsens data

experimental_data(84).patient_identifier = 22;
experimental_data(84).age = 22;
experimental_data(84).patient_group = 1; %young
experimental_data(84).exercise = 'steplateral';
experimental_data(84).data = readtable('part22steplateral.xlsx'); %read in data
experimental_data(84).data = removevars(experimental_data(84).data, 'RightTensorFasciaeLatae');
experimental_data(84).xsensdata = readtable('part22_steplateral_foot.txt'); %read in xsens data


%% participant 23

% discarded

%% participant 24

experimental_data(85).patient_identifier = 24;
experimental_data(85).age = 57;
experimental_data(85).patient_group = 2; %old
experimental_data(85).exercise = 'walking';
experimental_data(85).data = readtable('part24walking.xlsx'); %read in data
experimental_data(85).data = removevars(experimental_data(85).data, 'RightTensorFasciaeLatae');
experimental_data(85).xsensdata = readtable('part24_walking_foot.txt'); %read in xsens data

experimental_data(86).patient_identifier = 24;
experimental_data(86).age = 57;
experimental_data(86).patient_group = 2; %old
experimental_data(86).exercise = 'walkingincline';
experimental_data(86).data = readtable('part24walkingincline.xlsx'); %read in data
experimental_data(86).data = removevars(experimental_data(86).data, 'RightTensorFasciaeLatae');
experimental_data(86).xsensdata = readtable('part24_walkingincline_foot.txt'); %read in xsens data

experimental_data(87).patient_identifier = 24;
experimental_data(87).age = 57;
experimental_data(87).patient_group = 2; %old
experimental_data(87).exercise = 'stepforward';
experimental_data(87).data = readtable('part24stepforward.xlsx'); %read in data
experimental_data(87).data = removevars(experimental_data(87).data, 'RightTensorFasciaeLatae');
experimental_data(87).xsensdata = readtable('part24_stepforward_foot.txt'); %read in xsens data

experimental_data(88).patient_identifier = 24;
experimental_data(88).age = 57;
experimental_data(88).patient_group = 2; %old
experimental_data(88).exercise = 'steplateral';
experimental_data(88).data = readtable('part24steplateral.xlsx'); %read in data
experimental_data(88).data = removevars(experimental_data(88).data, 'RightTensorFasciaeLatae');
experimental_data(88).xsensdata = readtable('part24_steplateral_foot.txt'); %read in xsens data

%% participant 25
experimental_data(89).patient_identifier = 25;
experimental_data(89).age = 50;
experimental_data(89).patient_group = 2; %old
experimental_data(89).exercise = 'walking';
experimental_data(89).data = readtable('part25walking.xlsx'); %read in data
experimental_data(89).data = removevars(experimental_data(89).data, 'RightTensorFasciaeLatae');
experimental_data(89).xsensdata = readtable('part25_walking_foot.txt'); %read in xsens data

experimental_data(90).patient_identifier = 25;
experimental_data(90).age = 50;
experimental_data(90).patient_group = 2; %old
experimental_data(90).exercise = 'walkingincline';
experimental_data(90).data = readtable('part25walkingincline.xlsx'); %read in data
experimental_data(90).data = removevars(experimental_data(90).data, 'RightTensorFasciaeLatae');
experimental_data(90).xsensdata = readtable('part25_walkingincline_foot.txt'); %read in xsens data

experimental_data(91).patient_identifier = 25;
experimental_data(91).age = 50;
experimental_data(91).patient_group = 2; %old
experimental_data(91).exercise = 'stepforward';
experimental_data(91).data = readtable('part25stepforward.xlsx'); %read in data
experimental_data(91).data = removevars(experimental_data(91).data, 'RightTensorFasciaeLatae');
experimental_data(91).xsensdata = readtable('part25_stepforward_foot.txt'); %read in xsens data

experimental_data(92).patient_identifier = 25;
experimental_data(92).age = 50;
experimental_data(92).patient_group = 2; %old
experimental_data(92).exercise = 'steplateral';
experimental_data(92).data = readtable('part25steplateral.xlsx'); %read in data
experimental_data(92).data = removevars(experimental_data(92).data, 'RightTensorFasciaeLatae');
experimental_data(92).xsensdata = readtable('part25_steplateral_foot.txt'); %read in xsens data

%% participant 26
experimental_data(93).patient_identifier = 26;
experimental_data(93).age = 54;
experimental_data(93).patient_group = 2; %old
experimental_data(93).exercise = 'walking';
experimental_data(93).data = readtable('part26walking.xlsx'); %read in data
experimental_data(93).data = removevars(experimental_data(93).data, 'RightTensorFasciaeLatae');
experimental_data(93).xsensdata = readtable('part26_walking_foot.txt'); %read in xsens data

experimental_data(94).patient_identifier = 26;
experimental_data(94).age = 54;
experimental_data(94).patient_group = 2; %old
experimental_data(94).exercise = 'walkingincline';
experimental_data(94).data = readtable('part26walkingincline.xlsx'); %read in data
experimental_data(94).data = removevars(experimental_data(94).data, 'RightTensorFasciaeLatae');
experimental_data(94).xsensdata = readtable('part26_walkingincline_foot.txt'); %read in xsens data

experimental_data(95).patient_identifier = 26;
experimental_data(95).age = 54;
experimental_data(95).patient_group = 2; %old
experimental_data(95).exercise = 'stepforward';
experimental_data(95).data = readtable('part26stepforward.xlsx'); %read in data
experimental_data(95).data = removevars(experimental_data(95).data, 'RightTensorFasciaeLatae');
experimental_data(95).xsensdata = readtable('part26_stepforward_foot.txt'); %read in xsens data

experimental_data(96).patient_identifier = 26;
experimental_data(96).age = 54;
experimental_data(96).patient_group = 2; %old
experimental_data(96).exercise = 'steplateral';
experimental_data(96).data = readtable('part26steplateral.xlsx'); %read in data
experimental_data(96).data = removevars(experimental_data(96).data, 'RightTensorFasciaeLatae');
experimental_data(96).xsensdata = readtable('part26_steplateral_foot.txt'); %read in xsens data

%% participant 27
experimental_data(97).patient_identifier = 27;
experimental_data(97).age = 53;
experimental_data(97).patient_group = 2; %old
experimental_data(97).exercise = 'walking';
experimental_data(97).data = readtable('part27walking.xlsx'); %read in data
experimental_data(97).data = removevars(experimental_data(97).data, 'RightTensorFasciaeLatae');
experimental_data(97).xsensdata = readtable('part27_walking_foot.txt'); %read in xsens data

experimental_data(98).patient_identifier = 27;
experimental_data(98).age = 53;
experimental_data(98).patient_group = 2; %old
experimental_data(98).exercise = 'walkingincline';
experimental_data(98).data = readtable('part27walkingincline.xlsx'); %read in data
experimental_data(98).data = removevars(experimental_data(98).data, 'RightTensorFasciaeLatae');
experimental_data(98).xsensdata = readtable('part27_walkingincline_foot.txt'); %read in xsens data

experimental_data(99).patient_identifier = 27;
experimental_data(99).age = 53;
experimental_data(99).patient_group = 2; %old
experimental_data(99).exercise = 'stepforward';
experimental_data(99).data = readtable('part27stepforward.xlsx'); %read in data
experimental_data(99).data = removevars(experimental_data(99).data, 'RightTensorFasciaeLatae');
experimental_data(99).xsensdata = readtable('part27_stepforward_foot.txt'); %read in xsens data

experimental_data(100).patient_identifier = 27;
experimental_data(100).age = 53;
experimental_data(100).patient_group = 2; %old
experimental_data(100).exercise = 'steplateral';
experimental_data(100).data = readtable('part27steplateral.xlsx'); %read in data
experimental_data(100).data = removevars(experimental_data(100).data, 'RightTensorFasciaeLatae');
experimental_data(100).xsensdata = readtable('part27_steplateral_foot.txt'); %read in xsens data

%% participant 28
%discarded

%% participant 29
experimental_data(101).patient_identifier = 29;
experimental_data(101).age = 53;
experimental_data(101).patient_group = 2; %old
experimental_data(101).exercise = 'walking';
experimental_data(101).data = readtable('part29walking.xlsx'); %read in data
experimental_data(101).data = removevars(experimental_data(101).data, 'RightTensorFasciaeLatae');
experimental_data(101).xsensdata = readtable('part29_walking_foot.txt'); %read in xsens data

experimental_data(102).patient_identifier = 29;
experimental_data(102).age = 53;
experimental_data(102).patient_group = 2; %old
experimental_data(102).exercise = 'walkingincline';
experimental_data(102).data = readtable('part29walkingincline.xlsx'); %read in data
experimental_data(102).data = removevars(experimental_data(102).data, 'RightTensorFasciaeLatae');
experimental_data(102).xsensdata = readtable('part29_stepforward_foot.txt'); % mixed up read in xsens data

experimental_data(103).patient_identifier = 29;
experimental_data(103).age = 53;
experimental_data(103).patient_group = 2; %old
experimental_data(103).exercise = 'stepforward';
experimental_data(103).data = readtable('part29stepforward.xlsx'); %read in data
experimental_data(103).data = removevars(experimental_data(103).data, 'RightTensorFasciaeLatae');
experimental_data(103).xsensdata = readtable('part29_walkingincline_foot.txt'); %read in xsens data

experimental_data(104).patient_identifier = 29;
experimental_data(104).age = 53;
experimental_data(104).patient_group = 2; %old
experimental_data(104).exercise = 'steplateral';
experimental_data(104).data = readtable('part29steplateral.xlsx'); %read in data
experimental_data(104).data = removevars(experimental_data(104).data, 'RightTensorFasciaeLatae');
experimental_data(104).xsensdata = readtable('part29_steplateral_foot.txt'); %read in xsens data

%% participant 30 105 108
experimental_data(105).patient_identifier = 30;
experimental_data(105).age = 53;
experimental_data(105).patient_group = 2; %old
experimental_data(105).exercise = 'walking';
experimental_data(105).data = readtable('part30walking.xlsx'); %read in data
experimental_data(105).data = removevars(experimental_data(105).data, 'RightTensorFasciaeLatae');
experimental_data(105).xsensdata = readtable('part30_walking_foot.txt'); %read in xsens data

experimental_data(106).patient_identifier = 30;
experimental_data(106).age = 53;
experimental_data(106).patient_group = 2; %old
experimental_data(106).exercise = 'walkingincline';
experimental_data(106).data = readtable('part30walkingincline.xlsx'); %read in data
experimental_data(106).data = removevars(experimental_data(106).data, 'RightTensorFasciaeLatae');
experimental_data(106).xsensdata = readtable('part30_walkingincline_foot.txt'); %read in xsens data

experimental_data(107).patient_identifier = 30;
experimental_data(107).age = 53;
experimental_data(107).patient_group = 2; %old
experimental_data(107).exercise = 'stepfoward';
experimental_data(107).data = readtable('part30stepforward.xlsx'); %read in data
experimental_data(107).data = removevars(experimental_data(107).data, 'RightTensorFasciaeLatae');
experimental_data(107).xsensdata = readtable('part30_stepforward_foot.txt'); %read in xsens data

experimental_data(108).patient_identifier = 30;
experimental_data(108).age = 53;
experimental_data(108).patient_group = 2; %old
experimental_data(108).exercise = 'steplateral';
experimental_data(108).data = readtable('part30steplateral.xlsx'); %read in data
experimental_data(108).data = removevars(experimental_data(108).data, 'RightTensorFasciaeLatae');
experimental_data(108).xsensdata = readtable('part30_steplateral_foot.txt'); %read in xsens 

%% participant 31 109 112
experimental_data(109).patient_identifier = 31;
experimental_data(109).age = 74;
experimental_data(109).patient_group = 2; %old
experimental_data(109).exercise = 'walking';
experimental_data(109).data = readtable('part31walking.xlsx'); %read in data
experimental_data(109).data = removevars(experimental_data(109).data, 'RightTensorFasciaeLatae');
experimental_data(109).xsensdata = readtable('part31_walking_foot.txt'); %read in xsens data

experimental_data(110).patient_identifier = 31;
experimental_data(110).age = 74;
experimental_data(110).patient_group = 2; %old
experimental_data(110).exercise = 'walkingincline';
experimental_data(110).data = readtable('part31walkingincline.xlsx'); %read in data
experimental_data(110).data = removevars(experimental_data(110).data, 'RightTensorFasciaeLatae');
experimental_data(110).xsensdata = readtable('part31_walkingincline_foot.txt'); %read in xsens data

experimental_data(111).patient_identifier = 31;
experimental_data(111).age = 74;
experimental_data(111).patient_group = 2; %old
experimental_data(111).exercise = 'stepfoward';
experimental_data(111).data = readtable('part31stepforward.xlsx'); %read in data
experimental_data(111).data = removevars(experimental_data(111).data, 'RightTensorFasciaeLatae');
experimental_data(111).xsensdata = readtable('part31_stepforward_foot.txt'); %read in xsens data

experimental_data(112).patient_identifier = 31;
experimental_data(112).age = 74;
experimental_data(112).patient_group = 2; %old
experimental_data(112).exercise = 'steplateral';
experimental_data(112).data = readtable('part31steplateral.xlsx'); %read in data
experimental_data(112).data = removevars(experimental_data(112).data, 'RightTensorFasciaeLatae');
experimental_data(112).xsensdata = readtable('part31_steplateral_foot.txt'); %read in xsens data
%% participant 32
experimental_data(113).patient_identifier = 32;
experimental_data(113).age = 53;
experimental_data(113).patient_group = 2; %old
experimental_data(113).exercise = 'walking';
experimental_data(113).data = readtable('part32walking.xlsx'); %read in data
experimental_data(113).data = removevars(experimental_data(113).data, 'RightTensorFasciaeLatae');
experimental_data(113).xsensdata = readtable('part32_walking_foot.txt'); %read in xsens data

experimental_data(114).patient_identifier = 32;
experimental_data(114).age = 53;
experimental_data(114).patient_group = 2; %old
experimental_data(114).exercise = 'walkingincline';
experimental_data(114).data = readtable('part32walkingincline.xlsx'); %read in data
experimental_data(114).data = removevars(experimental_data(114).data, 'RightTensorFasciaeLatae');
experimental_data(114).xsensdata = readtable('part32_walkingincline_foot.txt'); %read in xsens data

experimental_data(115).patient_identifier = 32;
experimental_data(115).age = 53;
experimental_data(115).patient_group = 2; %old
experimental_data(115).exercise = 'stepforward';
experimental_data(115).data = readtable('part32stepforward.xlsx'); %read in data
experimental_data(115).data = removevars(experimental_data(115).data, 'RightTensorFasciaeLatae');
experimental_data(115).xsensdata = readtable('part32_stepforward_foot.txt'); %read in xsens data

experimental_data(116).patient_identifier = 32;
experimental_data(116).age = 53;
experimental_data(116).patient_group = 2; %old
experimental_data(116).exercise = 'steplateral';
experimental_data(116).data = readtable('part32steplateral.xlsx'); %read in data
experimental_data(116).data = removevars(experimental_data(116).data, 'RightTensorFasciaeLatae');
experimental_data(116).xsensdata = readtable('part32_steplateral_foot.txt'); %read in xsens data

%% participant 33
experimental_data(117).patient_identifier = 33;
experimental_data(117).age = 34;
experimental_data(117).patient_group = 1; %young
experimental_data(117).exercise = 'walking';
experimental_data(117).data = readtable('part33walking.xlsx'); %read in data
experimental_data(117).data = removevars(experimental_data(117).data, 'RightTensorFasciaeLatae');
experimental_data(117).xsensdata = readtable('part33_walking_foot.txt'); %read in xsens data

experimental_data(118).patient_identifier = 33;
experimental_data(118).age = 34;
experimental_data(118).patient_group = 1; %young
experimental_data(118).exercise = 'walkingincline';
experimental_data(118).data = readtable('part33walkingincline.xlsx'); %read in data
experimental_data(118).data = removevars(experimental_data(118).data, 'RightTensorFasciaeLatae');
experimental_data(118).xsensdata = readtable('part33_walkingincline_foot.txt'); %read in xsens data

experimental_data(119).patient_identifier = 33;
experimental_data(119).age = 34;
experimental_data(119).patient_group = 1; %young
experimental_data(119).exercise = 'stepforward';
experimental_data(119).data = readtable('part33stepforward.xlsx'); %read in data
experimental_data(119).data = removevars(experimental_data(119).data, 'RightTensorFasciaeLatae');
experimental_data(119).xsensdata = readtable('part33_stepforward_foot.txt'); %read in xsens data

experimental_data(120).patient_identifier = 33;
experimental_data(120).age = 34;
experimental_data(120).patient_group = 1; %young
experimental_data(120).exercise = 'steplateral';
experimental_data(120).data = readtable('part3steplateral.xlsx'); %read in data
experimental_data(120).data = removevars(experimental_data(120).data, 'RightTensorFasciaeLatae');
experimental_data(120).xsensdata = readtable('part33_steplateral_foot.txt'); %read in xsens data

%% participant 34
%discarded

%% participant 35
experimental_data(121).patient_identifier = 35;
experimental_data(121).age = 60;
experimental_data(121).patient_group = 2; %old
experimental_data(121).exercise = 'walking';
experimental_data(121).data = readtable('part35walking.xlsx'); %read in data
experimental_data(121).data = removevars(experimental_data(121).data, 'RightTensorFasciaeLatae');
experimental_data(121).xsensdata = readtable('part35_walking_foot.txt'); %read in xsens data

experimental_data(122).patient_identifier = 35;
experimental_data(122).age = 60;
experimental_data(122).patient_group = 2; %old
experimental_data(122).exercise = 'walkingincline';
experimental_data(122).data = readtable('part35walkingincline.xlsx'); %read in data
experimental_data(122).data = removevars(experimental_data(122).data, 'RightTensorFasciaeLatae');
experimental_data(122).xsensdata = readtable('part35_walkingincline_foot.txt'); %read in xsens data

experimental_data(123).patient_identifier = 35;
experimental_data(123).age = 60;
experimental_data(123).patient_group = 2; %old
experimental_data(123).exercise = 'stepforward';
experimental_data(123).data = readtable('part35stepforward.xlsx'); %read in data
experimental_data(123).data = removevars(experimental_data(123).data, 'RightTensorFasciaeLatae');
experimental_data(123).xsensdata = readtable('part35_stepforward_foot.txt'); %read in xsens data

experimental_data(124).patient_identifier = 35;
experimental_data(124).age = 60;
experimental_data(124).patient_group = 2; %old
experimental_data(124).exercise = 'steplateral';
experimental_data(124).data = readtable('part35steplateral.xlsx'); %read in data
experimental_data(124).data = removevars(experimental_data(124).data, 'RightTensorFasciaeLatae');
experimental_data(124).xsensdata = readtable('part35_steplateral_foot.txt'); %read in xsens data

%% participant 36
experimental_data(125).patient_identifier = 36;
experimental_data(125).age = 22;
experimental_data(125).patient_group = 1; %young
experimental_data(125).exercise = 'walking';
experimental_data(125).data = readtable('part36walking.xlsx'); %read in data
experimental_data(125).data = removevars(experimental_data(125).data, 'RightTensorFasciaeLatae');
experimental_data(125).xsensdata = readtable('part36_walking_foot.txt'); %read in xsens data

experimental_data(126).patient_identifier = 36;
experimental_data(126).age = 22;
experimental_data(126).patient_group = 1; %young
experimental_data(126).exercise = 'walkingincline';
experimental_data(126).data = readtable('part36walkingincline.xlsx'); %read in data
experimental_data(126).data = removevars(experimental_data(126).data, 'RightTensorFasciaeLatae');
experimental_data(126).xsensdata = readtable('part36_walkingincline_foot.txt'); %read in xsens data

experimental_data(127).patient_identifier = 36;
experimental_data(127).age = 22;
experimental_data(127).patient_group = 1; %young
experimental_data(127).exercise = 'stepforward';
experimental_data(127).data = readtable('part36stepforward.xlsx'); %read in data
experimental_data(127).data = removevars(experimental_data(127).data, 'RightTensorFasciaeLatae');
experimental_data(127).xsensdata = readtable('part36_stepforward_foot.txt'); %read in xsens data

experimental_data(128).patient_identifier = 36;
experimental_data(128).age = 22;
experimental_data(128).patient_group = 1; %young
experimental_data(128).exercise = 'steplateral';
experimental_data(128).data = readtable('part36steplateral.xlsx'); %read in data
experimental_data(128).data = removevars(experimental_data(128).data, 'RightTensorFasciaeLatae');
experimental_data(128).xsensdata = readtable('part36_steplateral_foot.txt'); %read in xsens data
end