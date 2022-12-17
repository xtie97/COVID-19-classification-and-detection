clear; 
T = readtable('./HF_Oct_extra_info_opacity_covid_dicom_map_study_time.csv');
Acc = T.Acc; 
Filename = T.Filename(contains(Acc, '-001')); 
label_positive = T.label_positive(contains(Acc, '-001'));
score_positive = T.covid_score_HF_total(contains(Acc, '-001')); 
Opacity = T.Opacity_score_mimic(contains(Acc, '-001')); 
delta = T.delta(contains(Acc, '-001')); 
Time_oct = T.StudyTime(contains(Acc, '-001')); 
Time_oct = floor(Time_oct/100); % {}h{}min{}s --> {}h{}min
Time_oct_h = floor(Time_oct/100); % {}h
Time_oct_min = Time_oct - Time_oct_h*100; % {}min 
Time_oct = Time_oct_h*60 + Time_oct_min; 
r = MgSetFigureTheme("dark");
edges = 0:20:max(Time_oct);
figure(1); 
histogram(Time_oct, edges); 
xlabel('Distribution of CXR acquisition time starting from 00:00 (min)'); ylabel('Counts');

Acc = Acc(contains(Acc, '-001')); 


Filename = Filename(delta>=-7 & delta<=3);
label_positive = label_positive(delta>=-7 & delta<=3);
score_positive = score_positive(delta>=-7 & delta<=3);
Opacity = Opacity(delta>=-7 & delta<=3);
Acc = Acc(delta>=-7 & delta<=3);
delta = delta(delta>=-7 & delta<=3);
Time_oct = Time_oct(delta>=-7 & delta<=3);
Date_oct = cell(size(Filename)); 


T = readtable('./covidcxr_share_info_total.xlsx');
DeIDACC = T.DeIDACC; 
Date_all = T.StudyDate;

parfor ii = 1: length(Filename)
    accind = Acc(ii);
    if sum(contains(DeIDACC, accind)) > 0
        Date_oct{ii} = Date_all(contains(DeIDACC, accind)); 
    end
end 


T = table(Filename, Acc, label_positive, score_positive, Opacity, Date_oct, Time_oct, delta); 
writetable(T,'./HF_Oct_info_simulation.csv','WriteRowNames',false); 


