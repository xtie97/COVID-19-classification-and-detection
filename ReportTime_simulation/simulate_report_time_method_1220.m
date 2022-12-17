
%%
clear; 
clc; 
T = readtable('./HF_Oct_info_simulation.csv'); % delta<1, first CXR
Filename = T.Filename; 
label_positive = T.label_positive; 
score_positive = T.score_positive; 
opacity = T.Opacity;
Acc = T.Acc;
StudyDate = T.Date_oct;
patient_list = [Acc, num2cell(label_positive), num2cell(score_positive)]; 
StudyDate_order = unique(StudyDate); 

%% Parameters 
num_days = 31; % days in Oct
avgreport_time = 20;  % min
stdreport_time = 2; % min 
acquisition_time = 24; % {} h for CXR acquisition per day  10h: 8am-6pm
working_time = 100; % h 
opacity_thres = 0.2;
num_radiologists = 2; 
rand_seed = 716; 
time_pos_FCFS_list = [];
time_neg_FCFS_list = [];
time_pos_CV19_list = [];
time_neg_CV19_list = [];
time_pos_ideal_list = [];
time_neg_ideal_list = []; 
%% FIFO
rng(rand_seed); 
for day_ind = 1: num_days
    % refresh radiologists' workload 
    radiologist_workload = zeros(1, num_radiologists); 
    working_list = {};
    working_list_label = [];
    working_list_score = [];
    start_time = [];
    if day_ind == num_days
        patient_day = patient_list(StudyDate>=StudyDate_order(day_ind),:); 
    else
        patient_day = patient_list(StudyDate==StudyDate_order(day_ind),:); 
    end
    ptx_index = randperm(length(patient_day));
    patient_day = patient_day(ptx_index, :);
    CXR_time = sort(round(rand(1,length(patient_day))*acquisition_time*60), 'ascend'); % randomly assign an event time for these patients 
    
    for check_time = 1: working_time*60 % 1 day 
        if check_time > 24*60 && isempty(working_list)
            break;
        end
        
        for radio = 1: num_radiologists
            if radiologist_workload(radio) > 0
                radiologist_workload(radio) = radiologist_workload(radio)-1;
            end 
        end
        
        time_index = find(CXR_time==check_time); 
        if ~isempty(time_index)
            for time_ind = 1: length(time_index)
                working_list(end+1) = patient_day(time_index(time_ind),1);
                working_list_label = [working_list_label, patient_day{time_index(time_ind),2}];
                working_list_score = [working_list_score, patient_day{time_index(time_ind),3}];
                start_time = [start_time, CXR_time(time_index(time_ind))]; 
            end
        end
    
        if isempty(working_list)
            continue 
        else
            radio_index = find(radiologist_workload<1);
            if ~isempty(radio_index)
                 for radio_ind = 1: min(length(working_list) , length(radio_index))
                     reporting_time = round(normrnd(avgreport_time, stdreport_time));
                     if reporting_time < 1
                         reporting_time = 1; 
                     end
                     radiologist_workload(radio_index(radio_ind)) = radiologist_workload(radio_index(radio_ind)) + reporting_time; 
                     
                     if working_list_label(1) == 1 % positive 
                         time_pos_FCFS_list = [time_pos_FCFS_list, check_time + reporting_time - start_time(1)]; 
                     else % negative 
                         time_neg_FCFS_list = [time_neg_FCFS_list, check_time + reporting_time - start_time(1)]; 
                     end
                     working_list(1) = [];
                     working_list_label(1) = [];
                     working_list_score(1) = [];
                     start_time(1) = []; 
                 end
            end
        end
    end
end


%% CV19-Net
rng(rand_seed); 
for day_ind = 1: num_days
    % refresh radiologists' workload 
    radiologist_workload = zeros(1, num_radiologists); 
    working_list = {};
    working_list_label = [];
    working_list_score = [];
    start_time = [];
    if day_ind == num_days
        patient_day = patient_list(StudyDate>=StudyDate_order(day_ind),:); 
    else
        patient_day = patient_list(StudyDate==StudyDate_order(day_ind),:); 
    end
    ptx_index = randperm(length(patient_day));
    patient_day = patient_day(ptx_index, :);
    CXR_time = sort(round(rand(1,length(patient_day))*acquisition_time*60), 'ascend'); % randomly assign an event time for these patients 
    
    for check_time = 1: working_time*60 % 1 day 
        if check_time > 24*60 && isempty(working_list)
            break;
        end
        
        for radio = 1: num_radiologists
            if radiologist_workload(radio) > 0
                radiologist_workload(radio) = radiologist_workload(radio)-1;
            end 
        end
        
        time_index = find(CXR_time==check_time); 
        if ~isempty(time_index)
            for time_ind = 1: length(time_index)
                working_list(end+1) = patient_day(time_index(time_ind),1);
                working_list_label = [working_list_label, patient_day{time_index(time_ind),2}];
                working_list_score = [working_list_score, patient_day{time_index(time_ind),3}];
                start_time = [start_time, CXR_time(time_index(time_ind))]; 
                [working_list_score, sort_index] = sort(working_list_score,'descend'); 
                working_list = working_list(sort_index); 
                working_list_label = working_list_label(sort_index);
                start_time = start_time(sort_index); 
            end
        end
    
        if isempty(working_list)
            continue 
        else
            radio_index = find(radiologist_workload<1);
            if ~isempty(radio_index)
                 for radio_ind = 1: min(length(working_list) , length(radio_index))
                     reporting_time = round(normrnd(avgreport_time, stdreport_time));
                     if reporting_time < 1
                         reporting_time = 1; 
                     end
                     radiologist_workload(radio_index(radio_ind)) = radiologist_workload(radio_index(radio_ind)) + reporting_time; 
                     
                     if working_list_label(1) == 1 % positive 
                         time_pos_CV19_list = [time_pos_CV19_list, check_time + reporting_time - start_time(1)]; 
                     else % negative 
                         time_neg_CV19_list = [time_neg_CV19_list, check_time + reporting_time - start_time(1)]; 
                     end
                     working_list(1) = [];
                     working_list_label(1) = [];
                     working_list_score(1) = [];
                     start_time(1) = []; 
                 end
            end
        end
    end
end

%% Ideal 
rng(rand_seed); 
for day_ind = 1: num_days
    % refresh radiologists' workload 
    radiologist_workload = zeros(1, num_radiologists); 
    working_list = {};
    working_list_label = [];
    working_list_score = [];
    start_time = [];
    if day_ind == num_days
        patient_day = patient_list(StudyDate>=StudyDate_order(day_ind),:); 
    else
        patient_day = patient_list(StudyDate==StudyDate_order(day_ind),:); 
    end
    ptx_index = randperm(length(patient_day));
    patient_day = patient_day(ptx_index, :);
    CXR_time = sort(round(rand(1,length(patient_day))*acquisition_time*60), 'ascend'); % randomly assign an event time for these patients 
    
    for check_time = 1: working_time*60 % 1 day 
        if check_time > 24*60 && isempty(working_list)
            break;
        end
        
        for radio = 1: num_radiologists
            if radiologist_workload(radio) > 0
                radiologist_workload(radio) = radiologist_workload(radio)-1;
            end 
        end
        
        time_index = find(CXR_time==check_time); 
        if ~isempty(time_index)
            for time_ind = 1: length(time_index)
                working_list(end+1) = patient_day(time_index(time_ind),1);
                working_list_label = [working_list_label, patient_day{time_index(time_ind),2}];
                working_list_score = [working_list_score, patient_day{time_index(time_ind),3}];
                start_time = [start_time, CXR_time(time_index(time_ind))]; 
                [working_list_label, sort_index] = sort(working_list_label,'descend'); 
                working_list = working_list(sort_index); 
                working_list_score = working_list_score(sort_index);
                start_time = start_time(sort_index); 
            end
        end
    
        if isempty(working_list)
            continue 
        else
            radio_index = find(radiologist_workload<1);
            if ~isempty(radio_index)
                 for radio_ind = 1: min(length(working_list) , length(radio_index))
                     reporting_time = round(normrnd(avgreport_time, stdreport_time));
                     if reporting_time < 1
                         reporting_time = 1; 
                     end
                     radiologist_workload(radio_index(radio_ind)) = radiologist_workload(radio_index(radio_ind)) + reporting_time; 
                     
                     if working_list_label(1) == 1 % positive 
                         time_pos_ideal_list = [time_pos_ideal_list, check_time + reporting_time - start_time(1)]; 
                     else % negative 
                         time_neg_ideal_list = [time_neg_ideal_list, check_time + reporting_time - start_time(1)]; 
                     end
                     working_list(1) = [];
                     working_list_label(1) = [];
                     working_list_score(1) = [];
                     start_time(1) = []; 
                 end
            end
        end
    end
end

        
%%
fprintf('With CV19-Net: average waiting time for positive patients %.2f +- %.2f min\n', mean(time_pos_CV19_list), std(time_pos_CV19_list));  
fprintf('With CV19-Net: average waiting time for negative patients %.2f +- %.2f min\n', mean(time_neg_CV19_list), std(time_neg_CV19_list)); 
fprintf('With CV19-Net: median waiting time for positive patients %.2f (%.2f) min\n', median(time_pos_CV19_list), max(time_pos_CV19_list)); 
fprintf('With CV19-Net: median waiting time for negative patients %.2f (%.2f) min\n', median(time_neg_CV19_list), max(time_neg_CV19_list)); 

fprintf('First come, first served: average waiting time for positive patients %.2f +- %.2f min\n', mean(time_pos_FCFS_list), std(time_pos_FCFS_list));       
fprintf('First come, first served: average waiting time for negative patients %.2f +- %.2f min\n', mean(time_neg_FCFS_list), std(time_neg_FCFS_list));  
fprintf('First come, first served: median waiting time for positive patients %.2f (%.2f) min\n', median(time_pos_FCFS_list), max(time_pos_FCFS_list));       
fprintf('First come, first served: median waiting time for negative patients %.2f (%.2f) min\n', median(time_neg_FCFS_list), max(time_neg_FCFS_list)); 
% fprintf('Ideal: average waiting time for positive patients %.2f +- %.2f min\n', mean(time_pos_ideal_list), std(time_pos_ideal_list));  
% fprintf('Ideal: average waiting time for negative patients %.2f +- %.2f min\n', mean(time_neg_ideal_list), std(time_neg_ideal_list)); 

[h_pos, p_pos] = ttest2(time_pos_FCFS_list, time_pos_CV19_list); 
[h_neg, p_neg] = ttest2(time_neg_FCFS_list, time_neg_CV19_list); 
fprintf('FCFS / With CV19-Net (positive): p-value %.5f \n', p_pos);  
fprintf('FCFS / With CV19-Net (negative): p-value %.5f \n', p_neg); 
% [~, p_pos] = ttest2(time_pos_ideal_list, time_pos_CV19_list); 
% [~, p_neg] = ttest2(time_neg_ideal_list, time_neg_CV19_list); 
% fprintf('Ideal / With CV19-Net (positive): p-value %.5f \n', p_pos);  
% fprintf('Ideal / With CV19-Net (negative): p-value %.5f \n', p_neg); 

r = MgSetFigureTheme("dark");
edges = 0:20:1000;
figure(1); 
histogram(time_pos_CV19_list, edges); hold on; histogram(time_pos_FCFS_list, edges); %hold on; histogram(time_pos_ideal_list, 100); 
xlabel('Report Turnaround Time (RTAT) (min)'); ylabel('Counts');
legend('CV19-Net','FIFO')
figure(2); 
histogram(time_neg_CV19_list, edges); hold on; histogram(time_neg_FCFS_list, edges); %hold on; histogram(time_neg_ideal_list, 100); 
xlabel('Report Turnaround Time (RTAT) (min)'); ylabel('Counts');
legend('CV19-Net','FIFO')

