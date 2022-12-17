
%%
clear; 
clc; 
T = readtable('./HF_Oct_info_simulation.csv'); % delta<1, first CXR
Filename = T.Filename; 
label_positive = T.label_positive; 
score_positive = T.score_positive; 
opacity = T.Opacity;
Acc = T.Acc;
StudyDate = round(T.Date_oct);
StudyTime = round(T.Time_oct); 
patient_list = [Acc, num2cell(label_positive), num2cell(score_positive), num2cell(StudyTime)]; 
StudyDate_order = unique(StudyDate); 

%% Parameters 
num_days = 31; % days in Oct
avgreport_time = 15;  % min
stdreport_time = 2; % min 
working_time = 30; % h 
max_waittime =  300; % min 
covid_thres = 0.7;
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
    [~, ptx_index] = sort(cell2mat(patient_day(:,4)), 'ascend'); 
    patient_day = patient_day(ptx_index, :);
    CXR_time = cell2mat(patient_day(:,4)); 
    
    for check_time = 1: working_time*60 % 30h * 60min 
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
    [~, ptx_index] = sort(cell2mat(patient_day(:,4)), 'ascend'); 
    patient_day = patient_day(ptx_index, :);
    CXR_time = cell2mat(patient_day(:,4)); 
    
    for check_time = 1: working_time*60 % 30h * 60min 
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
                
                working_list_pos = working_list(working_list_score>covid_thres); 
                working_list_label_pos = working_list_label(working_list_score>covid_thres); 
                working_list_score_pos = working_list_score(working_list_score>covid_thres); 
                start_time_pos = start_time(working_list_score>covid_thres); 
                working_list_neg = working_list(working_list_score<covid_thres); 
                working_list_label_neg = working_list_label(working_list_score<covid_thres); 
                working_list_score_neg = working_list_score(working_list_score<covid_thres); 
                start_time_neg = start_time(working_list_score<covid_thres); 
                
                [start_time_pos, sort_index] = sort(start_time_pos,'ascend'); 
                working_list_pos = working_list_pos(sort_index); 
                working_list_label_pos = working_list_label_pos(sort_index);
                working_list_score_pos = working_list_score_pos(sort_index); 
                
                [start_time_neg, sort_index] = sort(start_time_neg,'ascend'); 
                working_list_neg = working_list_neg(sort_index); 
                working_list_label_neg = working_list_label_neg(sort_index);
                working_list_score_neg = working_list_score_neg(sort_index); 
                
                working_list = [working_list_pos, working_list_neg]; 
                working_list_label = [working_list_label_pos, working_list_label_neg];
                working_list_score = [working_list_score_pos, working_list_score_neg];
                start_time = [start_time_pos, start_time_neg];
                
                %% < maximal waitting time
                working_list_long_wait = working_list((check_time - start_time) > max_waittime);
                working_list_label_long_wait = working_list_label((check_time - start_time) > max_waittime);
                working_list_score_long_wait = working_list_score((check_time - start_time) > max_waittime);
                start_time_long_wait = start_time((check_time - start_time) > max_waittime);
                
                [start_time_long_wait, sort_index] = sort(start_time_long_wait,'ascend'); 
                working_list_long_wait = working_list_long_wait(sort_index); 
                working_list_label_long_wait = working_list_label_long_wait(sort_index);
                working_list_score_long_wait = working_list_score_long_wait(sort_index);
                
                
                working_list((check_time - start_time) > max_waittime) = []; 
                working_list_label((check_time - start_time) > max_waittime) = [];
                working_list_score((check_time - start_time) > max_waittime) = []; 
                start_time((check_time - start_time) > max_waittime) = [];
                
                working_list = [working_list_long_wait, working_list]; 
                working_list_label = [working_list_label_long_wait, working_list_label];
                working_list_score = [working_list_score_long_wait, working_list_score];
                start_time = [start_time_long_wait, start_time];
                
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
    [~, ptx_index] = sort(cell2mat(patient_day(:,4)), 'ascend'); 
    patient_day = patient_day(ptx_index, :);
    CXR_time = cell2mat(patient_day(:,4)); 
    
    for check_time = 1: working_time*60 % 30h * 60min 
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
                
                working_list_pos = working_list(working_list_label>0.5); 
                working_list_label_pos = working_list_label(working_list_label>0.5); 
                working_list_score_pos = working_list_score(working_list_label>0.5); 
                start_time_pos = start_time(working_list_label>0.5); 
                working_list_neg = working_list(working_list_label<0.5); 
                working_list_label_neg = working_list_label(working_list_label<0.5); 
                working_list_score_neg = working_list_score(working_list_label<0.5); 
                start_time_neg = start_time(working_list_label<0.5); 
                
                [start_time_pos, sort_index] = sort(start_time_pos,'ascend'); 
                working_list_pos = working_list_pos(sort_index); 
                working_list_label_pos = working_list_label_pos(sort_index);
                working_list_score_pos = working_list_score_pos(sort_index); 
                
                [start_time_neg, sort_index] = sort(start_time_neg,'ascend'); 
                working_list_neg = working_list_neg(sort_index); 
                working_list_label_neg = working_list_label_neg(sort_index);
                working_list_score_neg = working_list_score_neg(sort_index); 
                
                working_list = [working_list_pos, working_list_neg]; 
                working_list_label = [working_list_label_pos, working_list_label_neg];
                working_list_score = [working_list_score_pos, working_list_score_neg];
                start_time = [start_time_pos, start_time_neg];
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

        

%% box plot 
close all; 

col=@(x)reshape(x,numel(x),1);
boxplot2=@(C,varargin)boxplot(cell2mat(cellfun(col,col(C),'uni',0)),cell2mat(arrayfun(@(I)I*ones(numel(C{I}),1),col(1:numel(C)),'uni',0)),varargin{:});

fprintf('With CV19-Net: average waiting time for positive patients %.2f +- %.2f min\n', mean(time_pos_CV19_list), std(time_pos_CV19_list));  
fprintf('With CV19-Net: average waiting time for negative patients %.2f +- %.2f min\n', mean(time_neg_CV19_list), std(time_neg_CV19_list)); 
fprintf('With CV19-Net: median waiting time for positive patients %.2f (%.2f) min\n', median(time_pos_CV19_list), iqr(time_pos_CV19_list)); 
fprintf('With CV19-Net: median waiting time for negative patients %.2f (%.2f) min\n', median(time_neg_CV19_list), iqr(time_neg_CV19_list)); 

fprintf('First come, first served: average waiting time for positive patients %.2f +- %.2f min\n', mean(time_pos_FCFS_list), std(time_pos_FCFS_list));       
fprintf('First come, first served: average waiting time for negative patients %.2f +- %.2f min\n', mean(time_neg_FCFS_list), std(time_neg_FCFS_list));  
fprintf('First come, first served: median waiting time for positive patients %.2f (%.2f) min\n', median(time_pos_FCFS_list), iqr(time_pos_FCFS_list));       
fprintf('First come, first served: median waiting time for negative patients %.2f (%.2f) min\n', median(time_neg_FCFS_list), iqr(time_neg_FCFS_list)); 

[h_pos, p_pos] = ttest2(time_pos_FCFS_list, time_pos_CV19_list); 
[h_neg, p_neg] = ttest2(time_neg_FCFS_list, time_neg_CV19_list); 
fprintf('FCFS / With CV19-Net (positive): p-value %.5f \n', p_pos);  
fprintf('FCFS / With CV19-Net (negative): p-value %.5f \n', p_neg); 

r = MgSetFigureTheme("light");

edges = 0:20:1000;
figure(1); 
histogram(time_pos_CV19_list, edges); hold on; histogram(time_pos_FCFS_list, edges); %hold on; histogram(time_pos_ideal_list, 100); 
xlabel('RTAT (min)'); ylabel('Number of CXRs');
legend('CV19-Net','FIFO');  title('COVID-19'); 
figure(2); 
histogram(time_neg_CV19_list, edges); hold on; histogram(time_neg_FCFS_list, edges); %hold on; histogram(time_neg_ideal_list, 100); 
xlabel('RTAT (min)'); ylabel('Number of CXRs');
legend('CV19-Net','FIFO'); title('Non-COVID-19'); 


figure(3); 
time_pos = cell(1);
time_pos{1} =  time_pos_CV19_list;
time_pos{2} =  time_pos_FCFS_list;
boxplot2(time_pos); xlabel('Left: CV19-Net; Right FIFO'); ylabel('RTAT (min)'); title('COVID-19'); %ylim([0, 650]); 
figure(4); 
time_neg = cell(1);
time_neg{1} =  time_neg_CV19_list;
time_neg{2} =  time_neg_FCFS_list;
boxplot2(time_neg); xlabel('CV19-Net / FIFO'); ylabel('RTAT (min)'); title('Non-COVID-19'); %ylim([0, 700]);
